"""LoRA adapter container with PEFT-compatible load/save.

An adapter is a set of per-module low-rank pairs (A, B). For a target linear
layer with weight W of shape [out, in], the adapter's contribution is

    delta_W = scaling * B @ A        A: [r, in], B: [out, r]

Because the update is purely additive, adapters can be merged into W, removed
again, stacked (tree paths), or applied unmerged per batch row.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

PEFT_PREFIX = "base_model.model."
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass
class LoRAAdapter:
    """One LoRA module set. ``weights`` maps target module name -> (A, B)."""

    name: str
    rank: int
    alpha: float
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank

    @property
    def modules(self) -> list[str]:
        return list(self.weights)

    @property
    def nbytes(self) -> int:
        return sum(a.numel() * a.element_size() + b.numel() * b.element_size()
                   for a, b in self.weights.values())

    @property
    def device(self) -> torch.device:
        a, _ = next(iter(self.weights.values()))
        return a.device

    def delta(self, module: str, dtype: torch.dtype = torch.float32,
              device: torch.device | str | None = None) -> torch.Tensor:
        """Dense delta_W [out, in] for one module, computed in ``dtype`` (fp32 by default)."""
        a, b = self.weights[module]
        return (b.to(device, dtype) @ a.to(device, dtype)) * self.scaling

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "LoRAAdapter":
        """Copy to ``device``. Returns a new adapter; the original is untouched."""
        moved = {
            m: (a.to(device, non_blocking=non_blocking), b.to(device, non_blocking=non_blocking))
            for m, (a, b) in self.weights.items()
        }
        return LoRAAdapter(self.name, self.rank, self.alpha, moved, self.use_rslora)

    def pin_memory(self) -> "LoRAAdapter":
        """Page-lock host tensors so host->GPU copies can run async (CUDA only)."""
        pinned = {m: (a.pin_memory(), b.pin_memory()) for m, (a, b) in self.weights.items()}
        return LoRAAdapter(self.name, self.rank, self.alpha, pinned, self.use_rslora)

    # ------------------------------------------------------------------ PEFT I/O

    @classmethod
    def load_peft(cls, path: str | Path, name: str | None = None) -> "LoRAAdapter":
        """Load a PEFT LoRA directory (adapter_config.json + adapter_model.safetensors)."""
        from safetensors.torch import load_file

        path = Path(path)
        config = json.loads((path / CONFIG_NAME).read_text())
        if config.get("rank_pattern") or config.get("alpha_pattern"):
            raise NotImplementedError(
                "per-module rank_pattern/alpha_pattern adapters are not supported yet"
            )
        state = load_file(str(path / WEIGHTS_NAME))
        weights: dict[str, list[torch.Tensor | None]] = {}
        for key, tensor in state.items():
            for part, slot in ((".lora_A.weight", 0), (".lora_B.weight", 1)):
                if key.endswith(part):
                    module = key[: -len(part)]
                    if module.startswith(PEFT_PREFIX):
                        module = module[len(PEFT_PREFIX):]
                    weights.setdefault(module, [None, None])[slot] = tensor
                    break
            else:
                raise ValueError(f"unexpected tensor in adapter {path}: {key}")
        incomplete = [m for m, pair in weights.items() if None in pair]
        if incomplete:
            raise ValueError(f"adapter {path} is missing A or B for: {incomplete}")
        return cls(
            name=name or path.name,
            rank=int(config["r"]),
            alpha=float(config["lora_alpha"]),
            weights={m: (a, b) for m, (a, b) in weights.items()},
            use_rslora=bool(config.get("use_rslora", False)),
        )

    def save_peft(self, path: str | Path, base_model_name: str | None = None) -> None:
        """Write a PEFT-format adapter. llama.cpp's convert_lora_to_gguf.py reads this."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {}
        for module, (a, b) in self.weights.items():
            state[f"{PEFT_PREFIX}{module}.lora_A.weight"] = a.detach().contiguous().cpu()
            state[f"{PEFT_PREFIX}{module}.lora_B.weight"] = b.detach().contiguous().cpu()
        save_file(state, str(path / WEIGHTS_NAME))
        leaf_names = sorted({m.rsplit(".", 1)[-1] for m in self.weights})
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name,
            "r": self.rank,
            "lora_alpha": self.alpha,
            "use_rslora": self.use_rslora,
            "target_modules": leaf_names,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
        }
        (path / CONFIG_NAME).write_text(json.dumps(config, indent=2))


def compose(adapters: list[LoRAAdapter], name: str = "composed") -> LoRAAdapter:
    """Fold a stack of adapters (e.g. a tree path) into one exact LoRA.

    sum_i s_i * B_i @ A_i == [s_1 B_1 | s_2 B_2 | ...] @ [A_1; A_2; ...]

    so the result has rank sum(r_i) and scaling 1. Modules missing from an
    adapter get zero blocks. Useful for exporting a whole path as one file
    (e.g. to convert to GGUF for an Ollama ``ADAPTER`` line).
    """
    if not adapters:
        raise ValueError("compose needs at least one adapter")
    modules = sorted({m for ad in adapters for m in ad.weights})
    total_rank = sum(ad.rank for ad in adapters)
    weights = {}
    for module in modules:
        ref_a, ref_b = next(ad.weights[module] for ad in adapters if module in ad.weights)
        in_f, out_f = ref_a.shape[1], ref_b.shape[0]
        blocks_a, blocks_b = [], []
        for ad in adapters:
            if module in ad.weights:
                a, b = ad.weights[module]
                blocks_a.append(a.float())
                blocks_b.append(b.float() * ad.scaling)
            else:
                blocks_a.append(torch.zeros(ad.rank, in_f, device=ref_a.device))
                blocks_b.append(torch.zeros(out_f, ad.rank, device=ref_b.device))
        weights[module] = (torch.cat(blocks_a, 0).to(ref_a.dtype),
                           torch.cat(blocks_b, 1).to(ref_b.dtype))
    return LoRAAdapter(name=name, rank=total_rank, alpha=float(total_rank), weights=weights)
