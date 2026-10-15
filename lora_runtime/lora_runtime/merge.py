"""Merged-weight switching: W <- W + sum(s * B @ A), so inference runs at base-model cost.

Two restore strategies:

* ``"delta"`` (default): a switch computes the net change for each touched
  module in fp32 (subtract outgoing deltas, add incoming ones) and rounds into
  the weight dtype once. No extra memory. Each switch adds at most one
  rounding step of drift per touched weight; ``drift_from_snapshot`` measures
  it if you also keep a snapshot, and ``resync`` resets it.
* ``"snapshot"``: keeps a host copy of the original weights of every touched
  module and rebuilds W = W0 + sum(path deltas) on each switch. Exact and
  drift-free, at the cost of host RAM equal to the touched weights. On Apple
  Silicon that RAM is the same pool as the model, so prefer ``"delta"`` there.

All per-module work is queued on the model's device without host syncs, so on
a GPU the kernels pipeline back to back.
"""

from __future__ import annotations

import torch
from torch import nn

from .adapter import LoRAAdapter


def target_weight(module: nn.Module) -> tuple[torch.Tensor, bool]:
    """Return (weight, transposed). HF GPT-2 style Conv1D stores weight as [in, out]."""
    if isinstance(module, nn.Linear):
        weight, transposed = module.weight, False
    elif type(module).__name__ == "Conv1D" and hasattr(module, "weight"):
        weight, transposed = module.weight, True
    else:
        raise TypeError(f"LoRA target must be nn.Linear or Conv1D, got {type(module).__name__}")
    if not weight.is_floating_point():
        # e.g. bitsandbytes Linear4bit/Linear8bitLt subclass nn.Linear but store packed ints.
        raise TypeError(f"{type(module).__name__} has quantized weights; merging needs float weights")
    return weight, transposed


class WeightMerger:
    def __init__(self, model: nn.Module, restore: str = "delta") -> None:
        if restore not in ("delta", "snapshot"):
            raise ValueError("restore must be 'delta' or 'snapshot'")
        self.model = model
        self.restore = restore
        self._modules = dict(model.named_modules())
        self._snapshot: dict[str, torch.Tensor] = {}
        self.merged: list[LoRAAdapter] = []
        self.switch_count = 0

    @property
    def merged_names(self) -> list[str]:
        return [a.name for a in self.merged]

    def module(self, name: str) -> nn.Module:
        try:
            return self._modules[name]
        except KeyError:
            raise KeyError(f"adapter targets {name!r}, which is not a module of this model") from None

    @torch.no_grad()
    def transition(self, target: list[LoRAAdapter]) -> dict[str, int]:
        """Move the merged state from ``self.merged`` to ``target`` (an ordered path).

        Only adapters that differ between the two paths are applied, so a
        sibling-leaf switch under a shared trunk touches only the leaf modules.
        Returns counters for inspection.
        """
        target_names = {a.name for a in target}
        if len(target_names) != len(target):
            raise ValueError("an adapter appears twice in the target path")
        current_names = set(self.merged_names)
        outgoing = [a for a in self.merged if a.name not in target_names]
        incoming = [a for a in target if a.name not in current_names]
        changes = [(a, -1.0) for a in outgoing] + [(a, 1.0) for a in incoming]
        touched = sorted({m for a, _ in changes for m in a.weights})
        self._validate(changes)

        for name in touched:
            module = self.module(name)
            weight, transposed = target_weight(module)
            if self.restore == "snapshot":
                if name not in self._snapshot:
                    self._snapshot[name] = weight.detach().to("cpu", copy=True)
                acc = self._snapshot[name].to(weight.device, torch.float32, copy=True)
                terms = [(a, 1.0) for a in target if name in a.weights]
            else:
                acc = weight.detach().float()
                terms = [(a, c) for a, c in changes if name in a.weights]
            for adapter, coef in terms:
                delta = adapter.delta(name, device=weight.device)
                if transposed:
                    delta = delta.T
                acc.add_(delta, alpha=coef)
            weight.copy_(acc)

        self.merged = list(target)
        if changes:
            self.switch_count += 1
        return {"removed": len(outgoing), "added": len(incoming), "modules": len(touched)}

    def _validate(self, changes: list[tuple[LoRAAdapter, float]]) -> None:
        """Check every module and shape before any weight is modified."""
        for adapter, _ in changes:
            for name, (a, b) in adapter.weights.items():
                weight, transposed = target_weight(self.module(name))
                out_f, in_f = weight.shape[::-1] if transposed else weight.shape
                if a.shape != (adapter.rank, in_f) or b.shape != (out_f, adapter.rank):
                    raise ValueError(
                        f"adapter {adapter.name} module {name}: A{tuple(a.shape)} B{tuple(b.shape)} "
                        f"do not fit weight [out={out_f}, in={in_f}] at rank {adapter.rank}"
                    )

    @torch.no_grad()
    def resync(self) -> None:
        """Rebuild touched weights exactly from the snapshot (snapshot mode only)."""
        if self.restore != "snapshot":
            raise RuntimeError("resync needs restore='snapshot'")
        current, self.merged = self.merged, []
        for name, original in self._snapshot.items():
            weight, _ = target_weight(self.module(name))
            weight.copy_(original.to(weight.device, weight.dtype))
        self.transition(current)

    @torch.no_grad()
    def drift_from_snapshot(self) -> float:
        """Max |W - (W0 + sum deltas)| over snapshotted modules. Diagnostic only."""
        worst = 0.0
        for name, original in self._snapshot.items():
            weight, transposed = target_weight(self.module(name))
            expected = original.to(weight.device, torch.float32, copy=True)
            for adapter in self.merged:
                if name in adapter.weights:
                    d = adapter.delta(name, device=weight.device)
                    expected += d.T if transposed else d
            expected = expected.to(weight.dtype).float()
            worst = max(worst, (weight.float() - expected).abs().max().item())
        return worst

