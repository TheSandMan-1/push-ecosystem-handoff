"""Measure switch cost vs one forward pass, and disk vs RAM adapter fetch.

    python benchmarks/bench_switch.py --device mps     # Apple Silicon
    python benchmarks/bench_switch.py --device cuda
    python benchmarks/bench_switch.py                  # CPU

Uses a randomly initialised LLaMA-shaped model (needs ``transformers``), so
numbers reflect shapes, not quality.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch

from lora_runtime import AdapterCache, LoRAAdapter, LoRARuntime, LoRATree


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str, repeats: int) -> float:
    fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def random_adapter(model, name, rank, targets, seed):
    g = torch.Generator().manual_seed(seed)
    weights = {}
    for mname, mod in model.named_modules():
        if mname.rsplit(".", 1)[-1] in targets:
            weights[mname] = (torch.randn(rank, mod.in_features, generator=g) * 0.01,
                              torch.randn(mod.out_features, rank, generator=g) * 0.01)
    return LoRAAdapter(name, rank, 2.0 * rank, weights)


def main() -> None:
    import transformers

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)

    cfg = transformers.LlamaConfig(vocab_size=32000, hidden_size=args.hidden,
                                   intermediate_size=args.hidden * 4 // 1,
                                   num_hidden_layers=args.layers,
                                   num_attention_heads=max(1, args.hidden // 64),
                                   num_key_value_heads=max(1, args.hidden // 64))
    model = transformers.LlamaForCausalLM(cfg).to(args.device, dtype).eval()
    targets = {"q_proj", "v_proj"}

    cache = AdapterCache(pin_memory=args.device == "cuda")
    tree = LoRATree()
    tree.add("domain", rank=args.rank * 2)
    for i, name in enumerate(["domain", "task_a", "task_b"]):
        rank = args.rank * 2 if name == "domain" else args.rank
        cache.put(random_adapter(model, name, rank, targets, seed=i))
        if name != "domain":
            tree.add(name, parent="domain", rank=rank)

    rt = LoRARuntime(model, cache, tree)
    ids = torch.randint(0, cfg.vocab_size, (1, args.seq), device=args.device)

    with torch.no_grad():
        fwd = timed(lambda: model(ids), args.device, args.repeats)
    state = {"n": 0}

    def flip():
        state["n"] += 1
        rt.activate("task_a" if state["n"] % 2 else "task_b")

    rt.activate("task_a")
    leaf_switch = timed(flip, args.device, args.repeats)

    def full():
        state["n"] += 1
        rt.activate("task_a" if state["n"] % 2 else None)

    full_switch = timed(full, args.device, args.repeats)

    with tempfile.TemporaryDirectory() as tmp:
        cache.get("task_a").save_peft(Path(tmp) / "task_a")
        disk = timed(lambda: LoRAAdapter.load_peft(Path(tmp) / "task_a"), "cpu", args.repeats)
    ram_to_dev = timed(lambda: cache.get("task_a").to(args.device), args.device, args.repeats)

    mb = cache.get("task_a").nbytes / 1e6
    print(f"device={args.device} dtype={args.dtype} hidden={args.hidden} layers={args.layers} "
          f"seq={args.seq} leaf_rank={args.rank} adapter={mb:.1f} MB")
    print(f"forward pass (1x{args.seq} tokens):        {fwd:8.2f} ms")
    print(f"sibling-leaf switch (trunk stays merged):   {leaf_switch:8.2f} ms")
    print(f"full path switch (trunk + leaf):            {full_switch:8.2f} ms")
    print(f"adapter fetch from disk (safetensors):      {disk:8.2f} ms")
    print(f"adapter fetch RAM -> {args.device:<6}                {ram_to_dev:8.2f} ms")


if __name__ == "__main__":
    main()
