# lora-runtime

One base model, many LoRA adapters, four serving patterns:

| Pattern | API | What it does |
|---|---|---|
| Zero-latency switching | `rt.activate(node)` | Merges `s * B @ A` into the base weights. Forward pass is exactly the base model: no hooks, no extra matmuls. Switching subtracts outgoing deltas and adds incoming ones in fp32, then rounds once. |
| RAM-resident cache | `AdapterCache` | Adapters live in host RAM (LRU, byte budget). Serving paths never read disk: a cold adapter raises `CacheMiss` instead of stalling. Hot set is kept on the device; host copies are pinned on CUDA for async transfer. |
| Multi-LoRA batching | `with rt.batch([...])` | Each batch row goes through its own adapter path in one forward pass. Base matmul runs once; per-route low-rank terms are gathered and scattered (Punica/S-LoRA segmented approach, pure PyTorch). Works for training too: gradients reach only the adapters used in the batch. |
| Hierarchical tree | `LoRATree` | Base at the root, then language → domain → task/user adapters. A node's model is base + sum of the path. Sibling switches touch only the leaf. Warns if a child's rank exceeds its parent's. |

Merged and batched modes compose. Merge the shared trunk (e.g. `domain`), then
batch the leaves: each row pays only for its leaf, and a row that wants the
plain base model gets the trunk subtracted.

## Quick start

```python
from lora_runtime import AdapterCache, LoRARuntime, LoRATree

cache = AdapterCache(ram_budget_bytes=16 << 30)
cache.preload("adapters/")          # every PEFT adapter dir; the only disk reads

tree = LoRATree()
tree.add("medtech", rank=32)                     # domain
tree.add("dfmea", parent="medtech", rank=8)      # task
tree.add("redlines", parent="medtech", rank=8)   # task

rt = LoRARuntime(model, cache, tree)  # model: any torch model with nn.Linear / HF Conv1D targets

rt.activate("dfmea")       # base + medtech + dfmea merged, base-model speed
rt.activate("redlines")    # only the leaf changes
rt.activate(None)          # back to the plain base model

rt.activate("medtech")     # keep the trunk merged...
with rt.batch(["dfmea", "redlines", None]):   # ...and route rows per task
    logits = model(input_ids).logits          # input_ids: [3, T]
```

Adapters are read and written in Hugging Face PEFT format
(`adapter_config.json` + `adapter_model.safetensors`), so anything trained with
PEFT loads directly. `compose(path)` folds a whole tree path into one exact
adapter of rank `sum(r_i)` for export.

Restore modes for `activate`:

- `restore="delta"` (default): no extra memory, one rounding step per switch.
  The bf16 test bounds drift after 200 switches.
- `restore="snapshot"`: keeps the original touched weights in host RAM and
  rebuilds exactly. Bit-identical after any number of switches. It doubles
  memory for the target layers, which costs more on unified memory.

## Measured (CPU, fp32, random LLaMA-shaped model, q/v targets)

`python benchmarks/bench_switch.py [--device mps|cuda] [--hidden N --seq N]`

| Shape | Forward (prefill) | Leaf switch | Full path switch | Disk fetch | RAM fetch |
|---|---|---|---|---|---|
| h=1024, 8 layers, 256 tok | 205 ms | 9.0 ms | 9.7 ms | 1.2 ms* | 0.02 ms |
| h=2048, 8 layers, 512 tok | 1502 ms | 111 ms | 135 ms | 1.3 ms* | 0.02 ms |

\* Warm OS page cache. A cold read from SSD is slower, and from network storage
much slower.

A switch is roughly 10 to 20 times cheaper than a prefill forward pass. It is
**not** cheaper than a single decode step, so switch between requests, not in
the middle of generation. Run the benchmark on your own hardware before
trusting these ratios.

## Platform notes (Apple Silicon + Ollama)

- **Ollama cannot hot-swap weights.** It runs GGUF through llama.cpp, and a
  Modelfile `ADAPTER` is applied when the model loads. No API adds deltas to a
  running Ollama model. This runtime therefore runs in PyTorch (`mps` device
  on an M1 Max), next to Ollama rather than inside it. To bring a tree path
  into Ollama: `compose(path).save_peft(dir)`, convert it with llama.cpp's
  `convert_lora_to_gguf.py`, then add `ADAPTER` to a Modelfile, one model tag
  per path. Unsure whether Ollama shares base weights in memory across two
  loaded tags. Check with `ollama ps` before relying on it.
- **Unified memory has no RAM to VRAM transfer.** CPU and GPU share one pool
  on an M1 Max, and pinning does not apply. The cache still matters because it
  removes the disk read, and the device tier just avoids re-creating MPS
  buffers.
- **S-LoRA and Punica are CUDA kernels.** They do not run on a Mac. `batch()`
  uses plain PyTorch ops, so it runs on MPS, CUDA and CPU. With many distinct
  adapters per batch it is slower than those custom kernels.

## Tests

```bash
pip install -e '.[test,peft]'
pytest
```

The suite checks:

- Merged and batched outputs against an fp64 reference.
- LLaMA (`nn.Linear`) and GPT-2 (`Conv1D`) against PEFT itself, including
  rsLoRA, a mixed batch with the trunk merged, and a composed export loaded
  back through PEFT.
- Snapshot exactness and bf16 drift bounds.
- That cache serving paths never read disk.
- That trainable adapters get gradients only when routed.

Not covered yet: `rank_pattern`/`alpha_pattern` PEFT configs (load raises),
DoRA, quantized base weights (4/8-bit), and MLX.
