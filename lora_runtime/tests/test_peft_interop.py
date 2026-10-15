"""Cross-check against Hugging Face PEFT on tiny randomly initialised models."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from lora_runtime import AdapterCache, LoRAAdapter, LoRARuntime, LoRATree, compose  # noqa: E402


def tiny_llama():
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def tiny_gpt2():
    cfg = transformers.GPT2Config(vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_positions=64)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def train_like_adapter(base, targets, rank, seed, path, fan_in_fan_out=False, use_rslora=False):
    """Make a PEFT adapter with non-zero B (as after training) and save it."""
    torch.manual_seed(seed)
    cfg = peft.LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets,
                          init_lora_weights=False, fan_in_fan_out=fan_in_fan_out,
                          use_rslora=use_rslora)
    pm = peft.get_peft_model(copy.deepcopy(base), cfg).eval()
    pm.save_pretrained(path)
    return pm


ids = torch.randint(0, 128, (3, 10), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("use_rslora", [False, True])
def test_llama_merged_matches_peft(tmp_path, use_rslora):
    base = tiny_llama()
    pm = train_like_adapter(base, ["q_proj", "v_proj", "down_proj"], 8, 1, tmp_path / "a",
                            use_rslora=use_rslora)
    cache = AdapterCache(pin_memory=False)
    cache.load(tmp_path / "a")
    model = copy.deepcopy(base)
    LoRARuntime(model, cache).activate("a")
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, pm(ids).logits, atol=1e-4, rtol=1e-4)


def test_gpt2_conv1d_merged_and_batched_match_peft(tmp_path):
    base = tiny_gpt2()
    pm = train_like_adapter(base, ["c_attn", "c_proj"], 4, 2, tmp_path / "g", fan_in_fan_out=True)
    cache = AdapterCache(pin_memory=False)
    cache.load(tmp_path / "g")
    model = copy.deepcopy(base)
    rt = LoRARuntime(model, cache)
    with torch.no_grad():
        expected = pm(ids).logits
        with rt.batch(["g", "g", "g"]):
            torch.testing.assert_close(model(ids).logits, expected, atol=1e-4, rtol=1e-4)
        rt.activate("g")
        torch.testing.assert_close(model(ids).logits, expected, atol=1e-4, rtol=1e-4)


def test_llama_tree_mixed_batch_and_composed_export(tmp_path):
    base = tiny_llama()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_models = {}
    for i, (name, rank) in enumerate([("domain", 16), ("task_a", 4), ("task_b", 4)]):
        peft_models[name] = train_like_adapter(base, targets, rank, 10 + i, tmp_path / name)
    cache = AdapterCache(pin_memory=False)
    cache.preload(tmp_path)
    tree = LoRATree()
    tree.add("domain", rank=16)
    tree.add("task_a", parent="domain", rank=4)
    tree.add("task_b", parent="domain", rank=4)

    # Reference for a path: base with every adapter on the path merged by PEFT.
    def peft_path_logits(path):
        ref = copy.deepcopy(base)
        for name in path:
            ref = peft.PeftModel.from_pretrained(ref, tmp_path / name).merge_and_unload()
        with torch.no_grad():
            return ref(ids).logits

    model = copy.deepcopy(base)
    rt = LoRARuntime(model, cache, tree)
    rt.activate("domain")  # keep the shared trunk merged, batch the leaves
    nodes = ["task_a", None, "task_b"]
    with torch.no_grad(), rt.batch(nodes):
        out = model(ids).logits
    refs = {"task_a": peft_path_logits(["domain", "task_a"]),
            None: peft_path_logits([]),
            "task_b": peft_path_logits(["domain", "task_b"])}
    for row, node in enumerate(nodes):
        torch.testing.assert_close(out[row], refs[node][row], atol=1e-4, rtol=1e-4)

    # Export the whole path as one PEFT adapter and load it back through PEFT.
    path = [cache.get(n) for n in tree.path("task_b")]
    compose(path, name="export").save_peft(tmp_path / "export_dir" / "export")
    exported = peft.PeftModel.from_pretrained(copy.deepcopy(base),
                                              tmp_path / "export_dir" / "export").eval()
    with torch.no_grad():
        torch.testing.assert_close(exported(ids).logits, refs["task_b"], atol=1e-4, rtol=1e-4)
