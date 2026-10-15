import copy
import warnings

import pytest
import torch

from lora_runtime import AdapterCache, CacheMiss, LoRAAdapter, LoRARuntime, LoRATree, compose

from .conftest import HIDDEN, TARGETS, ToyModel, make_adapter, reference_merge


def build_tree(cache):
    """base -> lang(r16) -> domain(r8) -> {task(r4), user(r2)}; plus other_lang(r16)."""
    tree = LoRATree()
    specs = [("lang", None, 16), ("domain", "lang", 8), ("task", "domain", 4),
             ("user", "domain", 2), ("other_lang", None, 16)]
    for i, (name, parent, rank) in enumerate(specs):
        cache.put(make_adapter(name, rank, seed=i + 1))
        tree.add(name, parent=parent, rank=rank)
    return tree


x = torch.randn(3, 5, HIDDEN, generator=torch.Generator().manual_seed(42))


# ---------------------------------------------------------------- merged switching

def test_merged_matches_reference(model, cache):
    tree = build_tree(cache)
    base = copy.deepcopy(model)
    rt = LoRARuntime(model, cache, tree)
    rt.activate("task")
    ref = reference_merge(base, [cache.get(n) for n in ["lang", "domain", "task"]])
    with torch.no_grad():
        torch.testing.assert_close(model(x), ref(x.double()).float(), atol=1e-5, rtol=1e-5)


def test_merged_mode_installs_no_hooks(model, cache):
    tree = build_tree(cache)
    rt = LoRARuntime(model, cache, tree)
    rt.activate("user")
    for m in model.modules():
        assert not m._forward_hooks and not m._forward_pre_hooks


def test_sibling_switch_only_touches_leaf(model, cache):
    tree = build_tree(cache)
    rt = LoRARuntime(model, cache, tree)
    rt.activate("task")
    stats = rt.activate("user")
    assert stats == {"removed": 1, "added": 1, "modules": len(TARGETS)}
    assert rt.merger.merged_names == ["lang", "domain", "user"]


def test_switch_equals_fresh_merge(model, cache):
    tree = build_tree(cache)
    fresh = copy.deepcopy(model)
    rt = LoRARuntime(model, cache, tree)
    for node in ["task", "other_lang", "user", "task", None, "domain"]:
        rt.activate(node)
    LoRARuntime(fresh, cache, tree).activate("domain")
    with torch.no_grad():
        torch.testing.assert_close(model(x), fresh(x), atol=1e-5, rtol=1e-5)


def test_snapshot_restore_is_exact(model, cache):
    tree = build_tree(cache)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    rt = LoRARuntime(model, cache, tree, restore="snapshot")
    for _ in range(50):
        rt.activate("task")
        rt.activate("other_lang")
    rt.deactivate()
    for k, v in model.state_dict().items():
        assert torch.equal(v, original[k]), k


def test_delta_drift_is_bounded_in_bf16(cache):
    """bf16 weights: many delta-mode switches, drift stays at rounding scale."""
    torch.manual_seed(0)
    model = ToyModel().to(torch.bfloat16)
    tree = build_tree(cache)
    rt = LoRARuntime(model, cache, tree, restore="delta")
    snap = LoRARuntime(copy.deepcopy(model), cache, tree, restore="snapshot")
    for _ in range(100):
        rt.activate("task")
        rt.activate("user")
    snap.activate("user")
    diffs = [(a.float() - b.float()).abs().max().item()
             for a, b in zip(model.parameters(), snap.model.parameters())]
    # bf16 has ~3 significant digits; weights here are O(0.1-1). 200 roundings
    # of a random walk stay within a few dozen ulps.
    assert max(diffs) < 0.05


def test_resync_clears_drift(cache):
    torch.manual_seed(0)
    model = ToyModel().to(torch.bfloat16)
    tree = build_tree(cache)
    rt = LoRARuntime(model, cache, tree, restore="snapshot")
    rt.activate("task")
    rt.merger.resync()
    assert rt.merger.drift_from_snapshot() == 0.0


# ---------------------------------------------------------------- tree

def test_tree_path_and_rank_warning():
    tree = LoRATree()
    tree.add("lang", rank=16)
    tree.add("domain", parent="lang", rank=8)
    assert tree.path("domain") == ["lang", "domain"]
    assert tree.path(None) == []
    with pytest.warns(UserWarning, match="ranks usually shrink"):
        tree.add("leaf", parent="domain", rank=32)
    with pytest.raises(KeyError):
        tree.add("orphan", parent="missing")


# ---------------------------------------------------------------- cache

def test_cache_never_reads_disk_on_serving_path(tmp_path, model):
    for i in range(5):
        make_adapter(f"a{i}", 4, seed=i).save_peft(tmp_path / f"a{i}")
    cache = AdapterCache(pin_memory=False)
    assert cache.preload(tmp_path) == [f"a{i}" for i in range(5)]
    reads = cache.stats.disk_reads
    rt = LoRARuntime(model, cache)
    for i in range(20):
        rt.activate(f"a{i % 5}")
    assert cache.stats.disk_reads == reads == 5
    with pytest.raises(CacheMiss):
        rt.activate("not_loaded")


def test_cache_lru_respects_ram_budget():
    one = make_adapter("x", 4)
    cache = AdapterCache(ram_budget_bytes=one.nbytes * 3, pin_memory=False)
    for i in range(5):
        cache.put(make_adapter(f"a{i}", 4, seed=i))
    cache.get("a2")  # touch so a2 survives the next eviction
    cache.put(make_adapter("a5", 4, seed=5))
    assert cache.names() == ["a4", "a2", "a5"]
    assert cache.stats.ram_evictions == 3
    assert cache.ram_bytes <= cache.ram_budget_bytes


def test_cache_device_tier_counts_transfers():
    cache = AdapterCache(pin_memory=False, device_slots=2)
    for i in range(3):
        cache.put(make_adapter(f"a{i}", 4, seed=i))
    device = "meta"  # stand-in non-CPU device; transfers are logged without real hardware
    cache.on_device("a0", device)
    cache.on_device("a0", device)
    cache.on_device("a1", device)
    cache.on_device("a2", device)  # evicts a0 from the device tier
    assert cache.stats.device_transfers == 3
    assert cache.stats.device_hits == 1
    assert cache.stats.device_evictions == 1


# ---------------------------------------------------------------- batched

def per_row_reference(model, cache, tree, nodes):
    outs = []
    for i, node in enumerate(nodes):
        path = tree.path(node) if node in tree else ([node] if node else [])
        ref = reference_merge(model, [cache.get(n) for n in path])
        with torch.no_grad():
            outs.append(ref(x[i:i + 1].double()).float())
    return torch.cat(outs)


@pytest.mark.parametrize("merged_first", [None, "domain", "other_lang"])
def test_mixed_batch_matches_per_row_merge(model, cache, merged_first):
    tree = build_tree(cache)
    base = copy.deepcopy(model)
    rt = LoRARuntime(model, cache, tree)
    rt.activate(merged_first)
    nodes = ["task", None, "user"]
    with torch.no_grad(), rt.batch(nodes):
        out = model(x)
    torch.testing.assert_close(out, per_row_reference(base, cache, tree, nodes),
                               atol=1e-5, rtol=1e-5)
    for m in model.modules():
        assert not m._forward_hooks  # hooks removed after the block


def test_batched_training_grads_only_reach_used_adapters(model, cache):
    used = make_adapter("used", 4, seed=7)
    unused = make_adapter("unused", 4, seed=8)
    for ad in (used, unused):
        for a, b in ad.weights.values():
            a.requires_grad_(True)
            b.requires_grad_(True)
    rt = LoRARuntime(model, cache)
    rt.register_trainable(used)
    rt.register_trainable(unused)
    with rt.batch(["used", None, "used"]):
        model(x).pow(2).mean().backward()
    assert all(a.grad is not None and b.grad is not None for a, b in used.weights.values())
    assert all(a.grad is None and b.grad is None for a, b in unused.weights.values())
    with pytest.raises(ValueError, match="trainable"):
        rt.activate("used")


def test_batch_dim_mismatch_is_reported(model, cache):
    cache.put(make_adapter("a", 4))
    rt = LoRARuntime(model, cache)
    with pytest.raises(RuntimeError, match="batch dim"):
        with rt.batch(["a", "a"]):
            model(x)


# ---------------------------------------------------------------- adapter I/O

def test_peft_roundtrip_and_compose(tmp_path, model):
    a = make_adapter("a", 8, seed=1)
    a.save_peft(tmp_path / "a")
    loaded = LoRAAdapter.load_peft(tmp_path / "a")
    assert loaded.rank == 8 and loaded.scaling == a.scaling
    for m in TARGETS:
        torch.testing.assert_close(loaded.delta(m), a.delta(m))

    b = make_adapter("b", 4, modules=TARGETS[:2], seed=2)
    c = compose([a, b])
    assert c.rank == 12 and c.scaling == 1.0
    for m in TARGETS:
        expected = a.delta(m) + (b.delta(m) if m in b.weights else 0)
        torch.testing.assert_close(c.delta(m), expected, atol=1e-6, rtol=1e-5)


def test_unknown_module_is_reported(model, cache):
    cache.put(make_adapter("bad", 4, modules=["blocks.9.q"]))
    rt = LoRARuntime(model, cache)
    with pytest.raises(KeyError, match="blocks.9.q"):
        rt.activate("bad")


def test_no_warning_noise_on_normal_tree():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tree = LoRATree()
        tree.add("a", rank=16)
        tree.add("b", parent="a", rank=8)


def test_quantized_target_is_rejected(model, cache):
    q = model.blocks[0].q
    q.weight = torch.nn.Parameter(torch.zeros(HIDDEN, HIDDEN, dtype=torch.int8), requires_grad=False)
    cache.put(make_adapter("a", 4))
    rt = LoRARuntime(model, cache)
    with pytest.raises(TypeError, match="quantized"):
        rt.activate("a")
    with pytest.raises(TypeError, match="quantized"):
        with rt.batch(["a", "a", "a"]):
            pass
