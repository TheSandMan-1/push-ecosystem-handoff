import pytest
import torch
from torch import nn

from lora_runtime import AdapterCache, LoRAAdapter

HIDDEN = 32
TARGETS = ["blocks.0.q", "blocks.0.v", "blocks.1.q", "blocks.1.v"]


class Block(nn.Module):
    def __init__(self, h: int) -> None:
        super().__init__()
        self.q = nn.Linear(h, h)
        self.v = nn.Linear(h, h)
        self.out = nn.Linear(h, h)

    def forward(self, x):
        return x + self.out(torch.tanh(self.q(x)) * self.v(x))


class ToyModel(nn.Module):
    def __init__(self, h: int = HIDDEN, layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(h) for _ in range(layers))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def make_adapter(name: str, rank: int, modules=TARGETS, seed: int = 0, alpha: float | None = None,
                 h: int = HIDDEN) -> LoRAAdapter:
    g = torch.Generator().manual_seed(seed)
    weights = {m: (torch.randn(rank, h, generator=g) * 0.1, torch.randn(h, rank, generator=g) * 0.1)
               for m in modules}
    return LoRAAdapter(name=name, rank=rank, alpha=alpha or 2.0 * rank, weights=weights)


def reference_merge(model: ToyModel, adapters: list[LoRAAdapter]) -> ToyModel:
    """Independent fp64 reference: fresh copy of the base with deltas added."""
    import copy

    ref = copy.deepcopy(model).double()
    mods = dict(ref.named_modules())
    with torch.no_grad():
        for ad in adapters:
            for m, (a, b) in ad.weights.items():
                mods[m].weight += (b.double() @ a.double()) * ad.scaling
    return ref


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ToyModel().eval()


@pytest.fixture
def cache():
    return AdapterCache(ram_budget_bytes=64 << 20, pin_memory=False)
