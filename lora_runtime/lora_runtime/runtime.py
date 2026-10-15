"""LoRARuntime: one base model, many adapters, two serving modes.

* ``activate(node)``: merge the tree path for ``node`` into the weights. The
  forward pass is then exactly the base model's (no hooks, no extra matmuls).
  Switching nodes applies only the adapters that differ between the paths.
* ``batch(nodes)``: route each row of the next forward pass through its own
  node's path, relative to whatever is merged. Use it for mixed-task batches.

Adapters come from an ``AdapterCache`` (RAM-resident, never reads disk on
these paths) or from ``register_trainable`` for adapters being trained.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .adapter import LoRAAdapter
from .batched import relative_terms, routed
from .cache import AdapterCache
from .merge import WeightMerger
from .tree import LoRATree


class LoRARuntime:
    def __init__(self, model: nn.Module, cache: AdapterCache, tree: LoRATree | None = None,
                 restore: str = "delta") -> None:
        self.model = model
        self.cache = cache
        self.tree = tree or LoRATree()
        self.merger = WeightMerger(model, restore=restore)
        self.active: str | None = None
        self._trainable: dict[str, LoRAAdapter] = {}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def register_trainable(self, adapter: LoRAAdapter) -> None:
        """Use this adapter object as-is (e.g. tensors with requires_grad) in ``batch``."""
        self._trainable[adapter.name] = adapter

    def resolve(self, node: str | None) -> list[LoRAAdapter]:
        """Device-resident adapters for the tree path to ``node`` (root first)."""
        if node is not None and node not in self.tree:
            # Allow a bare adapter name without building a tree.
            names = [node]
        else:
            names = self.tree.path(node)
        return [self._trainable.get(n) or self.cache.on_device(n, self.device) for n in names]

    # ------------------------------------------------------------------ merged mode

    def activate(self, node: str | None) -> dict[str, int]:
        """Merge the path to ``node`` (``None`` = plain base model)."""
        target = self.resolve(node)
        if any(a.name in self._trainable for a in target):
            raise ValueError("trainable adapters are routed with batch(), not merged")
        stats = self.merger.transition(target)
        self.active = node
        return stats

    def deactivate(self) -> dict[str, int]:
        return self.activate(None)

    # ------------------------------------------------------------------ batched mode

    @contextmanager
    def batch(self, nodes: list[str | None]):
        """Route row i of the next forward pass(es) through ``nodes[i]``'s path."""
        merged = self.merger.merged
        route_ids: dict[str | None, int] = {}
        routes = []
        row_routes = []
        for node in nodes:
            if node not in route_ids:
                route_ids[node] = len(routes)
                routes.append(relative_terms(self.resolve(node), merged))
            row_routes.append(route_ids[node])
        with routed(self.model, routes, row_routes, self.device):
            yield
