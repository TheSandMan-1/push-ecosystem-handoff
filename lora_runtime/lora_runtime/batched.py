"""Multi-LoRA batching: one base model, each batch row routed through its own adapters.

Rows are grouped by route (the segmented-gather idea behind Punica/S-LoRA):
for every target linear, each distinct route computes x_rows @ A^T @ B^T for
its rows only and scatters the result back. The base matmul still runs once
for the whole batch, so N tasks share one forward pass.

Routes are expressed relative to whatever is currently merged. If a shared
trunk (say the language + domain adapters) is merged into the weights, a row
routed to a leaf under that trunk only pays for the leaf, and a row that
should see the plain base model subtracts the trunk. That lets you keep the
common path merged at zero cost and batch the cheap leaves unmerged.

Works for training as well as inference: if the adapter tensors require grad,
gradients flow only into the adapters used by rows in the batch.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch
from torch import nn

from .adapter import LoRAAdapter
from .merge import target_weight

# One term = (A, B, coefficient * scaling) applied to a module for a route.
Term = tuple[torch.Tensor, torch.Tensor, float]


def relative_terms(route: list[LoRAAdapter], merged: list[LoRAAdapter]) -> list[tuple[LoRAAdapter, float]]:
    """Adapters to add (+1) or subtract (-1) so a row sees exactly ``route``."""
    route_names = {a.name for a in route}
    merged_names = {a.name for a in merged}
    return ([(a, 1.0) for a in route if a.name not in merged_names]
            + [(a, -1.0) for a in merged if a.name not in route_names])


class _RoutingHook:
    def __init__(self, module_name: str, groups: list[tuple[torch.Tensor, list[Term]]],
                 num_rows: int) -> None:
        self.module_name = module_name
        self.groups = groups
        self.num_rows = num_rows

    def __call__(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        x = args[0]
        if x.shape[0] != self.num_rows:
            raise RuntimeError(
                f"{self.module_name}: input batch dim is {x.shape[0]} but {self.num_rows} routes "
                "were given; multi-LoRA routing needs the batch dimension first"
            )
        delta = torch.zeros_like(output)
        for rows, terms in self.groups:
            xs = x.index_select(0, rows)
            acc = None
            for a, b, coef in terms:
                part = (xs @ a.to(xs.dtype).T) @ b.to(xs.dtype).T * coef
                acc = part if acc is None else acc + part
            delta.index_add_(0, rows, acc.to(delta.dtype))
        return output + delta


@contextmanager
def routed(model: nn.Module, routes: list[list[tuple[LoRAAdapter, float]]],
           row_routes: list[int], device: torch.device | str):
    """Install routing hooks for one forward pass.

    ``routes[i]`` is a list of (adapter, coefficient) terms; ``row_routes[j]``
    is the route index for batch row j.
    """
    modules = dict(model.named_modules())
    device = torch.device(device)
    rows_by_route: dict[int, list[int]] = defaultdict(list)
    for row, route_id in enumerate(row_routes):
        rows_by_route[route_id].append(row)
    row_index = {rid: torch.tensor(rows, device=device, dtype=torch.long)
                 for rid, rows in rows_by_route.items()}

    per_module: dict[str, list[tuple[torch.Tensor, list[Term]]]] = defaultdict(list)
    for rid, rows in row_index.items():
        by_module: dict[str, list[Term]] = defaultdict(list)
        for adapter, coef in routes[rid]:
            for name, (a, b) in adapter.weights.items():
                by_module[name].append((a, b, coef * adapter.scaling))
        for name, terms in by_module.items():
            per_module[name].append((rows, terms))

    handles = []
    try:
        for name, groups in per_module.items():
            if name not in modules:
                raise KeyError(f"adapter targets {name!r}, which is not a module of this model")
            target_weight(modules[name])  # validates the layer type
            handles.append(modules[name].register_forward_hook(
                _RoutingHook(name, groups, len(row_routes))))
        yield
    finally:
        for h in handles:
            h.remove()
