"""Hierarchical LoRA tree: base model at the root, adapters stacked along paths.

    base (root, loaded once)
    └── lang-en            (rank 64)   language
        └── medtech        (rank 32)   domain
            ├── dfmea      (rank 8)    task
            └── user-ali   (rank 4)    user

The effective model at a node is base + sum of deltas on the path to it.
Switching between two nodes only touches the adapters that differ between the
two paths; shared ancestors stay merged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class TreeNode:
    name: str
    adapter: str
    parent: str | None
    rank: int | None = None


class LoRATree:
    def __init__(self) -> None:
        self._nodes: dict[str, TreeNode] = {}

    def add(self, name: str, adapter: str | None = None, parent: str | None = None,
            rank: int | None = None) -> TreeNode:
        """Add a node. ``parent=None`` hangs it directly off the base model."""
        if name in self._nodes:
            raise ValueError(f"node {name!r} already exists")
        if parent is not None and parent not in self._nodes:
            raise KeyError(f"parent {parent!r} does not exist")
        if parent is not None and rank is not None:
            parent_rank = self._nodes[parent].rank
            if parent_rank is not None and rank > parent_rank:
                warnings.warn(
                    f"node {name!r} has rank {rank} > parent {parent!r} rank {parent_rank}; "
                    "ranks usually shrink toward the leaves",
                    stacklevel=2,
                )
        node = TreeNode(name=name, adapter=adapter or name, parent=parent, rank=rank)
        self._nodes[name] = node
        return node

    def path(self, name: str | None) -> list[str]:
        """Adapter names from the top of the tree down to ``name`` (inclusive)."""
        if name is None:
            return []
        if name not in self._nodes:
            raise KeyError(f"unknown tree node {name!r}")
        chain = []
        node: TreeNode | None = self._nodes[name]
        while node is not None:
            chain.append(node.adapter)
            node = self._nodes[node.parent] if node.parent else None
        return chain[::-1]

    def children(self, name: str | None) -> list[str]:
        return [n.name for n in self._nodes.values() if n.parent == name]

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)
