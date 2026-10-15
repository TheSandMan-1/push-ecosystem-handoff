"""Two-tier adapter cache: host RAM (large) -> accelerator memory (hot set).

Disk is touched only by ``preload``/``load``. On the serving path, ``get`` and
``on_device`` never read disk: a miss raises ``CacheMiss`` so a cold adapter
shows up as an error instead of a silent multi-millisecond stall.

On CUDA, host copies are pinned so host->device transfers run asynchronously.
On Apple Silicon (MPS) memory is unified: there is no separate VRAM, pinning
does not apply, and the "transfer" is a memcpy inside the same pool. The cache
still matters there because it removes the disk read.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapter import LoRAAdapter


class CacheMiss(KeyError):
    """Adapter is not resident in RAM and disk reads are disabled on this path."""


@dataclass
class CacheStats:
    disk_reads: int = 0
    ram_hits: int = 0
    device_hits: int = 0
    device_transfers: int = 0
    bytes_transferred: int = 0
    ram_evictions: int = 0
    device_evictions: int = 0


class AdapterCache:
    def __init__(
        self,
        ram_budget_bytes: int = 8 << 30,
        device_slots: int = 32,
        pin_memory: bool | None = None,
    ) -> None:
        self.ram_budget_bytes = ram_budget_bytes
        self.device_slots = device_slots
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self._ram: OrderedDict[str, LoRAAdapter] = OrderedDict()
        self._device: OrderedDict[tuple[str, str], LoRAAdapter] = OrderedDict()
        self.stats = CacheStats()

    # ---------------------------------------------------------------- loading

    def load(self, path: str | Path, name: str | None = None) -> LoRAAdapter:
        """Read one PEFT adapter from disk into RAM. Not for the serving path."""
        adapter = LoRAAdapter.load_peft(path, name=name)
        self.stats.disk_reads += 1
        return self.put(adapter)

    def preload(self, directory: str | Path) -> list[str]:
        """Load every adapter subdirectory under ``directory`` into RAM."""
        names = []
        for sub in sorted(Path(directory).iterdir()):
            if (sub / "adapter_config.json").exists():
                names.append(self.load(sub).name)
        return names

    def put(self, adapter: LoRAAdapter) -> LoRAAdapter:
        """Register an in-memory adapter (e.g. fresh from training)."""
        if adapter.device.type != "cpu":
            adapter = adapter.to("cpu")
        if self.pin_memory:
            adapter = adapter.pin_memory()
        if adapter.nbytes > self.ram_budget_bytes:
            raise ValueError(
                f"adapter {adapter.name} ({adapter.nbytes} B) exceeds the RAM budget"
            )
        self._ram[adapter.name] = adapter
        self._ram.move_to_end(adapter.name)
        self._drop_device_copies(adapter.name)
        while self.ram_bytes > self.ram_budget_bytes:
            evicted, _ = self._ram.popitem(last=False)
            self._drop_device_copies(evicted)
            self.stats.ram_evictions += 1
        return adapter

    # ---------------------------------------------------------------- serving

    def get(self, name: str) -> LoRAAdapter:
        """RAM copy of an adapter. Never reads disk."""
        try:
            adapter = self._ram[name]
        except KeyError:
            raise CacheMiss(
                f"adapter {name!r} is not resident in RAM; preload it before serving"
            ) from None
        self._ram.move_to_end(name)
        self.stats.ram_hits += 1
        return adapter

    def on_device(self, name: str, device: torch.device | str) -> LoRAAdapter:
        """Device copy of an adapter, transferring from RAM if needed. Never reads disk."""
        device = torch.device(device)
        if device.type == "cpu":
            return self.get(name)
        key = (name, str(device))
        if key in self._device:
            self._device.move_to_end(key)
            self.stats.device_hits += 1
            return self._device[key]
        host = self.get(name)
        moved = host.to(device, non_blocking=self.pin_memory)
        self.stats.device_transfers += 1
        self.stats.bytes_transferred += host.nbytes
        self._device[key] = moved
        while len(self._device) > self.device_slots:
            self._device.popitem(last=False)
            self.stats.device_evictions += 1
        return moved

    # ---------------------------------------------------------------- introspection

    @property
    def ram_bytes(self) -> int:
        return sum(a.nbytes for a in self._ram.values())

    def __contains__(self, name: str) -> bool:
        return name in self._ram

    def __len__(self) -> int:
        return len(self._ram)

    def names(self) -> list[str]:
        return list(self._ram)

    def _drop_device_copies(self, name: str) -> None:
        for key in [k for k in self._device if k[0] == name]:
            del self._device[key]
