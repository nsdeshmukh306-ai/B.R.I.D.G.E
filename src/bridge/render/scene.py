"""Scene: the thread-safe set of primitives currently projected."""
from __future__ import annotations

import threading
from typing import Iterable, Optional

from bridge.render.primitives import Primitive


class Scene:
    def __init__(self) -> None:
        self._items: dict[str, Primitive] = {}
        self._lock = threading.RLock()
        self.version = 0

    def add(self, prim: Primitive) -> str:
        with self._lock:
            self._items[prim.id] = prim
            self.version += 1
        return prim.id

    def add_many(self, prims: Iterable[Primitive]) -> list[str]:
        return [self.add(p) for p in prims]

    def replace(self, prim: Primitive) -> None:
        self.add(prim)

    def remove(self, prim_id: str) -> None:
        with self._lock:
            self._items.pop(prim_id, None)
            self.version += 1

    def remove_group(self, group: str) -> int:
        with self._lock:
            ids = [k for k, v in self._items.items() if v.group == group]
            for k in ids:
                del self._items[k]
            self.version += 1
        return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.version += 1

    def get(self, prim_id: str) -> Optional[Primitive]:
        with self._lock:
            return self._items.get(prim_id)

    def items(self) -> list[Primitive]:
        with self._lock:
            return list(self._items.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
