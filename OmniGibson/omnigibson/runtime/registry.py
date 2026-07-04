"""Runtime entity registry."""

from collections import defaultdict
from collections.abc import Iterable

from omnigibson.runtime.entity import SimEntity


class EntityRegistry:
    """Lookup table for loaded runtime entities."""

    def __init__(self):
        self._by_name = {}
        self._by_kind = defaultdict(list)
        self._by_category = defaultdict(list)
        self._by_source_path = {}

    def add(self, entity: SimEntity):
        if entity.name in self._by_name:
            raise ValueError(f"Entity with name {entity.name!r} already exists.")
        self._by_name[entity.name] = entity
        self._by_kind[entity.kind].append(entity)
        self._by_category[entity.category].append(entity)
        self._by_source_path[str(entity.source_path)] = entity

    def clear(self):
        self._by_name.clear()
        self._by_kind.clear()
        self._by_category.clear()
        self._by_source_path.clear()

    def get(self, name: str, default=None):
        return self._by_name.get(name, default)

    def by_kind(self, kind: str) -> tuple[SimEntity, ...]:
        return tuple(self._by_kind.get(kind, ()))

    def by_category(self, category: str) -> tuple[SimEntity, ...]:
        return tuple(self._by_category.get(category, ()))

    def by_source_path(self, source_path: str, default=None):
        return self._by_source_path.get(str(source_path), default)

    def values(self) -> tuple[SimEntity, ...]:
        return tuple(self._by_name.values())

    def extend(self, entities: Iterable[SimEntity]):
        for entity in entities:
            self.add(entity)

    def __len__(self):
        return len(self._by_name)

    def __iter__(self):
        return iter(self._by_name.values())
