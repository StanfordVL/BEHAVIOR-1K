"""Arena composer class."""

from pathlib import Path

from dm_control import composer, mjcf

_ARENA_XML = Path(__file__).resolve().parent / "arena.xml"


class Arena(composer.Entity):
    """Base arena class."""

    def _build(self, name: str = "arena") -> None:
        self._mjcf_root = mjcf.from_path(str(_ARENA_XML))
        if name is not None:
            self._mjcf_root.model = name

    def add_free_entity(self, entity) -> mjcf.Element:
        """Includes an entity as a free moving body, i.e., with a freejoint."""
        frame = self.attach(entity)
        frame.add("freejoint")
        return frame

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root
