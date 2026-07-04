"""Small runtime interfaces shared by simulator implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SimBody(ABC):
    """A runtime rigid body handle."""

    name: str
    index: int

    @abstractmethod
    def get_pose(self) -> Any:
        """Return the backend-native body pose."""
        raise NotImplementedError

    @abstractmethod
    def get_velocity(self) -> Any:
        """Return the backend-native body velocity."""
        raise NotImplementedError


@dataclass
class SimJoint(ABC):
    """A runtime joint handle."""

    name: str
    index: int

    @abstractmethod
    def get_position(self) -> Any:
        """Return the backend-native joint position value."""
        raise NotImplementedError

    @abstractmethod
    def get_velocity(self) -> Any:
        """Return the backend-native joint velocity value."""
        raise NotImplementedError


@dataclass
class SimShape(ABC):
    """A runtime collision or visual shape handle."""

    name: str
    index: int
    body_index: int | None = None


@dataclass
class SimEntity(ABC):
    """A loaded runtime asset instance."""

    name: str
    category: str
    kind: str
    source_path: Path
    body_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]
    shape_indices: tuple[int, ...]

    @property
    @abstractmethod
    def bodies(self) -> dict[str, SimBody]:
        raise NotImplementedError

    @property
    @abstractmethod
    def joints(self) -> dict[str, SimJoint]:
        raise NotImplementedError

    @property
    @abstractmethod
    def shapes(self) -> dict[str, SimShape]:
        raise NotImplementedError

    @property
    def root_body(self) -> SimBody | None:
        return next(iter(self.bodies.values()), None)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "source_path": str(self.source_path),
            "body_count": len(self.body_indices),
            "joint_count": len(self.joint_indices),
            "shape_count": len(self.shape_indices),
        }


class SimulatorBase(ABC):
    """Minimal simulator lifecycle and data interface."""

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def entities(self) -> tuple[SimEntity, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def objects(self) -> tuple[SimEntity, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def robots(self) -> tuple[SimEntity, ...]:
        raise NotImplementedError
