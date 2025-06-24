"""Simple floor arenas."""

from typing import Tuple

import numpy as np
from dm_control import mjcf

from gello.dm_control_tasks.arenas import base

_GROUNDPLANE_QUAD_SIZE = 0.25


class FixedManipulationArena(base.Arena):
    @property
    def arm_attachment_site(self) -> np.ndarray:
        return self.attachment_site


class Floor(base.Arena):
    """An arena with a checkered pattern."""

    def _build(
        self,
        name: str = "floor",
        size: Tuple[float, float] = (8, 8),
        reflectance: float = 0.2,
        top_camera_y_padding_factor: float = 1.1,
        top_camera_distance: float = 5.0,
    ) -> None:
        super()._build(name=name)

        self._size = size
        self._top_camera_y_padding_factor = top_camera_y_padding_factor
        self._top_camera_distance = top_camera_distance

        assert self._mjcf_root.worldbody is not None

        z_offset = 0.00

        # Add arm attachement site
        self._mjcf_root.worldbody.add(
            "site",
            name="arm_attachment",
            pos=(0, 0, z_offset),
            size=(0.01, 0.01, 0.01),
            type="sphere",
            rgba=(0, 0, 0, 0),
        )

        # Add light.
        self._mjcf_root.worldbody.add(
            "light",
            pos=(0, 0, 1.5),
            dir=(0, 0, -1),
            directional=True,
        )

        self._ground_texture = self._mjcf_root.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=200,
            height=200,
            mark="edge",
            markrgb=[0.8, 0.8, 0.8],
        )

        self._ground_material = self._mjcf_root.asset.add(
            "material",
            name="groundplane",
            texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
            texuniform=True,
            reflectance=reflectance,
            texture=self._ground_texture,
        )

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="groundplane",
            material=self._ground_material,
            size=list(size) + [_GROUNDPLANE_QUAD_SIZE],
        )

        # Choose the FOV so that the floor always fits nicely within the frame
        # irrespective of actual floor size.
        fovy_radians = 2 * np.arctan2(
            top_camera_y_padding_factor * size[1], top_camera_distance
        )
        self._top_camera = self._mjcf_root.worldbody.add(
            "camera",
            name="top_camera",
            pos=[0, 0, top_camera_distance],
            quat=[1, 0, 0, 0],
            fovy=np.rad2deg(fovy_radians),
        )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def size(self):
        return self._size

    @property
    def arm_attachment_site(self) -> mjcf.Element:
        return self._mjcf_root.worldbody.find("site", "arm_attachment")
