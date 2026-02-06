from collections.abc import Iterable

import cv2
import numpy as np
from transforms3d.quaternions import quat2mat

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import TriggerGeomPrim
from omnigibson.sensors.sensor_base import BaseSensor
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative

m = create_module_macros(module_path=__file__)
m.DEBUG_VISUALIZATION = False


class VoxelSensor(BaseSensor):
    """
    General 3D voxelized occupancy grid sensor.

    Args:
        relative_prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Default is "all", which corresponds
            to all modalities being used. Otherwise, valid options should be part of cls.all_modalities.
            For this scan sensor, this includes any of:
                {voxel}
        enabled (bool): Whether this sensor should be enabled by default
        noise (None or BaseSensorNoise): If specified, sensor noise model to apply to this sensor.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this sensor's prim at runtime.
        dist (float): Maximum diameter to sense in meters
        cell_count (float): How many voxels to subdivide the range sensor into in each dimension
        ignore_object_names (None or list of str): If specified, a list of object names to ignore when sensing.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        modalities="all",
        enabled=True,
        noise=None,
        load_config=None,
        # Basic LIDAR kwargs
        dist=0.5,
        cell_count=5,
        ignore_object_names=None,
    ):
        # Create variables that will be filled in at runtime
        self._triggers = None

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["dist"] = dist
        load_config["cell_count"] = cell_count
        load_config["ignore_object_names"] = ignore_object_names
        assert cell_count % 2 == 1, "cell_count must be odd for centering purposes"

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            modalities=modalities,
            enabled=enabled,
            noise=noise,
            load_config=load_config,
        )

    def _load(self):
        # Create the prim if it doesn't exist
        prim = og.sim.stage.GetPrimAtPath(self.prim_path)
        if not prim:
            prim = og.sim.stage.DefinePrim(self.prim_path, "Xform")

        # Load the prims
        cell_count = self._load_config["cell_count"]
        dist = self._load_config["dist"]
        extent = dist / cell_count
        self._triggers = np.zeros((cell_count, cell_count, cell_count), dtype=object)
        trigger_idx = 0
        start = -(cell_count // 2)
        end = (cell_count // 2) + 1
        for x in range(start, end):
            for y in range(start, end):
                for z in range(start, end):
                    # Get the center of the voxel
                    center = np.array([x, y, z]) * extent
                    trigger_prim_path = f"{self.prim_path}/trigger_{trigger_idx}"
                    trigger_prim = lazy.pxr.UsdGeom.Cube.Define(og.sim.stage, trigger_prim_path)
                    trigger_prim.CreateSizeAttr().Set(extent)
                    relative_trigger_prim_path = absolute_prim_path_to_scene_relative(self.scene, trigger_prim_path)
                    trigger = TriggerGeomPrim(
                        relative_prim_path=relative_trigger_prim_path, name=f"{self.name}_trigger_{trigger_idx}"
                    )
                    trigger.load(self.scene)
                    trigger.color = [1.0, 0.0, 0.0]
                    if m.DEBUG_VISUALIZATION:
                        trigger.purpose = "default"
                    trigger.set_local_pose(center, np.array([0, 0, 0, 1]))

                    self._triggers[x, y, z] = trigger

                    trigger_idx += 1

        return prim

    @property
    def _obs_space_mapping(self):
        # Set the remaining modalities' values
        cell_count = self._load_config["cell_count"]
        obs_space_mapping = dict(
            voxel=((cell_count, cell_count, cell_count), 0.0, 1.0, np.bool),
        )

        return obs_space_mapping

    def _get_obs(self):
        # Run super first to grab any upstream obs
        obs, info = super()._get_obs()

        # Add scan info (normalized to [0.0, 1.0])
        if "voxel" in self._modalities:

            def _get_sensor_hits(trigger_geom):
                trigger_geom.visible = False

                trigger_colliders = trigger_geom.get_colliding_prim_paths()
                for collision in trigger_colliders:
                    # Strip the collisions part of the path if it exists
                    link_prim_path = str(collision).split("/collisions")[0]
                    obj_prim_path, _ = link_prim_path.rsplit("/", 1)
                    obj = trigger_geom.scene.object_registry("prim_path", obj_prim_path)
                    if obj is not None:
                        if (
                            not self._load_config["ignore_object_names"]
                            or obj.name not in self._load_config["ignore_object_names"]
                        ):
                            trigger_geom.visible = True
                            return True

                return False

            _get_sensor_hits_vec = np.vectorize(_get_sensor_hits)
            cell_count = self._load_config["cell_count"]
            obs["voxel"] = _get_sensor_hits_vec(self._triggers).astype(np.bool)
            assert obs["voxel"].shape == (
                cell_count,
                cell_count,
                cell_count,
            ), f"Invalid voxel shape {obs['voxel'].shape}!"

        return obs, info

    @classproperty
    def all_modalities(cls):
        return {"voxel"}
