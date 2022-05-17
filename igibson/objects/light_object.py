# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import Iterable, OrderedDict
from typing import Optional, Tuple
from pxr import Gf, Usd, UsdGeom, UsdShade, UsdPhysics, UsdLux
from omni.isaac.core.utils.types import XFormPrimState
from omni.isaac.core.materials import PreviewSurface, OmniGlass, OmniPBR, VisualMaterial
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    move_prim,
    query_parent_path,
    is_prim_path_valid,
    define_prim,
    get_prim_parent,
    get_prim_object_type,
)
import numpy as np
import carb
import logging
from omni.isaac.core.utils.stage import get_current_stage
from igibson.objects.stateful_object import StatefulObject
from igibson.prims.xform_prim import XFormPrim
from igibson.utils.python_utils import assert_valid_key


class LightObject(StatefulObject):
    """
    LightObjects are objects that generate light in the simulation
    """
    LIGHT_TYPES = {
        "Cylinder",
        "Disk",
        "Distant",
        "Dome",
        "Geometry",
        "Rect",
        "Sphere",
    }

    """
    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        light_type (str): Type of light to create. Valid options are LIGHT_TYPES
        name (str): Name for the object. Names need to be unique per scene.
        category (str): Category for the object. Defaults to "object".
        class_id (str): What class ID the object should be assigned in semantic segmentation rendering mode.
        scale (None or float or 3-array): If specified, sets the scale for this object.
            A single number corresponds to uniform scaling along the x,y,z axes, whereas a 3-array 
            specifies per-axis scaling.
        rendering_params (dict): Any relevant rendering settings for this object.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. For this xform prim, the below values can be specified
        abilities (dict): dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        radius (float): Radius for this light.
        intensity (float): Intensity for this light.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
    """

    def __init__(
        self,
        prim_path,
        light_type,
        name=None,
        category="light",
        class_id=None,
        scale=None,
        rendering_params=None,
        load_config=None,
        abilities=None,
        radius=1.0,
        intensity=50000.0,
        **kwargs,
    ):
        # Compose load config and add rgba values
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = scale
        load_config["intensity"] = intensity
        load_config["radius"] = radius

        # Make sure primitive type is valid
        assert_valid_key(key=light_type, valid_keys=self.LIGHT_TYPES, name="light_type")
        self.light_type = light_type

        # Other attributes to be filled in at runtime
        self._light_link = None

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=True,
            fixed_base=False,
            visual_only=True,
            self_collisions=False,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    def _load(self, simulator=None):
        logging.info(f"Loading the following light: {self.light_type}")

        # Define a light prim at the current stage, or the simulator's stage if specified
        stage = get_current_stage()

        # Define XForm and base link for this light
        prim = stage.DefinePrim(self._prim_path, "Xform")
        base_link = stage.DefinePrim(f"{self._prim_path}/base_link", "Xform")

        # Define the actual light link
        light_prim = UsdLux.__dict__[f"{self.light_type}Light"].Define(stage, f"{self._prim_path}/base_link/light").GetPrim()

        return prim

    def _post_load(self):
        # run super first
        super()._post_load()

        # Grab reference to light link
        self._light_link = XFormPrim(prim_path=f"{self._prim_path}/base_link/light", name=f"{self.name}:light_link")

        # Optionally set the intensity
        if self._load_config.get("intensity", None) is not None:
            self.intensity = self._load_config["intensity"]

        # Optionally set the radius
        if self._load_config.get("radius", None) is not None:
            self.radius = self._load_config["radius"]

    def _initialize(self):
        # Run super
        super()._initialize()

        # Initialize light link
        self._light_link.initialize()

    @property
    def light_link(self):
        """
        Returns:
            XFormPrim: Link corresponding to the light prim itself
        """
        return self._light_link

    @property
    def radius(self):
        """
        Gets this joint's radius

        Returns:
            float: radius for this light
        """
        return self._light_link.get_attribute("radius")

    @radius.setter
    def radius(self, radius):
        """
        Sets this joint's radius

        Args:
            radius (float): radius to set
        """
        self._light_link.set_attribute("radius", radius)

    @property
    def intensity(self):
        """
        Gets this joint's intensity

        Returns:
            float: intensity for this light
        """
        return self._light_link.get_attribute("intensity")

    @intensity.setter
    def intensity(self, intensity):
        """
        Sets this joint's intensity

        Args:
            intensity (float): intensity to set
        """
        self._light_link.set_attribute("intensity", intensity)

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        # Add additional kwargs (fit_avg_dim_volume and bounding_box are already captured in load_config)
        return self.__class__(
            prim_path=prim_path,
            light_type=self.light_type,
            name=name,
            intensity=self.intensity,
            load_config=load_config,
        )
