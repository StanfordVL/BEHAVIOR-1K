# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple
from pxr import Gf, Usd, UsdGeom, UsdShade
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
from omni.isaac.core.utils.stage import get_current_stage


class BasePrim(metaclass=ABCMeta):
    """
    Provides high level functions to deal with a basic prim and its attributes/ properties.
        If there is an Xform prim present at the path, it will use it. Otherwise, a new XForm prim at
        the specified prim path will be created.

        Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
                unless it is a non-root articulation link.

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. Note that this is only needed if the prim does not already exist at
                @prim_path -- it will be ignored if it already exists. Subclasses should define the exact keys expected
                for their class.
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        self._prim_path = prim_path
        self._name = name
        self._load_config = load_config

        # Other values that will be filled in at runtime
        self._applied_visual_material = None
        self._binding_api = None
        self._loaded = False
        self._prim = None

        # Run some post-loading steps if this prim has already been loaded
        if is_prim_path_valid(prim_path=self._prim_path):
            self._prim = get_prim_at_path(prim_path=self._prim_path)
            self._setup_references()
            self._loaded = True

    def _setup_references(self):
        """
        Sets up any references necessary post-loading
        """
        pass

    def load(self, simulator=None):
        """
        Load this prim into omniverse, optionally integrating this prim with simulator context @simulator, and return
        loaded prim reference.

        Args:
            simulator (None or SimulationContext): If specified, should be simulator into which this prim will be
                loaded. Otherwise, it will be loaded into the default stage

        :return Usd.Prim: Prim object loaded into the simulator
        """
        if self._loaded:
            raise ValueError("Cannot load a single prim multiple times.")

        # For generalizability, make sure load_config is a dict even if not specified
        self._load_config = {} if self._load_config is None else self._load_config

        # Load prim
        self._prim = self._load(simulator=simulator)

        # Clear load config (should never be used again, since all these values can be polled in real-time directly
        # from the stage)
        self._load_config = None

        # Setup any references
        self._setup_references()

        self._loaded = True

        return self._prim

    @abstractmethod
    def _load(self, simulator=None):
        pass

    @property
    def loaded(self):
        return self._loaded

    @property
    def prim_path(self):
        """
        Returns:
            str: prim path in the stage.
        """
        return self._prim_path

    @property
    def name(self):
        """
        Returns:
            str: name assigned to this prim
        """
        return self._name

    @property
    def prim(self) -> Usd.Prim:
        """
        Returns:
            Usd.Prim: USD Prim object that this object holds.
        """
        return self._prim

    @property
    def visible(self):
        """
        Returns:
            bool: true if the prim is visible in stage. false otherwise.
        """
        return UsdGeom.Imageable(self.prim).ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible

    @visible.setter
    def visible(self, visible):
        """
        Sets the visibility of the prim in stage.

        Args:
            visible (bool): flag to set the visibility of the usd prim in stage.
        """
        imageable = UsdGeom.Imageable(self.prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        return

    def is_valid(self) -> bool:
        """
        Returns:
            bool: True is the current prim path corresponds to a valid prim in stage. False otherwise.
        """
        return is_prim_path_valid(self.prim_path)

    def change_prim_path(self, new_prim_path: str) -> None:
        """Moves prim from the old path to a new one.

        Args:
            new_prim_path (str): new path of the prim to be moved to.
        """
        move_prim(path_from=self.prim_path, path_to=new_prim_path)
        self._prim_path = new_prim_path
        self._prim = get_prim_at_path(self._prim_path)
        return

    def get_attribute(self, attr):
        """
        Get this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Returns:
            any: value of the requested @attribute
        """
        return self._prim.GetAttribute(attr).Get()

    def set_attribute(self, attr, val):
        """
        Set this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Args:
            attr (str): Attribute to set
            val (any): Value to set for the attribute. This should be the valid type for that attribute.
        """
        self._prim.GetAttribute(attr).Set(val)

    def get_property(self, prop):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to get. See Raw USD Properties in the GUI for examples of property names

        Returns:
            any: Property value
        """
        self._prim.GetProperty(prop).Get()

    def set_property(self, prop, val):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to set. See Raw USD Properties in the GUI for examples of property names
            val (any): Value to set for the property. Should be valid for that property
        """
        self._prim.GetProperty(prop).Set(val)

    def get_custom_data(self):
        """
        Get custom data associated with this prim
        :return dict: Dictionary of any custom information
        """
        return self._prim.GetCustomData()
