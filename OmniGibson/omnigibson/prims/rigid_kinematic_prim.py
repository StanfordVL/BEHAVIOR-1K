from typing import Literal

import torch as th

from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import PoseAPI

from .rigid_prim import RigidPrim


class RigidKinematicPrim(RigidPrim):
    """
    Provides high level functions to deal with a kinematic-only rigid prim and its attributes/properties.
    A kinematic-only object is not subject to simulator dynamics, and remains fixed unless the user
    explicitly sets the body's pose / velocities.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # Make sure it's set to be kinematic
        if not self.is_attribute_valid("physics:kinematicEnabled"):
            self.create_attribute("physics:kinematicEnabled", True)
        if not self.is_attribute_valid("physics:rigidBodyEnabled"):
            self.create_attribute("physics:rigidBodyEnabled", False)
        self.set_attribute("physics:kinematicEnabled", True)
        self.set_attribute("physics:rigidBodyEnabled", False)

        # Run super method to handle common functionality
        super()._post_load()

    def set_position_orientation(self, position=None, orientation=None, frame: Literal["world", "scene"] = "world"):
        """
        Set the position and orientation of the kinematic rigid body.

        Args:
            position (None or 3-array): The position to set the object to. If None, the position is not changed.
            orientation (None or 4-array): The orientation to set the object to. If None, the orientation is not changed.
            frame (Literal): The frame in which to set the position and orientation. Defaults to world.
                Scene frame sets position relative to the scene.
        """
        # Use the XFormPrim implementation directly
        XFormPrim.set_position_orientation(self, position=position, orientation=orientation, frame=frame)

        # Invalidate pose API
        PoseAPI.invalidate()

    # The following methods implement the same interface as RigidDynamicPrim, but as no-op
    # versions for kinematic-only prims. This allows code to call these methods on any RigidPrim
    # without type checking, while maintaining proper physics behavior based on the actual
    # runtime type (dynamic vs. kinematic).

    @property
    def center_of_mass(self):
        """
        Returns:
            th.Tensor: (x,y,z) position of link CoM in the link frame
        """
        return th.zeros(3)

    @center_of_mass.setter
    def center_of_mass(self, com):
        """
        Args:
            com (th.Tensor): (x,y,z) position of link CoM in the link frame
        """
        pass

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        return 0.0

    @mass.setter
    def mass(self, _):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        pass

    @property
    def density(self):
        """
        Returns:
            float: density of the rigid body in kg / m^3.
        """
        return 0.0

    @density.setter
    def density(self, _):
        """
        Args:
            density (float): density of the rigid body in kg / m^3.
        """
        pass

    def enable_gravity(self):
        """
        Enables gravity for this rigid body
        """
        pass

    def disable_gravity(self):
        """
        Disables gravity for this rigid body
        """
        pass
