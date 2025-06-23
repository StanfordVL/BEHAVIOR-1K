import math
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
        # Caches for kinematic-only objects
        # This exists because RigidPrimView uses USD pose read, which is very slow
        # For scene-relative poses, we also manually compute pose transforms, which can be slow if repeatedly queried
        self._kinematic_pose_cache = dict()  # "scene", "world" keys

        # Run super init
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

        # Invalidate kinematic-only object pose cache when new pose is set
        self.clear_kinematic_only_cache()

        # Invalidate pose API
        PoseAPI.invalidate()

    def get_position_orientation(self, frame: Literal["world", "scene"] = "world", clone=True):
        """
        Gets prim's pose with respect to the specified frame.

        Args:
            frame (Literal): frame to get the pose with respect to. Default to world.
                scene frame gets position relative to the scene.
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            2-tuple:
                - th.Tensor: (x,y,z) position in the specified frame
                - th.Tensor: (x,y,z,w) quaternion orientation in the specified frame
        """
        # If we don't have the raw (world-frame) pose, query it now
        if "world" not in self._kinematic_pose_cache:
            # If this is the first time we're getting the pose, use XFormPrim implementation
            position, orientation = XFormPrim.get_position_orientation(self, clone=clone)

            # Assert that the orientation is a unit quaternion
            assert math.isclose(
                th.norm(orientation).item(), 1, abs_tol=1e-3
            ), f"{self.prim_path} orientation {orientation} is not a unit quaternion."

            # Cache world pose
            self._kinematic_pose_cache["world"] = (position, orientation)

        # Grab the pose in the desired frame
        if frame == "scene":
            assert self.scene is not None, "Cannot get position and orientation relative to scene without a scene"
            if "scene" not in self._kinematic_pose_cache:
                # Transform the pose into the scene-relative frame
                self._kinematic_pose_cache["scene"] = self.scene.convert_world_pose_to_scene_relative(
                    *self._kinematic_pose_cache["world"]
                )
            pos, ori = self._kinematic_pose_cache["scene"]
        else:
            pos, ori = self._kinematic_pose_cache["world"]

        return pos, ori

    def clear_kinematic_only_cache(self):
        """
        Clears the internal kinematic only cached pose. Useful if the parent prim's pose
        changes without explicitly calling this prim's pose setter
        """
        self._kinematic_pose_cache = dict()

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
