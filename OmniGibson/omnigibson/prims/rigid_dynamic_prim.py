import math
from typing import Literal

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.usd_utils import PoseAPI

from .rigid_prim import RigidPrim


class RigidDynamicPrim(RigidPrim):
    """
    Provides high level functions to deal with a dynamic rigid body prim and its attributes/properties.
    This class is used for rigid bodies that are not kinematic-only, meaning they are subject to physics simulation
    dynamics like gravity, forces, and collisions.

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
        # Set up rigid body view which will be filled in later
        self._rigid_prim_view = None

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # Make sure it's set to be a rigid body (not kinematic)
        if not self.is_attribute_valid("physics:kinematicEnabled"):
            self.create_attribute("physics:kinematicEnabled", False)
        if not self.is_attribute_valid("physics:rigidBodyEnabled"):
            self.create_attribute("physics:rigidBodyEnabled", True)
        self.set_attribute("physics:kinematicEnabled", False)
        self.set_attribute("physics:rigidBodyEnabled", True)

        # Create the rigid prim view
        # Import now to avoid too-eager load of Omni classes due to inheritance
        from omnigibson.utils.deprecated_utils import RigidPrimView

        # set reset_xform_properties to False for load time
        self._rigid_prim_view = RigidPrimView(self.prim_path, reset_xform_properties=False)

        # Run super method to handle common functionality
        super()._post_load()

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        # Validate that the view is valid if physics is running
        if og.sim.is_playing() and self.initialized:
            assert (
                self._rigid_prim_view.is_physics_handle_valid() and self._rigid_prim_view._physics_view.check()
            ), "Rigid prim view must be valid if physics is running!"

        assert not (
            og.sim.is_playing() and not self._rigid_prim_view.is_valid
        ), "Rigid prim view must be valid if physics is running!"

        self._rigid_prim_view.initialize(og.sim.physics_sim_view)

    def set_linear_velocity(self, velocity):
        """
        Sets the linear velocity of the prim in stage.

        Args:
            velocity (th.tensor): linear velocity to set the rigid prim to. Shape (3,).
        """
        self._rigid_prim_view.set_linear_velocities(velocity[None, :])

    def get_linear_velocity(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            th.tensor: current linear velocity of the the rigid prim. Shape (3,).
        """
        return self._rigid_prim_view.get_linear_velocities(clone=clone)[0]

    def set_angular_velocity(self, velocity):
        """
        Sets the angular velocity of the prim in stage.

        Args:
            velocity (th.tensor): angular velocity to set the rigid prim to. Shape (3,).
        """
        self._rigid_prim_view.set_angular_velocities(velocity[None, :])

    def get_angular_velocity(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            th.tensor: current angular velocity of the the rigid prim. Shape (3,).
        """
        return self._rigid_prim_view.get_angular_velocities(clone=clone)[0]

    def set_position_orientation(self, position=None, orientation=None, frame: Literal["world", "scene"] = "world"):
        """
        Set the position and orientation of the dynamic rigid body.

        Args:
            position (None or 3-array): The position to set the object to. If None, the position is not changed.
            orientation (None or 4-array): The orientation to set the object to. If None, the orientation is not changed.
            frame (Literal): The frame in which to set the position and orientation. Defaults to world.
                Scene frame sets position relative to the scene.
        """
        assert frame in ["world", "scene"], f"Invalid frame '{frame}'. Must be 'world' or 'scene'."

        # If no position or no orientation are given, get the current position and orientation of the object
        if position is None or orientation is None:
            current_position, current_orientation = self.get_position_orientation(frame=frame)
        position = current_position if position is None else position
        orientation = current_orientation if orientation is None else orientation

        # Convert to th.Tensor if necessary
        position = th.as_tensor(position, dtype=th.float32)
        orientation = th.as_tensor(orientation, dtype=th.float32)

        # Assert validity of the orientation
        assert math.isclose(
            th.norm(orientation).item(), 1, abs_tol=1e-3
        ), f"{self.prim_path} desired orientation {orientation} is not a unit quaternion."

        # Convert to from scene-relative to world if necessary
        if frame == "scene":
            assert self.scene is not None, "cannot set position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_scene_relative_pose_to_world(position, orientation)

        self._rigid_prim_view.set_world_poses(positions=position[None, :], orientations=orientation[None, [3, 0, 1, 2]])
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
        assert frame in ["world", "scene"], f"Invalid frame '{frame}'. Must be 'world', or 'scene'."

        # Get the pose from the rigid prim view and convert to our format
        positions, orientations = self._rigid_prim_view.get_world_poses(clone=clone)
        position = positions[0]
        orientation = orientations[0][[1, 2, 3, 0]]  # Convert from (w,x,y,z) to (x,y,z,w)

        # Assert that the orientation is a unit quaternion
        assert math.isclose(
            th.norm(orientation).item(), 1, abs_tol=1e-3
        ), f"{self.prim_path} orientation {orientation} is not a unit quaternion."

        # If requested, compute the scene-local transform
        if frame == "scene":
            assert self.scene is not None, "Cannot get position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_world_pose_to_scene_relative(position, orientation)

        return position, orientation

    @property
    def center_of_mass(self):
        """
        Returns:
            th.Tensor: (x,y,z) position of link CoM in the link frame
        """
        positions, orientations = self._rigid_prim_view.get_coms(clone=True)
        position = positions[0][0]
        return position

    @center_of_mass.setter
    def center_of_mass(self, com):
        """
        Args:
            com (th.Tensor): (x,y,z) position of link CoM in the link frame
        """
        self._rigid_prim_view.set_coms(positions=com.reshape(1, 1, 3))

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        mass = self._rigid_prim_view.get_masses()[0]

        # Fallback to analytical computation of volume * density
        if mass == 0:
            return self.volume * self.density

        return mass

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._rigid_prim_view.set_masses(th.tensor([mass]))

    @property
    def density(self):
        """
        Returns:
            float: density of the rigid body in kg / m^3.
        """
        mass = self._rigid_prim_view.get_masses()[0]
        # We first check if the mass is specified, since mass overrides density. If so, density = mass / volume.
        # Otherwise, we try to directly grab the raw usd density value, and if that value does not exist,
        # we return 1000 since that is the canonical density assigned by omniverse
        if mass != 0.0:
            density = mass / self.volume
        else:
            density = self._rigid_prim_view.get_densities()[0]
            if density == 0.0:
                density = 1000.0

        return density

    @density.setter
    def density(self, density):
        """
        Args:
            density (float): density of the rigid body in kg / m^3.
        """
        self._rigid_prim_view.set_densities(th.tensor([density]))

    @property
    def is_asleep(self):
        """
        Returns:
            bool: whether this rigid prim is asleep or not
        """
        return og.sim.psi.is_sleeping(og.sim.stage_id, lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path))

    def enable_gravity(self):
        """
        Enables gravity for this rigid body
        """
        self._rigid_prim_view.enable_gravities()

    def disable_gravity(self):
        """
        Disables gravity for this rigid body
        """
        self._rigid_prim_view.disable_gravities()

    def wake(self):
        """
        Enable physics for this rigid body
        """
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path)
        og.sim.psi.wake_up(og.sim.stage_id, prim_id)

    def sleep(self):
        """
        Disable physics for this rigid body
        """
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path)
        og.sim.psi.put_to_sleep(og.sim.stage_id, prim_id)

    @property
    def stabilization_threshold(self):
        """
        Returns:
            float: threshold for stabilizing this rigid body
        """
        return self.get_attribute("physxRigidBody:stabilizationThreshold")

    @stabilization_threshold.setter
    def stabilization_threshold(self, threshold):
        """
        Sets threshold for stabilizing this rigid body

        Args:
            threshold (float): stabilizing threshold
        """
        self.set_attribute("physxRigidBody:stabilizationThreshold", threshold)

    @property
    def sleep_threshold(self):
        """
        Returns:
            float: threshold for sleeping this rigid body
        """
        return self.get_attribute("physxRigidBody:sleepThreshold")

    @sleep_threshold.setter
    def sleep_threshold(self, threshold):
        """
        Sets threshold for sleeping this rigid body

        Args:
            threshold (float): Sleeping threshold
        """
        self.set_attribute("physxRigidBody:sleepThreshold", threshold)

    @property
    def solver_position_iteration_count(self):
        """
        Returns:
            int: How many position iterations to take per physics step by the physx solver
        """
        return self.get_attribute("physxRigidBody:solverPositionIterationCount")

    @solver_position_iteration_count.setter
    def solver_position_iteration_count(self, count):
        """
        Sets how many position iterations to take per physics step by the physx solver

        Args:
            count (int): How many position iterations to take per physics step by the physx solver
        """
        self.set_attribute("physxRigidBody:solverPositionIterationCount", count)

    @property
    def solver_velocity_iteration_count(self):
        """
        Returns:
            int: How many velocity iterations to take per physics step by the physx solver
        """
        return self.get_attribute("physxRigidBody:solverVelocityIterationCount")

    @solver_velocity_iteration_count.setter
    def solver_velocity_iteration_count(self, count):
        """
        Sets how many velocity iterations to take per physics step by the physx solver

        Args:
            count (int): How many velocity iterations to take per physics step by the physx solver
        """
        self.set_attribute("physxRigidBody:solverVelocityIterationCount", count)

    def _dump_state(self):
        # Grab pose from super class
        state = super()._dump_state()

        state["lin_vel"] = self.get_linear_velocity(clone=False)
        state["ang_vel"] = self.get_angular_velocity(clone=False)

        return state

    def _load_state(self, state):
        # If we are part of an articulation, there's nothing to do, the entityprim will take care
        # of setting everything for us.
        if self._belongs_to_articulation:
            return

        # Call super first
        super()._load_state(state=state)

        # Set velocities
        self.set_linear_velocity(
            state["lin_vel"] if isinstance(state["lin_vel"], th.Tensor) else th.tensor(state["lin_vel"])
        )
        self.set_angular_velocity(
            state["ang_vel"] if isinstance(state["ang_vel"], th.Tensor) else th.tensor(state["ang_vel"])
        )

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        return th.cat(
            [
                state_flat,
                state["lin_vel"],
                state["ang_vel"],
            ]
        )

    def deserialize(self, state):
        # Call supermethod first
        state_dic, idx = super().deserialize(state=state)
        # We deserialize deterministically by knowing the order of values -- lin_vel, ang_vel
        state_dic["lin_vel"] = state[idx : idx + 3]
        state_dic["ang_vel"] = state[idx + 3 : idx + 6]

        return state_dic, idx + 6
