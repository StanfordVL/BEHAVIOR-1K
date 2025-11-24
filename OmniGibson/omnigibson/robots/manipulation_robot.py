import math
import os
from abc import abstractmethod
from collections import namedtuple
from functools import cached_property
from typing import Literal

import networkx as nx
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import (
    ControlType,
    GripperController,
    InverseKinematicsController,
    IsGraspingState,
    ManipulationController,
    MultiFingerGripperController,
    OperationalSpaceController,
)
from omnigibson.macros import create_module_macros, gm
from omnigibson.object_states import ContactBodies
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.constants import JointType, PrimType
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sampling_utils import raytest_batch
from omnigibson.utils.usd_utils import (
    ControllableObjectViewAPI,
    GripperRigidContactAPI,
    create_joint,
    create_primitive_mesh,
    absolute_prim_path_to_scene_relative,
    delete_or_deactivate_prim,
)
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Assisted grasping parameters
m.ASSIST_FRACTION = 1.0
m.ASSIST_GRASP_MASS_THRESHOLD = 10.0
m.ARTICULATED_ASSIST_FRACTION = 0.7
m.MIN_ASSIST_FORCE = 0
m.MAX_ASSIST_FORCE = 100
m.MIN_AG_DEFAULT_GRASP_POINT_PROP = 0.2
m.MAX_AG_DEFAULT_GRASP_POINT_PROP = 0.95
m.AG_DEFAULT_GRASP_POINT_Z_PROP = 0.4

m.CONSTRAINT_VIOLATION_THRESHOLD = 0.1
m.GRASP_WINDOW = 1 / 30.0  # grasp window in seconds
m.RELEASE_WINDOW = 1 / 30.0  # release window in seconds

AG_MODES = {
    "physical",
    "assisted",
    "sticky",
}
GraspingPoint = namedtuple("GraspingPoint", ["link_name", "position"])  # link_name (str), position (x,y,z tuple)


class ManipulationRobot(BaseRobot):
    """
    Robot that is is equipped with grasping (manipulation) capabilities.
    Provides common interface for a wide variety of robots.

    NOTE: controller_config should, at the minimum, contain:
        arm: controller specifications for the controller to control this robot's arm (manipulation).
            Should include:

            - name: Controller to create
            - <other kwargs> relevant to the controller being created. Note that all values will have default
                values specified, but setting these individual kwargs will override them
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        grasping_direction="lower",
        disable_grasp_handling=False,
        finger_static_friction=None,
        finger_dynamic_friction=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers. In this mode,
                    at least two "fingers" need to touch the object.
                If "sticky", will magnetize any object touching the gripper's fingers. In this mode, only one finger
                    needs to touch the object.
            grasping_direction (str): One of {"lower", "upper"}. If "lower", lower limit represents a closed grasp,
                otherwise upper limit represents a closed grasp.
            disable_grasp_handling (bool): If True, the robot will not automatically handle assisted or sticky grasps.
                Instead, you will need to call the grasp handling methods yourself.
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store relevant internal vars
        assert_valid_key(key=grasping_mode, valid_keys=AG_MODES, name="grasping_mode")
        assert_valid_key(key=grasping_direction, valid_keys=["lower", "upper"], name="grasping direction")
        self._grasping_mode = grasping_mode
        self._grasping_direction = grasping_direction
        self._disable_grasp_handling = disable_grasp_handling

        # Other variables filled in at runtime
        self._eef_to_fingertip_lengths = None  # dict mapping arm name to finger name to offset

        # Initialize other variables used for assistive grasping
        self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
        self._ag_obj_constraints = {arm: None for arm in self.arm_names}
        self._ag_obj_constraint_params = {arm: {} for arm in self.arm_names}
        self._ag_freeze_gripper = {arm: None for arm in self.arm_names}
        self._ag_release_counter = {arm: None for arm in self.arm_names}
        self._ag_grasp_counter = {arm: None for arm in self.arm_names}

        # Call super() method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            **kwargs,
        )

        # Update finger link material dictionary based on desired values
        if finger_static_friction is not None or finger_dynamic_friction is not None:
            for arm, finger_link_names in self.finger_link_names.items():
                for finger_link_name in finger_link_names:
                    if finger_link_name not in self._link_physics_materials:
                        self._link_physics_materials[finger_link_name] = dict()
                    if finger_static_friction is not None:
                        self._link_physics_materials[finger_link_name]["static_friction"] = finger_static_friction
                    if finger_dynamic_friction is not None:
                        self._link_physics_materials[finger_link_name]["dynamic_friction"] = finger_dynamic_friction

    def _validate_configuration(self):
        # Iterate over all arms
        for arm in self.arm_names:
            # If we have an arm controller, make sure it is a manipulation controller
            if f"arm_{arm}" in self._controllers:
                assert isinstance(
                    self._controllers["arm_{}".format(arm)], ManipulationController
                ), "Arm {} controller must be a ManipulationController!".format(arm)

            # If we have a gripper controller, make sure it is a manipulation controller
            if f"gripper_{arm}" in self._controllers:
                assert isinstance(
                    self._controllers["gripper_{}".format(arm)], GripperController
                ), "Gripper {} controller must be a GripperController!".format(arm)

        # run super
        super()._validate_configuration()

    def _initialize(self):
        super()._initialize()

        # make eef link not visible
        for arm in self.arm_names:
            self._links[self.eef_link_names[arm]].visible = False

        # Infer relevant link properties, e.g.: fingertip location, AG grasping points
        # We use a try / except to maintain backwards-compatibility with robots that do not follow our
        # OG-specified convention
        try:
            self._infer_finger_properties()
        except AssertionError as e:
            log.warning(f"Could not infer relevant finger link properties because:\n\n{e}")

    def _infer_finger_properties(self):
        """
        Infers relevant finger properties based on the given finger meshes of the robot

        NOTE: This assumes the given EEF convention for parallel jaw grippers -- i.e.:
        z points in the direction of the fingers, y points in the direction of grasp articulation, and x
        is then inferred automatically
        """
        # Calculate and cache fingertip to eef frame offsets, as well as AG grasping points
        self._eef_to_fingertip_lengths = dict()
        self._default_ag_start_points = dict()
        self._default_ag_end_points = dict()
        for arm, finger_links in self.finger_links.items():
            self._eef_to_fingertip_lengths[arm] = dict()
            eef_link = self.eef_links[arm]
            world_to_eef_tf = T.pose2mat(eef_link.get_position_orientation())
            eef_to_world_tf = T.pose_inv(world_to_eef_tf)

            # Infer parent link for this finger
            finger_parent_link, finger_parent_max_z = None, None
            is_parallel_jaw = len(finger_links) == 2
            assert (
                is_parallel_jaw
            ), "Inferring finger link information can only be done for parallel jaw gripper robots!"
            finger_pts_in_eef_frame = []
            for i, finger_link in enumerate(finger_links):
                # Find parent, and make sure one exists
                parent_prim_path, parent_link = None, None
                for joint in self.joints.values():
                    if finger_link.prim_path == joint.body1:
                        parent_prim_path = joint.body0
                        break
                assert (
                    parent_prim_path is not None
                ), f"Expected articulated parent joint for finger link {finger_link.name} but found none!"
                for link in self.links.values():
                    if parent_prim_path == link.prim_path:
                        parent_link = link
                        break
                assert parent_link is not None, f"Expected parent link located at {parent_prim_path} but found none!"
                # Make sure all fingers share the same parent
                if finger_parent_link is None:
                    finger_parent_link = parent_link
                    finger_parent_pts = finger_parent_link.collision_boundary_points_world
                    assert (
                        finger_parent_pts is not None
                    ), f"Expected finger parent points to be defined for parent link {finger_parent_link.name}, but got None!"
                    # Convert from world frame -> eef frame
                    finger_parent_pts = th.concatenate([finger_parent_pts, th.ones(len(finger_parent_pts), 1)], dim=-1)
                    finger_parent_pts = (finger_parent_pts @ eef_to_world_tf.T)[:, :3]
                    finger_parent_max_z = finger_parent_pts[:, 2].max().item()
                else:
                    assert (
                        finger_parent_link == parent_link
                    ), f"Expected all fingers to have same parent link, but found multiple parents at {finger_parent_link.prim_path} and {parent_link.prim_path}"

                # Calculate this finger's collision boundary points in the world frame
                finger_pts = finger_link.collision_boundary_points_world
                assert (
                    finger_pts is not None
                ), f"Expected finger points to be defined for link {finger_link.name}, but got None!"
                # Convert from world frame -> eef frame
                finger_pts = th.concatenate([finger_pts, th.ones(len(finger_pts), 1)], dim=-1)
                finger_pts = (finger_pts @ eef_to_world_tf.T)[:, :3]
                finger_pts_in_eef_frame.append(finger_pts)

            # Determine how each finger is located relative to the other in the EEF frame along the y-axis
            # This is used to infer which side of each finger's set of points correspond to the "inner" surface
            finger_pts_mean = [finger_pts[:, 1].mean().item() for finger_pts in finger_pts_in_eef_frame]
            first_finger_is_lower_y_finger = finger_pts_mean[0] < finger_pts_mean[1]
            is_lower_y_fingers = [first_finger_is_lower_y_finger, not first_finger_is_lower_y_finger]

            for i, (finger_link, finger_pts, is_lower_y_finger) in enumerate(
                zip(finger_links, finger_pts_in_eef_frame, is_lower_y_fingers)
            ):
                # Since we know the EEF frame always points with z outwards towards the fingers, the outer-most point /
                # fingertip is the maximum z value
                finger_max_z = finger_pts[:, 2].max().item()
                assert (
                    finger_max_z > 0
                ), f"Expected positive fingertip to eef frame offset for link {finger_link.name}, but got: {finger_max_z}. Does the EEF frame z-axis point in the direction of the fingers?"
                self._eef_to_fingertip_lengths[arm][finger_link.name] = finger_max_z

                # Now, only keep points that are above the parent max z by 20% for inferring y values
                finger_range = finger_max_z - finger_parent_max_z
                valid_idxs = th.where(
                    finger_pts[:, 2] > (finger_parent_max_z + finger_range * m.MIN_AG_DEFAULT_GRASP_POINT_PROP)
                )[0]
                finger_pts = finger_pts[valid_idxs]
                # Infer which side of the gripper corresponds to the inner side (i.e.: the side that touches between the
                # two fingers
                # We use the heuristic that given a set of points defining a gripper finger, we assume that it's one
                # of (y_min, y_max) over all points, with the selection being chosen by inferring which of the limits
                # corresponds to the inner side of the finger.
                # This is the upper side of the y values if this finger is the lower finger, else the lower side
                # of the y values
                y_min, y_max = finger_pts[:, 1].min(), finger_pts[:, 1].max()
                y_offset = y_max if is_lower_y_finger else y_min
                y_sign = 1.0 if is_lower_y_finger else -1.0

                # Compute the default grasping points for this finger
                # For now, we only have a strong heuristic defined for parallel jaw grippers, which assumes that
                # there are exactly 2 fingers
                # In this case, this is defined as the x2 (x,y,z) tuples where:
                # z - the +/-40% from the EEF frame, bounded by the 20% and 100% length between the range from
                #       [finger_parent_max_z, finger_max_z]
                #       This is synonymous to inferring the length of the finger (lower bounded by the gripper base,
                #       assumed to be the parent link), and then taking the values +/-%, bounded by the MIN% and MAX%
                #       along its length
                # y - the value closest to the edge of the finger surface in the direction of +/- EEF y-axis.
                #       This assumes that each individual finger lies completely on one side of the EEF y-axis
                # x - 0. This assumes that the EEF axis evenly splits each finger symmetrically on each side
                # (x,y,z,1) -- homogenous form for efficient transforming into finger link frame
                z_lower = max(
                    finger_parent_max_z + finger_range * m.MIN_AG_DEFAULT_GRASP_POINT_PROP,
                    -finger_range * m.AG_DEFAULT_GRASP_POINT_Z_PROP,
                )
                z_upper = min(
                    finger_parent_max_z + finger_range * m.MAX_AG_DEFAULT_GRASP_POINT_PROP,
                    finger_range * m.AG_DEFAULT_GRASP_POINT_Z_PROP,
                )
                # We want to ensure the z value is symmetric about the EEF z frame, so make sure z_lower is negative
                # and z_upper is positive, and use +/- the absolute minimum value between the two
                assert (
                    z_lower < 0 and z_upper > 0
                ), f"Expected computed z_lower / z_upper bounds for finger grasping points to be negative / positive, but instead got: {z_lower}, {z_upper}"
                z_offset = min(abs(z_lower), abs(z_upper))

                grasp_pts = th.tensor(
                    [
                        [
                            0,
                            y_offset + 0.002 * y_sign,
                            -z_offset,
                            1,
                        ],
                        [
                            0,
                            y_offset + 0.002 * y_sign,
                            z_offset,
                            1,
                        ],
                    ]
                )
                # Convert the grasping points from the EEF frame -> finger frame
                finger_to_world_tf = T.pose_inv(T.pose2mat(finger_link.get_position_orientation()))
                finger_to_eef_tf = finger_to_world_tf @ world_to_eef_tf
                grasp_pts = (grasp_pts @ finger_to_eef_tf.T)[:, :3]
                grasping_points = [
                    GraspingPoint(link_name=finger_link.body_name, position=grasp_pt) for grasp_pt in grasp_pts
                ]
                if i == 0:
                    # Start point
                    self._default_ag_start_points[arm] = grasping_points
                else:
                    # End point
                    self._default_ag_end_points[arm] = grasping_points

        # For each grasping point, if we're in DEBUG mode, visualize with spheres
        if gm.DEBUG:
            for ag_points in (self.assisted_grasp_start_points, self.assisted_grasp_end_points):
                for arm_ag_points in ag_points.values():
                    # Skip if None exist
                    if arm_ag_points is None:
                        continue
                    # For each ag point, generate a small sphere at that point
                    for i, arm_ag_point in enumerate(arm_ag_points):
                        link = self.links[arm_ag_point.link_name]
                        local_pos = arm_ag_point.position
                        vis_mesh_prim_path = f"{link.prim_path}/ag_point_{i}"
                        create_primitive_mesh(
                            prim_path=vis_mesh_prim_path,
                            extents=0.005,
                            primitive_type="Sphere",
                        )
                        vis_geom = VisualGeomPrim(
                            relative_prim_path=absolute_prim_path_to_scene_relative(
                                scene=self.scene,
                                absolute_prim_path=vis_mesh_prim_path,
                            ),
                            name=f"ag_point_{i}",
                        )
                        vis_geom.load(self.scene)
                        vis_geom.set_position_orientation(
                            position=local_pos,
                            frame="parent",
                        )

    def is_grasping(self, arm="default", candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        Args:
            arm (str): specific arm to check for grasping. Default is "default" which corresponds to the first entry
                in self.arm_names
            candidate_obj (StatefulObject or None): object to check if this robot is currently grasping. If None, then
                will be a general (object-agnostic) check for grasping.
                Note: if self.grasping_mode is "physical", then @candidate_obj will be ignored completely

        Returns:
            IsGraspingState: For the specific manipulator appendage, returns IsGraspingState.TRUE if it is grasping
                (potentially @candidate_obj if specified), IsGraspingState.FALSE if it is not grasping,
                and IsGraspingState.UNKNOWN if unknown.
        """
        arm = self.default_arm if arm == "default" else arm
        if self.grasping_mode != "physical":
            is_grasping_obj = (
                self._ag_obj_in_hand[arm] is not None
                if candidate_obj is None
                else self._ag_obj_in_hand[arm] == candidate_obj
            )
            is_grasping = (
                IsGraspingState.TRUE
                if is_grasping_obj and self._ag_release_counter[arm] is None
                else IsGraspingState.FALSE
            )
        else:
            # Infer from the gripper controller the state
            is_grasping = self._controllers["gripper_{}".format(arm)].is_grasping()
            # If candidate obj is not None, we also check to see if our fingers are in contact with the object
            if is_grasping == IsGraspingState.TRUE and candidate_obj is not None:
                finger_links = {link for link in self.finger_links[arm]}
                if len(candidate_obj.states[ContactBodies].get_value().intersection(finger_links)) == 0:
                    is_grasping = IsGraspingState.FALSE
        return is_grasping

    def _find_gripper_contacts(self, arm="default", return_contact_positions=False):
        """
        For arm @arm, calculate any body IDs and corresponding link IDs that are not part of the robot
        itself that are in contact with any of this arm's gripper's fingers
        Args:
            arm (str): specific arm whose gripper will be checked for contact. Default is "default" which
                corresponds to the first entry in self.arm_names
            return_contact_positions (bool): if True, will additionally return the contact (x,y,z) position
        Returns:
            2-tuple:
                - set: set of unique contact prim_paths that are not the robot self-collisions.
                    If @return_contact_positions is True, then returns (prim_path, pos), where pos is the contact
                    (x,y,z) position
                    Note: if no objects that are not the robot itself are intersecting, the set will be empty.
                - dict: dictionary mapping unique contact objects defined by the contact prim_path to
                    set of unique robot link prim_paths that it is in contact with
        """
        arm = self.default_arm if arm == "default" else arm

        # Get robot finger links
        finger_paths = set([link.prim_path for link in self.finger_links[arm]])

        # Get robot links
        link_paths = set(self.link_prim_paths)

        if not return_contact_positions:
            raw_contact_data = {
                (row, col)
                for row, col in GripperRigidContactAPI.get_contact_pairs(self.scene.idx, column_prim_paths=finger_paths)
                if row not in link_paths
            }
        else:
            raw_contact_data = {
                (row, col, point)
                for row, col, force, point, normal, sep in GripperRigidContactAPI.get_contact_data(
                    self.scene.idx, column_prim_paths=finger_paths
                )
                if row not in link_paths
            }

        # Translate to robot contact data
        robot_contact_links = dict()
        contact_data = set()
        for con_data in raw_contact_data:
            if not return_contact_positions:
                other_contact, link_contact = con_data
                contact_data.add(other_contact)
            else:
                other_contact, link_contact, point = con_data
                contact_data.add((other_contact, point))
            if other_contact not in robot_contact_links:
                robot_contact_links[other_contact] = set()
            robot_contact_links[other_contact].add(link_contact)

        return contact_data, robot_contact_links

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Sets manipulation robot's pose with respect to the specified frame

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
            frame (Literal): frame to set the pose with respect to, defaults to "world".
                parent frame: set position relative to the object parent.
                scene frame: set position relative to the scene.
        """

        # Store the original EEF poses.
        original_poses = {}
        for arm in self.arm_names:
            original_poses[arm] = (self.get_eef_position(arm), self.get_eef_orientation(arm))

        # Run the super method
        super().set_position_orientation(position=position, orientation=orientation, frame=frame)

        # Now for each hand, if it was holding an AG object, teleport it.
        for arm in self.arm_names:
            if self._ag_obj_in_hand[arm] is not None:
                original_eef_pose = T.pose2mat(original_poses[arm])
                inv_original_eef_pose = T.pose_inv(pose_mat=original_eef_pose)
                original_obj_pose = T.pose2mat(self._ag_obj_in_hand[arm].get_position_orientation())
                new_eef_pose = T.pose2mat((self.get_eef_position(arm), self.get_eef_orientation(arm)))
                # New object pose is transform:
                # original --> "De"transform the original EEF pose --> "Re"transform the new EEF pose
                new_obj_pose = new_eef_pose @ inv_original_eef_pose @ original_obj_pose
                self._ag_obj_in_hand[arm].set_position_orientation(*T.mat2pose(hmat=new_obj_pose))

    def deploy_control(self, control, control_type):
        # We intercept the gripper control and replace it with the current joint position if we're freezing our gripper
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                control[self.gripper_control_idx[arm]] = (
                    self._ag_obj_constraint_params[arm]["gripper_pos"]
                    if self.controllers[f"gripper_{arm}"].control_type == ControlType.POSITION
                    else 0.0
                )

        super().deploy_control(control=control, control_type=control_type)

        # Then run assisted grasping
        if self.grasping_mode != "physical" and not self._disable_grasp_handling:
            self._handle_assisted_grasping()

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        Args:
            arm (str): specific arm whose grasp will be released.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm

        # Remove joint and filtered collision restraints
        delete_or_deactivate_prim(self._ag_obj_constraint_params[arm]["ag_joint_prim_path"])
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = {}
        self._ag_freeze_gripper[arm] = False
        self._ag_release_counter[arm] = 0

    def release_grasp_immediately(self, arm="default"):
        """
        Magic action to release this robot's grasp for one arm.
        As opposed to @_release_grasp, this method would bypass the release window mechanism and immediately release.
        """
        if self._ag_obj_constraints[arm] is not None:
            self._release_grasp(arm=arm)
            self._ag_release_counter[arm] = int(math.ceil(m.RELEASE_WINDOW / og.sim.get_physics_dt()))
            self._handle_release_window(arm=arm)
            assert not self._ag_obj_in_hand[arm], "Object still in ag list after release!"
            # TODO: Verify not needed!
            # for finger_link in self.finger_links[arm]:
            #     finger_link.remove_filtered_collision_pair(prim=self._ag_obj_in_hand[arm])

    def get_control_dict(self):
        # In addition to super method, add in EEF states
        fcns = super().get_control_dict()

        for arm in self.arm_names:
            self._add_task_frame_control_dict(fcns=fcns, task_name=f"eef_{arm}", link_name=self.eef_link_names[arm])

        return fcns

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Loop over all arms to grab proprio info
        joint_positions = dic["joint_qpos"]
        joint_velocities = dic["joint_qvel"]
        for arm in self.arm_names:
            # Add arm info
            dic["arm_{}_qpos".format(arm)] = joint_positions[self.arm_control_idx[arm]]
            dic["arm_{}_qpos_sin".format(arm)] = th.sin(joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qpos_cos".format(arm)] = th.cos(joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qvel".format(arm)] = joint_velocities[self.arm_control_idx[arm]]

            # Add eef and grasping info
            eef_pos, eef_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(
                self.articulation_root_path, self.eef_link_names[arm]
            )
            dic["eef_{}_pos".format(arm)], dic["eef_{}_quat".format(arm)] = cb.to_torch(eef_pos), cb.to_torch(eef_quat)
            dic["grasp_{}".format(arm)] = th.tensor([self.is_grasping(arm)])
            dic["gripper_{}_qpos".format(arm)] = joint_positions[self.gripper_control_idx[arm]]
            dic["gripper_{}_qvel".format(arm)] = joint_velocities[self.gripper_control_idx[arm]]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        for arm in self.arm_names:
            obs_keys += [
                "arm_{}_qpos_sin".format(arm),
                "arm_{}_qpos_cos".format(arm),
                "eef_{}_pos".format(arm),
                "eef_{}_quat".format(arm),
                "gripper_{}_qpos".format(arm),
                "grasp_{}".format(arm),
            ]
        return obs_keys

    @property
    def grasping_mode(self):
        """
        Grasping mode of this robot. Is one of AG_MODES

        Returns:
            str: Grasping mode for this robot
        """
        return self._grasping_mode

    @property
    def _raw_controller_order(self):
        # Assumes we have arm(s) and corresponding gripper(s)
        controllers = []
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # For best generalizability use, joint controller as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "JointController"
            controllers["gripper_{}".format(arm)] = "JointController"

        return controllers

    @classproperty
    def n_arms(cls):
        """
        Returns:
            int: Number of arms this robot has. Returns 1 by default
        """
        return 1

    @classproperty
    def arm_names(cls):
        """
        Returns:
            list of str: List of arm names for this robot. Should correspond to the keys used to index into
                arm- and gripper-related dictionaries, e.g.: eef_link_names, finger_link_names, etc.
                Default is string enumeration based on @self.n_arms.
        """
        return [str(i) for i in range(cls.n_arms)]

    @property
    def default_arm(self):
        """
        Returns:
            str: Default arm name for this robot, corresponds to the first entry in @arm_names by default
        """
        return self.arm_names[0]

    @property
    def arm_action_idx(self):
        arm_action_idx = {}
        for arm_name in self.arm_names:
            controller_idx = self.controller_order.index(f"arm_{arm_name}")
            action_start_idx = sum(
                [self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)]
            )
            arm_action_idx[arm_name] = th.arange(
                action_start_idx, action_start_idx + self.controllers[f"arm_{arm_name}"].command_dim
            )
        return arm_action_idx

    @property
    def gripper_action_idx(self):
        gripper_action_idx = {}
        for arm_name in self.arm_names:
            controller_idx = self.controller_order.index(f"gripper_{arm_name}")
            action_start_idx = sum(
                [self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)]
            )
            gripper_action_idx[arm_name] = th.arange(
                action_start_idx, action_start_idx + self.controllers[f"gripper_{arm_name}"].command_dim
            )
        return gripper_action_idx

    @cached_property
    @abstractmethod
    def arm_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm link names,
                should correspond to specific link names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def arm_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm joint names,
                should correspond to specific joint names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def eef_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding name of the EEF link,
                should correspond to specific link name in this robot's underlying model file
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def gripper_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of link names corresponding to
                this robot's gripper. Should be mutual exclusive from self.arm_link_names and self.finger_link_names!

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def finger_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of link names corresponding to
                this robot's fingers

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def finger_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of joint names corresponding to
                this robot's fingers.

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        raise NotImplementedError

    @cached_property
    def arm_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to arm joints.
        """
        return {
            arm: th.tensor([list(self.joints.keys()).index(name) for name in self.arm_joint_names[arm]])
            for arm in self.arm_names
        }

    @cached_property
    def gripper_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to gripper joints.
        """
        return {
            arm: th.tensor([list(self.joints.keys()).index(name) for name in self.finger_joint_names[arm]])
            for arm in self.arm_names
        }

    @cached_property
    def arm_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's links
        """
        return {arm: [self._links[link] for link in self.arm_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def eef_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot link corresponding to that arm's
                eef link. NOTE: These links always have a canonical local orientation frame -- assuming a parallel jaw
                eef morphology, it is assumed that the eef z-axis points out from the tips of the fingers, the y-axis
                points from the left finger to the right finger, and the x-axis inferred programmatically
        """
        return {arm: self._links[self.eef_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def gripper_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's gripper links
        """
        return {arm: [self._links[link] for link in self.gripper_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def finger_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's finger links
        """
        return {arm: [self._links[link] for link in self.finger_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def finger_joints(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot joints corresponding to
                that arm's finger joints
        """
        return {arm: [self._joints[joint] for joint in self.finger_joint_names[arm]] for arm in self.arm_names}

    @property
    def _assisted_grasp_start_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping start points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_end_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        # Should be optionally implemented by subclass
        return None

    @property
    def assisted_grasp_start_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping start points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_end_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        return (
            self._assisted_grasp_start_points
            if self._assisted_grasp_start_points is not None
            else self._default_ag_start_points
        )

    @property
    def _assisted_grasp_end_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping end points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_start_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        # Should be optionally implemented by subclass
        return None

    @property
    def assisted_grasp_end_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping end points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_start_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        return (
            self._assisted_grasp_end_points
            if self._assisted_grasp_end_points is not None
            else self._default_ag_end_points
        )

    @property
    def eef_to_fingertip_lengths(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to per-finger corresponding z-distance between EEF and each
                respective fingertip
        """
        return self._eef_to_fingertip_lengths

    @property
    def arm_workspace_range(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to a tuple indicating the start and end of the
                angular range of the arm workspace around the Z axis of the robot, where 0 is facing
                forward.
        """
        raise NotImplementedError

    def get_eef_pose(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef pose. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            2-tuple: End-effector pose, in (pos, quat) format, corresponding to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        return self._links[self.eef_link_names[arm]].get_position_orientation()

    def get_eef_position(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef position. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            3-array: (x,y,z) global end-effector Cartesian position for this robot's end-effector corresponding
                to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_eef_pose(arm=arm)[0]

    def get_eef_orientation(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef orientation. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            3-array: (x,y,z,w) global quaternion orientation for this robot's end-effector corresponding
                to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_eef_pose(arm=arm)[1]

    def get_relative_eef_pose(self, arm="default", mat=False):
        """
        Args:
            arm (str): specific arm to grab eef pose. Default is "default" which corresponds to the first entry
                in self.arm_names
            mat (bool): whether to return pose in matrix form (mat=True) or (pos, quat) tuple (mat=False)

        Returns:
            2-tuple or (4, 4)-array: End-effector pose, either in 4x4 homogeneous
                matrix form (if @mat=True) or (pos, quat) tuple (if @mat=False), corresponding to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        eef_link_pose = self.eef_links[arm].get_position_orientation()
        base_link_pose = self.get_position_orientation()
        pose = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
        return T.pose2mat(pose) if mat else pose

    def get_relative_eef_position(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef pos.
                Default is "default" which corresponds to the first entry in self.arm_names


        Returns:
            3-array: (x,y,z) Cartesian position of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[0]

    def get_relative_eef_orientation(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef orientation.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            4-array: (x,y,z,w) quaternion orientation of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[1]

    def get_relative_eef_lin_vel(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef linear velocity.
                Default is "default" which corresponds to the first entry in self.arm_names


        Returns:
            3-array: (x,y,z) Linear velocity of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        base_link_quat = self.get_position_orientation()[1]
        return T.quat2mat(base_link_quat).T @ self.eef_links[arm].get_linear_velocity()

    def get_relative_eef_ang_vel(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef angular velocity.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            3-array: (ax,ay,az) angular velocity of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        base_link_quat = self.get_position_orientation()[1]
        return T.quat2mat(base_link_quat).T @ self.eef_links[arm].get_angular_velocity()

    def _calculate_in_hand_object_rigid(self, arm="default"):
        """
        Calculates which object to assisted-grasp for arm @arm. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.

        Args:
            arm (str): specific arm to calculate in-hand object for.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            None or 2-tuple: If a valid assisted-grasp object is found, returns the corresponding
                (object, object_link) (i.e.: (BaseObject, RigidDynamicPrim)) pair to the contacted in-hand object.
                Otherwise, returns None
        """
        arm = self.default_arm if arm == "default" else arm

        # If we're not using physical grasping, we check for gripper contact
        if self.grasping_mode != "physical":
            candidates_set, robot_contact_links = self._find_gripper_contacts(arm=arm)
            # If we're using assisted grasping, we further filter candidates via ray-casting
            if self.grasping_mode == "assisted":
                candidates_set_raycast = self._find_gripper_raycast_collisions(arm=arm)
                candidates_set = candidates_set.intersection(candidates_set_raycast)
        else:
            raise ValueError("Invalid grasping mode for calculating in hand object: {}".format(self.grasping_mode))

        # Immediately return if there are no valid candidates
        if len(candidates_set) == 0:
            return None

        # Find the closest object to the gripper center
        gripper_center_pos = self.eef_links[arm].get_position_orientation()[0]

        candidate_data = []
        for prim_path in candidates_set:
            # Calculate position of the object link. Only allow this for objects currently.
            obj_prim_path, link_name = prim_path.rsplit("/", 1)
            candidate_obj = self.scene.object_registry("prim_path", obj_prim_path, None)
            if candidate_obj is None or link_name not in candidate_obj.links:
                continue
            candidate_link = candidate_obj.links[link_name]
            dist = th.norm(candidate_link.get_position_orientation()[0] - gripper_center_pos)
            candidate_data.append((prim_path, dist))

        if not candidate_data:
            return None

        candidate_data = sorted(candidate_data, key=lambda x: x[-1])
        ag_prim_path, _ = candidate_data[0]

        # Make sure the ag_prim_path is not a self collision
        assert ag_prim_path not in self.link_prim_paths, "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[ag_prim_path]
        touching_at_least_two_fingers = (
            True
            if self.grasping_mode == "sticky"
            else len({link.prim_path for link in self.finger_links[arm]}.intersection(robot_contacts)) >= 2
        )

        # TODO: Better heuristic, hacky, we assume the parent object prim path is the prim_path minus the last "/" item
        ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
        ag_obj_link_name = ag_prim_path.split("/")[-1]
        ag_obj = self.scene.object_registry("prim_path", ag_obj_prim_path)

        # Return None if object cannot be assisted grasped or not touching at least two fingers
        if ag_obj is None or not touching_at_least_two_fingers:
            return None

        # Get object and its contacted link
        return ag_obj, ag_obj.links[ag_obj_link_name]

    def _find_gripper_raycast_collisions(self, arm="default"):
        """
        For arm @arm, calculate any prims that are not part of the robot
        itself that intersect with rays cast between any of the gripper's start and end points

        Args:
            arm (str): specific arm whose gripper will be checked for raycast collisions. Default is "default"
            which corresponds to the first entry in self.arm_names

        Returns:
            set[str]: set of prim path of detected raycast intersections that
            are not the robot itself. Note: if no objects that are not the robot itself are intersecting,
            the set will be empty.
        """
        arm = self.default_arm if arm == "default" else arm
        # First, make sure start and end grasp points exist (i.e.: aren't None)
        assert (
            self.assisted_grasp_start_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_start_points must not be None!"
        assert (
            self.assisted_grasp_end_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_end_points must not be None!"

        # Iterate over all start and end grasp points and calculate their x,y,z positions in the world frame
        # (per arm appendage)
        # Since we'll be calculating the cartesian cross product between start and end points, we stack the start points
        # by the number of end points and repeat the individual elements of the end points by the number of start points
        n_start_points = len(self.assisted_grasp_start_points[arm])
        n_end_points = len(self.assisted_grasp_end_points[arm])
        start_and_end_points = th.zeros(n_start_points + n_end_points, 3)
        link_positions = th.zeros(n_start_points + n_end_points, 3)
        link_quats = th.zeros(n_start_points + n_end_points, 4)
        idx = 0
        for grasp_start_point in self.assisted_grasp_start_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_start_point.link_name].get_position_orientation()
            link_positions[idx] = link_pos
            link_quats[idx] = link_orn
            start_and_end_points[idx] = grasp_start_point.position
            idx += 1

        for grasp_end_point in self.assisted_grasp_end_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_end_point.link_name].get_position_orientation()
            link_positions[idx] = link_pos
            link_quats[idx] = link_orn
            start_and_end_points[idx] = grasp_end_point.position
            idx += 1

        # Transform start / end points into world frame (batched call for efficiency sake)
        start_and_end_points = link_positions + (T.quat2mat(link_quats) @ start_and_end_points.unsqueeze(-1)).squeeze(
            -1
        )
        # Stack the start points and repeat the end points, and add these values to the raycast dicts
        raycast_startpoints = th.tile(start_and_end_points[:n_start_points], (n_end_points, 1))
        raycast_endpoints = th.repeat_interleave(start_and_end_points[n_start_points:], n_start_points, dim=0) + 1e-8
        ray_data = set()
        # Calculate raycasts from each start point to end point -- this is n_startpoints * n_endpoints total rays
        for result in raytest_batch(raycast_startpoints, raycast_endpoints, only_closest=True):
            if result["hit"]:
                # filter out self body parts (we currently assume that the robot cannot grasp itself)
                if self.prim_path not in result["rigidBody"]:
                    ray_data.add(result["rigidBody"])
        return ray_data

    def _handle_release_window(self, arm="default"):
        """
        Handles releasing an object from arm @arm

        Args:
            arm (str): specific arm to handle release window.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        self._ag_release_counter[arm] += 1
        time_since_release = self._ag_release_counter[arm] * og.sim.get_physics_dt()
        if time_since_release >= m.RELEASE_WINDOW:
            self._ag_obj_in_hand[arm] = None
            self._ag_release_counter[arm] = None

    @property
    def curobo_path(self):
        """
        Returns:
            str or Dict[CuRoboEmbodimentSelection, str]: file path to the robot curobo file or a mapping from
                CuRoboEmbodimentSelection to the file path
        """
        # Import here to avoid circular imports
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        # By default, sets the standardized path
        model = self.model_name.lower()
        return {
            emb_sel: os.path.join(
                get_dataset_path("omnigibson-robot-assets"),
                f"models/{model}/curobo/{model}_description_curobo_{emb_sel.value}.yaml",
            )
            for emb_sel in CuRoboEmbodimentSelection
        }

    @property
    def curobo_attached_object_link_names(self):
        """
        Returns:
            Dict[str, str]: mapping from robot eef link names to the link names of the attached objects
        """
        # By default, sets the standardized path
        return {eef_link_name: f"attached_object_{eef_link_name}" for eef_link_name in self.eef_link_names.values()}

    @property
    def _default_arm_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config to control that
                robot's arm. Uses velocity control by default.
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": None,
                "motor_type": "position",
                "use_delta_commands": True,
                "use_impedances": False,
            }
        return dic

    @property
    def _default_arm_ik_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                Inverse kinematics controller to control this robot's arm
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "InverseKinematicsController",
                "task_name": f"eef_{arm}",
                "control_freq": self._control_freq,
                "reset_joint_pos": self.reset_joint_pos,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
                ),
                "mode": "pose_delta_ori",
                "smoothing_filter_size": 2,
                "workspace_pose_limiter": None,
            }
        return dic

    @property
    def _default_arm_osc_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                operational space controller to control this robot's arm
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "OperationalSpaceController",
                "task_name": f"eef_{arm}",
                "control_freq": self._control_freq,
                "reset_joint_pos": self.reset_joint_pos,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
                ),
                "mode": "pose_delta_ori",
                "workspace_pose_limiter": None,
            }
        return dic

    @property
    def _default_arm_null_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default arm null controller config
                to control this robot's arm i.e. dummy controller
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "NullJointController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "default_goal": self.reset_joint_pos[self.arm_control_idx[arm]],
                "use_impedances": False,
            }
        return dic

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config to control
                this robot's multi finger gripper. Assumes robot gripper idx has exactly two elements
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "mode": "binary",
                "limit_tolerance": 0.001,
                "inverted": self._grasping_direction == "upper",
            }
        return dic

    @property
    def _default_gripper_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default gripper joint controller config
                to control this robot's gripper
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "use_delta_commands": False,
                "use_impedances": False,
            }
        return dic

    @property
    def _default_gripper_null_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default gripper null controller config
                to control this robot's (non-prehensile) gripper i.e. dummy controller
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "NullJointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "default_goal": th.zeros(len(self.gripper_control_idx[arm])),
                "use_impedances": False,
            }
        return dic

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        arm_ik_configs = self._default_arm_ik_controller_configs
        arm_osc_configs = self._default_arm_osc_controller_configs
        arm_joint_configs = self._default_arm_joint_controller_configs
        arm_null_joint_configs = self._default_arm_null_joint_controller_configs
        gripper_pj_configs = self._default_gripper_multi_finger_controller_configs
        gripper_joint_configs = self._default_gripper_joint_controller_configs
        gripper_null_configs = self._default_gripper_null_controller_configs

        # Add arm and gripper defaults, per arm
        for arm in self.arm_names:
            cfg["arm_{}".format(arm)] = {
                arm_ik_configs[arm]["name"]: arm_ik_configs[arm],
                arm_osc_configs[arm]["name"]: arm_osc_configs[arm],
                arm_joint_configs[arm]["name"]: arm_joint_configs[arm],
                arm_null_joint_configs[arm]["name"]: arm_null_joint_configs[arm],
            }
            cfg["gripper_{}".format(arm)] = {
                gripper_pj_configs[arm]["name"]: gripper_pj_configs[arm],
                gripper_joint_configs[arm]["name"]: gripper_joint_configs[arm],
                gripper_null_configs[arm]["name"]: gripper_null_configs[arm],
            }

        return cfg

    def _get_assisted_grasp_joint_type(self, ag_obj, ag_link):
        """
        Check whether an object @obj can be grasped. If so, return the joint type to use for assisted grasping.
        Otherwise, return None.

        Args:
            ag_obj (BaseObject): Object targeted for an assisted grasp
            ag_link (RigidDynamicPrim): Link of the object to be grasped

        Returns:
            (None or str): If obj can be grasped, returns the joint type to use for assisted grasping.
        """

        # Deny objects that are too heavy and are not a non-base link of a fixed-base object)
        mass = ag_link.mass
        if mass > m.ASSIST_GRASP_MASS_THRESHOLD and not (ag_obj.fixed_base and ag_link != ag_obj.root_link):
            return None

        # Otherwise, compute the joint type. We use a fixed joint unless the link is a non-fixed link.
        # A link is non-fixed if it has any non-fixed parent joints.
        joint_type = "FixedJoint"
        for edge in nx.edge_dfs(ag_obj.articulation_tree, ag_link.body_name, orientation="reverse"):
            joint = ag_obj.articulation_tree.edges[edge[:2]]
            if joint["joint_type"] != JointType.JOINT_FIXED:
                joint_type = "SphericalJoint"
                break

        return joint_type

    def _establish_grasp_rigid(self, arm="default", ag_data=None, contact_pos=None):
        """
        Establishes an ag-assisted grasp, if enabled.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 2-tuple): if specified, assisted-grasp object, link tuple (i.e. :(BaseObject, RigidDynamicPrim)).
                Otherwise, does a no-op
            contact_pos (None or th.tensor): if specified, contact position to use for grasp.
        """
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return
        ag_obj, ag_link = ag_data
        # Get the appropriate joint type
        joint_type = self._get_assisted_grasp_joint_type(ag_obj, ag_link)
        if joint_type is None:
            return

        if contact_pos is None:
            force_data, _ = self._find_gripper_contacts(arm=arm, return_contact_positions=True)
            for c_link_prim_path, c_contact_pos in force_data:
                if c_link_prim_path == ag_link.prim_path:
                    contact_pos = c_contact_pos
                    break

        assert contact_pos is not None, (
            "contact_pos in self._find_gripper_contacts(return_contact_positions=True) is not found in "
            "self._find_gripper_contacts(return_contact_positions=False). This is likely because "
            "GripperRigidContactAPI.get_contact_pairs and get_contact_data return inconsistent results."
        )

        # Joint frame set at the contact point
        # Need to find distance between robot and contact point in robot link's local frame and
        # ag link and contact point in ag link's local frame
        joint_frame_pos = contact_pos
        joint_frame_orn = th.tensor([0, 0, 0, 1.0])
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        parent_frame_pos, parent_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, eef_link_pos, eef_link_orn
        )
        obj_link_pos, obj_link_orn = ag_link.get_position_orientation()
        child_frame_pos, child_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, obj_link_pos, obj_link_orn
        )

        # Create the joint
        joint_prim_path = f"{self.eef_links[arm].prim_path}/ag_constraint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=self.eef_links[arm].prim_path,
            body1=ag_link.prim_path,
            enabled=True,
            exclude_from_articulation=True,
            joint_frame_in_parent_frame_pos=parent_frame_pos / self.scale,
            joint_frame_in_parent_frame_quat=parent_frame_orn,
            joint_frame_in_child_frame_pos=child_frame_pos / ag_obj.scale,
            joint_frame_in_child_frame_quat=child_frame_orn,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        assist_force = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
        max_force = assist_force if joint_type == "FixedJoint" else assist_force * m.ARTICULATED_ASSIST_FRACTION
        # joint_prim.GetAttribute("physics:breakForce").Set(max_force)

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
            "contact_pos": contact_pos,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True

    def _handle_assisted_grasping(self):
        """
        Handles assisted grasping by creating or removing constraints.
        """
        # Loop over all arms
        for arm in self.arm_names:
            # We apply a threshold based on the control rather than the command here so that the behavior
            # stays the same across different controllers and control modes (absolute / delta). This way,
            # a zero action will actually keep the AG setting where it already is.
            controller = self._controllers[f"gripper_{arm}"]
            controlled_joints = controller.dof_idx
            control = cb.to_torch(controller.control)
            if control is None:
                applying_grasp = False
            elif self._grasping_direction == "lower":
                applying_grasp = (
                    th.any(control < self.joint_upper_limits[controlled_joints])
                    if controller.control_type == ControlType.POSITION
                    else th.any(control < 0)
                )
            else:
                applying_grasp = (
                    th.any(control > self.joint_lower_limits[controlled_joints])
                    if controller.control_type == ControlType.POSITION
                    else th.any(control > 0)
                )
            # Execute gradual release of object
            if self._ag_obj_in_hand[arm]:
                if self._ag_release_counter[arm] is not None:
                    self._handle_release_window(arm=arm)
                else:
                    if gm.AG_CLOTH:
                        self._update_constraint_cloth(arm=arm)

                    if not applying_grasp:
                        self._release_grasp(arm=arm)
            elif applying_grasp:
                current_ag_data = self._calculate_in_hand_object(arm=arm)
                if self._ag_grasp_counter[arm] is not None:
                    # We're in a grasp window already
                    if current_ag_data is None:
                        # Lost contact with object, reset window
                        self._ag_grasp_counter[arm] = None
                    else:
                        self._ag_grasp_counter[arm] += 1

                        # Check if window is complete
                        time_in_grasp = self._ag_grasp_counter[arm] * og.sim.get_physics_dt()
                        if time_in_grasp >= m.GRASP_WINDOW:
                            # Establish grasp with the LATEST ag_data
                            self._establish_grasp(arm=arm, ag_data=current_ag_data)
                            # Reset the grasp window tracking
                            self._ag_grasp_counter[arm] = None
                elif current_ag_data is not None:
                    # Start tracking a new potential grasp
                    self._ag_grasp_counter[arm] = 0
            else:
                # Not trying to grasp, reset any pending grasp window
                self._ag_grasp_counter[arm] = None

    def _update_constraint_cloth(self, arm="default"):
        """
        Update the AG constraint for cloth: for the fixed joint between the attachment point and the world, we set
        the local pos to match the current eef link position plus the attachment_point_pos_local offset. As a result,
        the joint will drive the attachment point to the updated position, which will then drive the cloth.
        See _establish_grasp_cloth for more details.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        attachment_point_pos_local = self._ag_obj_constraint_params[arm]["attachment_point_pos_local"]
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        attachment_point_pos, _ = T.pose_transform(
            eef_link_pos, eef_link_orn, attachment_point_pos_local, th.tensor([0, 0, 0, 1], dtype=th.float32)
        )
        joint_prim = self._ag_obj_constraints[arm]
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*attachment_point_pos))

    def _calculate_in_hand_object(self, arm="default"):
        if gm.AG_CLOTH:
            return self._calculate_in_hand_object_cloth(arm)
        else:
            return self._calculate_in_hand_object_rigid(arm)

    def _establish_grasp(self, arm="default", ag_data=None, contact_pos=None):
        if gm.AG_CLOTH:
            return self._establish_grasp_cloth(arm, ag_data)
        else:
            return self._establish_grasp_rigid(arm, ag_data, contact_pos)

    def _calculate_in_hand_object_cloth(self, arm="default"):
        """
        Same as _calculate_in_hand_object_rigid, except for cloth. Only one should be used at any given time.

        Calculates which object to assisted-grasp for arm @arm. Returns an (BaseObject, RigidDynamicPrim, th.Tensor) tuple or
        None if no valid AG-enabled object can be found.

        1) Check if the gripper is closed enough
        2) Go through each of the cloth object, and check if its attachment point link position is within the "ghost"
        box volume of the gripper link.

        Only returns the first valid object and ignore the rest.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            None or 3-tuple: If a valid assisted-grasp object is found,
                returns the corresponding (object, object_link, attachment_point_position), i.e.
                ((BaseObject, RigidDynamicPrim, th.Tensor)) to the contacted in-hand object. Otherwise, returns None
        """
        # TODO (eric): Assume joint_pos = 0 means fully closed
        GRIPPER_FINGER_CLOSE_THRESHOLD = 0.03
        gripper_finger_pos = self.get_joint_positions()[self.gripper_control_idx[arm]]
        gripper_finger_close = th.sum(gripper_finger_pos) < GRIPPER_FINGER_CLOSE_THRESHOLD
        if not gripper_finger_close:
            return None

        cloth_objs = self.scene.object_registry("prim_type", PrimType.CLOTH)
        if cloth_objs is None:
            return None

        # TODO (eric): Only AG one cloth at any given moment.
        # Returns the first cloth that overlaps with the "ghost" box volume
        for cloth_obj in cloth_objs:
            attachment_point_pos = cloth_obj.links["attachment_point"].get_position_orientation()[0]
            particles_in_volume = self.eef_links[arm].check_points_in_volume([attachment_point_pos])
            if particles_in_volume.sum() > 0:
                return cloth_obj, cloth_obj.links["attachment_point"], attachment_point_pos

        return None

    def _establish_grasp_cloth(self, arm="default", ag_data=None):
        """
        Same as _establish_grasp_cloth, except for cloth. Only one should be used at any given time.
        Establishes an ag-assisted grasp, if enabled.

        Create a fixed joint between the attachment point link of the cloth object and the world.
        In theory, we could have created a fixed joint to the eef link, but omni doesn't support this as the robot has
        an articulation root API attached to it, which is incompatible with the attachment API.

        We also store attachment_point_pos_local as the attachment point position in the eef link frame when the fixed
        joint is created. As the eef link frame changes its pose, we will use attachment_point_pos_local to figure out
        the new attachment_point_pos in the world frame and set the fixed joint to there. See _update_constraint_cloth
        for more details.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 3-tuple): If specified, should be the corresponding
                (object, object_link, attachment_point_position), i.e. ((BaseObject, RigidPrim, th.Tensor)) to the]
                contacted in-hand object
        """
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return

        ag_obj, ag_link, attachment_point_pos = ag_data

        # Find the attachment point position in the eef frame
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        attachment_point_pos_local, _ = T.relative_pose_transform(
            attachment_point_pos, th.tensor([0, 0, 0, 1], dtype=th.float32), eef_link_pos, eef_link_orn
        )

        # Create the joint
        joint_prim_path = f"{ag_link.prim_path}/ag_constraint"
        joint_type = "FixedJoint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=ag_link.prim_path,
            body1=None,
            enabled=False,
            exclude_from_articulation=True,
            joint_frame_in_child_frame_pos=attachment_point_pos,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        max_force = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
        # joint_prim.GetAttribute("physics:breakForce").Set(max_force)

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
            "attachment_point_pos_local": attachment_point_pos_local,
            "contact_pos": attachment_point_pos,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True

    def _dump_state(self):
        # Call super first
        state = super()._dump_state()

        # If we're using actual physical grasping, no extra state needed to save
        if self.grasping_mode == "physical":
            return state

        # Include AG_state
        ag_params = self._ag_obj_constraint_params.copy()
        for arm in ag_params.keys():
            if len(ag_params[arm]) > 0:
                assert self.scene is not None, "Cannot get position and orientation relative to scene without a scene"
                ag_params[arm]["contact_pos"], _ = self.scene.convert_world_pose_to_scene_relative(
                    ag_params[arm]["contact_pos"],
                    th.tensor([0, 0, 0, 1], dtype=th.float32),
                )
        state["ag_obj_constraint_params"] = ag_params
        return state

    def _load_state(self, state):
        super()._load_state(state=state)

        # No additional loading needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return

        # Include AG_state
        # TODO: currently does not take care of cloth objects
        # TODO: add unit tests
        for arm in self.arm_names:
            current_ag_constraint = self._ag_obj_constraint_params[arm]
            has_current_constraint = len(current_ag_constraint) > 0

            loaded_ag_constraint = {}
            has_loaded_constraint = False
            if "ag_obj_constraint_params" in state:
                loaded_ag_constraint = state["ag_obj_constraint_params"][arm]
                has_loaded_constraint = len(loaded_ag_constraint) > 0

            # Release existing grasp if needed
            should_release = False
            if has_current_constraint:
                if not has_loaded_constraint:
                    should_release = True
                else:
                    # Check if constraints are different
                    are_equal = current_ag_constraint.keys() == loaded_ag_constraint.keys() and all(
                        th.equal(v1, v2) if isinstance(v1, th.Tensor) and isinstance(v2, th.Tensor) else v1 == v2
                        for v1, v2 in zip(current_ag_constraint.values(), loaded_ag_constraint.values())
                    )
                    should_release = not are_equal

            if should_release:
                self.release_grasp_immediately(arm=arm)

            # Establish new grasp if needed
            if has_loaded_constraint and (not has_current_constraint or should_release):
                obj = self.scene.object_registry("prim_path", loaded_ag_constraint["ag_obj_prim_path"])
                link = obj.links[loaded_ag_constraint["ag_link_prim_path"].split("/")[-1]]
                contact_pos_global = loaded_ag_constraint["contact_pos"]
                assert self.scene is not None, "Cannot set position and orientation relative to scene without a scene"
                contact_pos_global, _ = self.scene.convert_scene_relative_pose_to_world(
                    contact_pos_global,
                    th.tensor([0, 0, 0, 1], dtype=th.float32),
                )
                self._establish_grasp(arm=arm, ag_data=(obj, link), contact_pos=contact_pos_global)

    def serialize(self, state):
        # Call super first
        state_flat = super().serialize(state=state)

        # No additional serialization needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return state_flat

        # TODO AG
        return state_flat

    def deserialize(self, state):
        # Call super first
        state_dict, idx = super().deserialize(state=state)

        # No additional deserialization needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return state_dict, idx

        # TODO AG
        return state_dict, idx

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ManipulationRobot")
        return classes

    @property
    def teleop_rotation_offset(self):
        """
        Rotational offset that will be applied for teleoperation
        such that [0, 0, 0, 1] as action will keep the robot eef pointing at +x axis
        """
        return {arm: th.tensor([0, 0, 0, 1]) for arm in self.arm_names}

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        """
        Generate action data from teleoperation action data
        NOTE: This implementation only supports IK/OSC controller for arm and MultiFingerGripperController for gripper.
        Overwrite this function if the robot is using a different controller.
        Args:
            teleop_action (TeleopAction): teleoperation action data
        Returns:
            th.tensor: array of action data for arm and gripper
        """
        action = super().teleop_data_to_action(teleop_action)
        hands = ["left", "right"] if self.n_arms == 2 else ["right"]
        for i, hand in enumerate(hands):
            arm_name = self.arm_names[i]
            arm_action = th.tensor(teleop_action[hand]).float()
            # arm action
            assert isinstance(self._controllers[f"arm_{arm_name}"], InverseKinematicsController) or isinstance(
                self._controllers[f"arm_{arm_name}"], OperationalSpaceController
            ), f"Only IK and OSC controllers are supported for arm {arm_name}!"
            target_pos, target_orn = arm_action[:3], T.quat2axisangle(T.euler2quat(arm_action[3:6]))
            action[self.arm_action_idx[arm_name]] = th.cat((target_pos, target_orn))
            # gripper action
            assert isinstance(
                self._controllers[f"gripper_{arm_name}"], MultiFingerGripperController
            ), f"Only MultiFingerGripperController is supported for gripper {arm_name}!"
            action[self.gripper_action_idx[arm_name]] = arm_action[6]
        return action
