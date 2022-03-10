from abc import abstractmethod
from collections import namedtuple
from enum import IntEnum

import numpy as np


import igibson.utils.transform_utils as T
from igibson.controllers import (
    ControlType,
    JointController,
    ManipulationController,
    MultiFingerGripperController,
    NullGripperController,
)
# from igibson.external.pybullet_tools.utils import (
    ContactResult,
    get_child_frame_pose,
    get_constraint_violation,
    get_link_pose,
)
from igibson.robots.robot_base import BaseRobot
from igibson.utils.python_utils import assert_valid_key


class IsGraspingState(IntEnum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


AG_MODES = {
    "physical",
    "assisted",
    "sticky",
}

# Assisted grasping parameters
VISUALIZE_RAYS = False
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
RELEASE_WINDOW = 1 / 30.0  # release window in seconds
GraspingPoint = namedtuple("GraspingPoint", ["link_name", "position"])  # link_name (str), position (x,y,z tuple)

# is_grasping heuristics parameters
POS_TOLERANCE = 0.002  # arbitrary heuristic
VEL_TOLERANCE = 0.01  # arbitrary heuristic


def get_relative_pose(origin_body, origin_link, target_body, target_link):
    """Get the pose of target related to origin."""
    origin_in_world_frame = get_link_pose(origin_body, origin_link)
    target_in_world_frame = get_link_pose(target_body, target_link)
    world_in_origin_frame = p.invertTransform(*origin_in_world_frame)
    target_in_origin_frame = p.multiplyTransforms(*world_in_origin_frame, *target_in_world_frame)
    return target_in_origin_frame


def set_coll_filter(target_body_id, source_links, enable):
    # TODO: mostly shared with behavior robot, can be factored out
    """
    Sets collision filters for body - to enable or disable them
    :param target_body_id: physics body to enable/disable collisions with
    :param source_links: RobotLink objects to disable collisions with
    :param enable: whether to enable/disable collisions
    """
    target_link_idxs = [-1] + list(range(p.getNumJoints(target_body_id)))

    for link in source_links:
        for target_link_idx in target_link_idxs:
            p.setCollisionFilterPair(link.body_id, target_body_id, link.link_id, target_link_idx, 1 if enable else 0)


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
        name=None,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        reset_joint_pos=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=False,
        class_id=None,
        rendering_params=None,
        grasping_mode="physical",
    ):
        """
        :param name: None or str, name of the robot object
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the first link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        """
        # Store relevant internal vars
        assert_valid_key(key=grasping_mode, valid_keys=AG_MODES, name="grasping_mode")
        self.grasping_mode = grasping_mode

        # Initialize other variables used for assistive grasping
        self._ag_data = {arm: None for arm in self.arm_names}
        self._ag_freeze_joint_pos = {
            arm: {} for arm in self.arm_names
        }  # Frozen positions for keeping fingers held still
        self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
        self._ag_obj_cid = {arm: None for arm in self.arm_names}
        self._ag_obj_cid_params = {arm: {} for arm in self.arm_names}
        self._ag_freeze_gripper = {arm: None for arm in self.arm_names}
        self._ag_release_counter = {arm: None for arm in self.arm_names}

        # Call super() method
        super().__init__(
            name=name,
            control_freq=control_freq,
            action_type=action_type,
            action_normalize=action_normalize,
            proprio_obs=proprio_obs,
            reset_joint_pos=reset_joint_pos,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            class_id=class_id,
            self_collision=self_collision,
            rendering_params=rendering_params,
        )

    def _validate_configuration(self):
        # Iterate over all arms
        for arm in self.arm_names:
            # We make sure that our arm controller exists and is a manipulation controller
            assert (
                "arm_{}".format(arm) in self._controllers
            ), "Controller 'arm_{}' must exist in controllers! Current controllers: {}".format(
                arm, list(self._controllers.keys())
            )
            assert isinstance(
                self._controllers["arm_{}".format(arm)], ManipulationController
            ), "Arm {} controller must be a ManipulationController!".format(arm)

            # We make sure that our gripper controller exists and is a manipulation controller
            assert (
                "gripper_{}".format(arm) in self._controllers
            ), "Controller 'gripper_{}' must exist in controllers! Current controllers: {}".format(
                arm, list(self._controllers.keys())
            )
            assert isinstance(
                self._controllers["gripper_{}".format(arm)], ManipulationController
            ), "Gripper {} controller must be a ManipulationController!".format(arm)

        # run super
        super()._validate_configuration()

    def is_grasping(self, arm="default", candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        :param arm: str, specific arm to check for grasping. Default is "default" which corresponds to the first entry
        in self.arm_names
        :param candidate_obj: body ID or None, object to check if this robot is currently grasping. If None, then
            will be a general (object-agnostic) check for grasping.
            Note: if self.grasping_mode is "physical", then @candidate_obj will be ignored completely

        :return int: For the specific manipulator appendage, returns IsGraspingState.TRUE if it is grasping
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
            gripper_controller = self._controllers["gripper_{}".format(arm)]

            # NullGripperController cannot grasp anything
            if isinstance(gripper_controller, NullGripperController):
                is_grasping = IsGraspingState.FALSE

            # JointController does not have any good heuristics to determine is_grasping
            elif isinstance(gripper_controller, JointController):
                is_grasping = IsGraspingState.UNKNOWN

            elif isinstance(gripper_controller, MultiFingerGripperController):
                # Independent mode of MultiFingerGripperController does not have any good heuristics to determine is_grasping
                if gripper_controller.mode == "independent":
                    is_grasping = IsGraspingState.UNKNOWN

                # No control has been issued before
                elif gripper_controller.control is None:
                    is_grasping = IsGraspingState.FALSE

                else:
                    assert np.all(
                        gripper_controller.control == gripper_controller.control[0]
                    ), "MultiFingerGripperController has different values in the command for non-independent mode"

                    assert POS_TOLERANCE > gripper_controller.limit_tolerance, (
                        "Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the "
                        "gripper controller's tolerance of zero-ing out velocities, which makes the heuristics invalid."
                    )

                    finger_pos = self.joint_positions[self.gripper_control_idx[arm]]

                    # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
                    if (
                        gripper_controller.motor_type == "position"
                        and np.mean(np.abs(finger_pos - gripper_controller.control)) < POS_TOLERANCE
                    ):
                        is_grasping = IsGraspingState.UNKNOWN

                    # For joint velocity / torque control, if the desired velocities / torques are zeros, is_grasping unknown
                    elif (
                        gripper_controller.motor_type in {"velocity", "torque"}
                        and np.mean(np.abs(gripper_controller.control)) < VEL_TOLERANCE
                    ):
                        is_grasping = IsGraspingState.UNKNOWN

                    # Otherwise, the last control signal intends to "move" the gripper
                    else:
                        finger_pos = self.joint_positions[self.gripper_control_idx[arm]]
                        finger_vel = self.joint_velocities[self.gripper_control_idx[arm]]
                        min_pos = self.joint_lower_limits[self.gripper_control_idx[arm]]
                        max_pos = self.joint_upper_limits[self.gripper_control_idx[arm]]

                        # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                        assert np.all(
                            (min_pos <= finger_pos) * (finger_pos <= max_pos)
                        ), "Got invalid finger joint positions when checking for grasp!"

                        # Check distance from both ends of the joint limits
                        dist_from_lower_limit = finger_pos - min_pos
                        dist_from_upper_limit = max_pos - finger_pos

                        # If the joint positions are not near the joint limits with some tolerance (POS_TOLERANCE)
                        valid_grasp_pos = (
                            np.mean(dist_from_lower_limit) > POS_TOLERANCE
                            and np.mean(dist_from_upper_limit) > POS_TOLERANCE
                        )

                        # And the joint velocities are close to zero with some tolerance (VEL_TOLERANCE)
                        valid_grasp_vel = np.all(np.abs(finger_vel) < VEL_TOLERANCE)

                        # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                        is_grasping = (
                            IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE
                        )

            else:
                # Add more cases once we have more gripper controllers available
                raise Exception("Unexpected gripper controller type: {}".format(type(gripper_controller)))

        # Return as a numerical array
        return is_grasping

    def _find_gripper_raycast_collisions(self, arm="default"):
        """
        For arm @arm, calculate any body IDs and corresponding link IDs that are not part of the robot
        itself that intersect with rays cast between any of the gripper's start and end points

        :param arm: str, specific arm whose gripper will be checked for raycast collisions. Default is "default"
            which corresponds to the first entry in self.arm_names

        :return set[tuple[int, int]]: set of unique (body_id, link_id) detected raycast intersections that
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
        startpoints = []
        endpoints = []
        for grasp_start_point in self.assisted_grasp_start_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_start_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to startpoints
            start_point, _ = p.multiplyTransforms(link_pos, link_orn, grasp_start_point.position, [0, 0, 0, 1])
            startpoints.append(start_point)
        # Repeat for end points
        for grasp_end_point in self.assisted_grasp_end_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_end_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to endpoints
            end_point, _ = p.multiplyTransforms(link_pos, link_orn, grasp_end_point.position, [0, 0, 0, 1])
            endpoints.append(end_point)
        # Stack the start points and repeat the end points, and add these values to the raycast dicts
        n_startpoints, n_endpoints = len(startpoints), len(endpoints)
        raycast_startpoints = startpoints * n_endpoints
        raycast_endpoints = []
        for endpoint in endpoints:
            raycast_endpoints += [endpoint] * n_startpoints

        # Potentially visualize rays for debugging
        if VISUALIZE_RAYS:
            for f, t in zip(raycast_startpoints, raycast_endpoints):
                p.addUserDebugLine(f, t, [1, 0, 0], 0.01, lifeTime=0.5)

        # Calculate raycasts from each start point to end point -- this is n_startpoints * n_endpoints total rays
        ray_results = []
        # Repeat twice, so that we avoid collisions with the fingers of each gripper themself
        for i in range(2):
            ray_results += p.rayTestBatch(
                raycast_startpoints,
                raycast_endpoints,
                numThreads=0,
                fractionEpsilon=1.0,  # Set to 1.0 so we don't trigger multiple same-body hits
                reportHitNumber=i,
            )
        # We filter the results, storing them in a set to reduce redundancy, and removing all
        # self-intersecting values. If both these conditions are met, we store a tuple of (body ID, link ID) for
        # each intersection. If no results are found, this will be an empty set
        ray_data = set(
            [(ray_res[0], ray_res[1]) for ray_res in ray_results if ray_res[0] not in {-1, self.eef_links[arm].body_id}]
        )

        return ray_data

    def _find_gripper_contacts(self, arm="default", return_contact_positions=False):
        """
        For arm @arm, calculate any body IDs and corresponding link IDs that are not part of the robot
        itself that are in contact with any of this arm's gripper's fingers

        :param arm: str, specific arm whose gripper will be checked for contact. Default is "default" which
            corresponds to the first entry in self.arm_names
        :param return_contact_positions: bool, if True, will additionally return the contact (x,y,z) position

        :return set[tuple[int, int[, Array]]], dict[tuple(int, int): set{int}]: first return value is set of unique
            (body_id, link_id) contact candidates that are not the robot itself. If @return_contact_positions is True,
            then returns (body_id, link_id, pos), where pos is the contact (x,y,z) position
            Note: if no objects that are not the robot itself are intersecting, the set will be empty.
            Second return value is dictionary mapping unique contact objects defined by the tuple (body_id, link_id) to
            set of unique link_ids of the robot that it is in contact with
        """
        arm = self.default_arm if arm == "default" else arm
        robot_contact_links = dict()
        contact_data = set()
        # Find all objects in contact with all finger joints for this arm
        con_results = [
            ContactResult(*res[:10])
            for link in self.finger_links[arm]
            for res in p.getContactPoints(bodyA=link.body_id, linkIndexA=link.link_id)
        ]
        for con_res in con_results:
            # Only add this contact if it's not a robot self-collision
            if con_res.bodyUniqueIdB not in self.get_body_ids():
                # Add to contact data
                obj_con_info = (con_res.bodyUniqueIdB, con_res.linkIndexB)
                contact_data.add((*obj_con_info, con_res.positionOnA) if return_contact_positions else obj_con_info)
                # Also add robot contact link info
                if obj_con_info not in robot_contact_links:
                    robot_contact_links[obj_con_info] = set()
                robot_contact_links[obj_con_info].add(con_res.linkIndexA)

        return contact_data, robot_contact_links

    def set_position_orientation(self, pos, quat):
        # Store the original EEF poses.
        original_poses = {}
        for arm in self.arm_names:
            original_poses[arm] = self.get_eef_position(arm)

        super(ManipulationRobot, self).set_position_orientation(pos, quat)

        # Now for each hand, if it was holding an AG object, teleport it.
        for arm in self.arm_names:
            if self._ag_obj_in_hand[arm] is not None:
                inv_original_pos, inv_original_orn = p.invertTransform(*original_poses[arm])
                local_pos, local_orn = p.multiplyTransforms(
                    inv_original_pos, inv_original_orn, *p.getBasePositionAndOrientation(self._ag_obj_in_hand[arm])
                )
                new_pos, new_orn = p.multiplyTransforms(
                    self.get_eef_position(arm), self.get_eef_orientation(arm), local_pos, local_orn
                )
                p.resetBasePositionAndOrientation(self._ag_obj_in_hand[arm], new_pos, new_orn)

    def apply_action(self, action):
        # First run assisted grasping
        if self.grasping_mode != "physical":
            self._handle_assisted_grasping(action=action)

        # Potentially freeze gripper joints
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                self._freeze_gripper(arm)

        # Run super method as normal
        super().apply_action(action)

    def _deploy_control(self, control, control_type):
        # We intercept the gripper control and replace it with velocity=0 if we're freezing our gripper
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                control[self.gripper_control_idx[arm]] = 0.0
                control_type[self.gripper_control_idx[arm]] = ControlType.VELOCITY

        super()._deploy_control(control=control, control_type=control_type)

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        :param arm: str, specific arm whose grasp will be released.
            Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        p.removeConstraint(self._ag_obj_cid[arm])
        self._ag_data[arm] = None
        self._ag_obj_cid[arm] = None
        self._ag_obj_cid_params[arm] = {}
        self._ag_freeze_gripper[arm] = False
        self._ag_release_counter[arm] = 0

    def get_control_dict(self):
        # In addition to super method, add in EEF states
        dic = super().get_control_dict()

        for arm in self.arm_names:
            dic["eef_{}_pos_relative".format(arm)] = self.get_relative_eef_position(arm)
            dic["eef_{}_quat_relative".format(arm)] = self.get_relative_eef_orientation(arm)

        return dic

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Loop over all arms to grab proprio info
        for arm in self.arm_names:
            # Add arm info
            dic["arm_{}_qpos".format(arm)] = self.joint_positions[self.arm_control_idx[arm]]
            dic["arm_{}_qpos_sin".format(arm)] = np.sin(self.joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qpos_cos".format(arm)] = np.cos(self.joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qvel".format(arm)] = self.joint_velocities[self.arm_control_idx[arm]]

            # Add eef and grasping info
            dic["eef_{}_pos_global".format(arm)] = self.get_eef_position(arm)
            dic["eef_{}_quat_global".format(arm)] = self.get_eef_orientation(arm)
            dic["eef_{}_pos".format(arm)] = self.get_relative_eef_position(arm)
            dic["eef_{}_quat".format(arm)] = self.get_relative_eef_orientation(arm)
            dic["grasp_{}".format(arm)] = np.array([self.is_grasping(arm)])
            dic["gripper_{}_qpos".format(arm)] = self.joint_positions[self.gripper_control_idx[arm]]
            dic["gripper_{}_qvel".format(arm)] = self.joint_velocities[self.gripper_control_idx[arm]]

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
    def controller_order(self):
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

    @property
    def n_arms(self):
        """
        :return int: Number of arms this robot has. Returns 1 by default
        """
        return 1

    @property
    def arm_names(self):
        """
        :return Array[str]: List of arm names for this robot. Should correspond to the keys used to index into
            arm- and gripper-related dictionaries, e.g.: eef_link_names, finger_link_names, etc.
            Default is string enumeration based on @self.n_arms.
        """
        return [str(i) for i in range(self.n_arms)]

    @property
    def default_arm(self):
        """
        :return str: Default arm name for this robot, corresponds to the first entry in @arm_names by default
        """
        return self.arm_names[0]

    @property
    @abstractmethod
    def eef_link_names(self):
        """
        :return dict[str, str]: Dictionary mapping arm appendage name to corresponding name of the EEF link,
            should correspond to specific link name in this robot's underlying model file
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_link_names(self):
        """
        :return dict[str, list]: Dictionary mapping arm appendage name to array of link names corresponding to
            this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_joint_names(self):
        """
        :return dict[str, list]: Dictionary mapping arm appendage name to array of joint names corresponding to
            this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        raise NotImplementedError

    @property
    def eef_links(self):
        """
        :return dict[str, RobotLink]: Dictionary mapping arm appendage name to robot link corresponding to that arm's
            eef link
        """
        return {arm: self._links[self.eef_link_names[arm]] for arm in self.arm_names}

    @property
    def finger_links(self):
        """
        :return dict[str, Array[RobotLink]]: Dictionary mapping arm appendage name to robot links corresponding to
            that arm's finger links
        """
        return {arm: [self._links[link] for link in self.finger_link_names[arm]] for arm in self.arm_names}

    @property
    def finger_joints(self):
        """
        :return dict[str, Array[RobotJoint]]: Dictionary mapping arm appendage name to robot joints corresponding to
            that arm's finger joints
        """
        return {arm: [self._joints[joint] for joint in self.finger_joint_names[arm]] for arm in self.arm_names}

    @property
    def assisted_grasp_start_points(self):
        """
        :return dict[str, None or Array[GraspingPoint]]: Dictionary mapping individual
            arm appendage names to array of GraspingPoint tuples, composed of (link_name, position) values
            specifying valid grasping start points located at cartesian (x,y,z) coordinates specified in link_name's
            local coordinate frame. These values will be used in conjunction with
            @self.assisted_grasp_end_points to trigger assisted grasps, where objects that intersect
            with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
            @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
            appendage). By default, each entry returns None, and must be implemented by any robot subclass that
            wishes to use assisted grasping.
        """
        return {arm: None for arm in self.arm_names}

    @property
    def assisted_grasp_end_points(self):
        """
        :return dict[str, None or Array[GraspingPoint]]: Dictionary mapping individual
            arm appendage names to array of GraspingPoint tuples, composed of (link_name, position) values
            specifying valid grasping end points located at cartesian (x,y,z) coordinates specified in link_name's
            local coordinate frame. These values will be used in conjunction with
            @self.assisted_grasp_start_points to trigger assisted grasps, where objects that intersect
            with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
            @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
            appendage). By default, each entry returns None, and must be implemented by any robot subclass that
            wishes to use assisted grasping.
        """
        return {arm: None for arm in self.arm_names}

    @property
    def eef_link_ids(self):
        """
        :return dict[str, int]: Dictionary mapping arm appendage name to link ID corresponding to the eef link
        """
        return {arm_name: self._links[eef_name].link_id for arm_name, eef_name in self.eef_link_names.items()}

    @property
    def finger_link_ids(self):
        """
        :return dict[str, list]: Dictionary mapping arm appendage name to link IDs corresponding to the eef fingers
        """
        return {
            arm_name: [self._links[link].link_id for link in link_names]
            for arm_name, link_names in self.finger_link_names.items()
        }

    @property
    def finger_joint_ids(self):
        """
        :return dict[str, list]: Dictionary mapping arm appendage name to joint IDs corresponding to the eef fingers
        """
        return {
            arm_name: [self._joints[joint].joint_id for joint in joint_names]
            for arm_name, joint_names in self.finger_joint_names.items()
        }

    @property
    def gripper_link_to_grasp_point(self):
        """
        :return dict[str, Array[float]]: Dictionary mapping arm appendage name to (dx,dy,dz) relative distance from
            the gripper link frame to the expected center of the robot's grasping point.
            Unique to each robot embodiment.
        """
        raise NotImplementedError

    def get_eef_position(self, arm="default"):
        """
        :param arm: str, specific arm to grab eef position. Default is "default" which corresponds to the first entry
        in self.arm_names

        :return Array[float]: (x,y,z) global end-effector Cartesian position for this robot's end-effector corresponding
            to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        return self._links[self.eef_link_names[arm]].get_position()

    def get_eef_orientation(self, arm="default"):
        """
        :param arm: str, specific arm to grab eef orientation. Default is "default" which corresponds to the first entry
        in self.arm_names

        :return Array[float]: (x,y,z,w) global quaternion orientation for this robot's end-effector corresponding
            to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        return self._links[self.eef_link_names[arm]].get_orientation()

    def get_relative_eef_pose(self, arm="default", mat=False):
        """
        :param arm: str, specific arm to grab eef pose. Default is "default" which corresponds to the first entry
        in self.arm_names
        :param mat: bool, whether to return pose in matrix form (mat=True) or (pos, quat) tuple (mat=False)

        :return Tuple[Array[float], Array[float]] or Array[Array[float]]: End-effector pose, either in 4x4 homogeneous
            matrix form (if @mat=True) or (pos, quat) tuple (if @mat=False), corresponding to arm @arm
        """
        arm = self.default_arm if arm == "default" else arm
        pose = get_relative_pose(
            self.base_link.body_id, self.base_link.link_id, self.eef_links[arm].body_id, self.eef_links[arm].link_id
        )
        return T.pose2mat(pose) if mat else pose

    def get_relative_eef_position(self, arm="default"):
        """
        :param arm: str, specific arm to grab relative eef pos.
        Default is "default" which corresponds to the first entry in self.arm_names

        :return Array[float]: (x,y,z) Cartesian position of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[0]

    def get_relative_eef_orientation(self, arm="default"):
        """
        :param arm: str, specific arm to grab relative eef orientation.
        Default is "default" which corresponds to the first entry in self.arm_names

        :return Array[float]: (x,y,z,z) quaternion orientation of end-effector relative to robot base frame
        """
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[1]

    def _calculate_in_hand_object(self, arm="default"):
        """
        Calculates which object to assisted-grasp for arm @arm. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.

        :param arm: str, specific arm to calculate in-hand object for.
        Default is "default" which corresponds to the first entry in self.arm_names

        :return None or Tuple[int, int]: If a valid assisted-grasp object is found, returns the corresponding
            (object_id, link_id) corresponding to the contact point of that object. Otherwise, returns None
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
        gripper_pos, gripper_orn = self.eef_links[arm].get_position_orientation()
        gripper_center_pos, _ = p.multiplyTransforms(
            gripper_pos, gripper_orn, self.gripper_link_to_grasp_point[arm], [0, 0, 0, 1]
        )

        candidate_data = []
        for (body_id, link_id) in candidates_set:
            # Calculate position of the object link
            link_pos, _ = (
                p.getBasePositionAndOrientation(body_id) if link_id == -1 else p.getLinkState(body_id, link_id)[:2]
            )
            dist = np.linalg.norm(np.array(link_pos) - np.array(gripper_center_pos))
            candidate_data.append((body_id, link_id, dist))

        candidate_data = sorted(candidate_data, key=lambda x: x[2])
        ag_bid, ag_link, _ = candidate_data[0]

        # Make sure the ag body_id is not a self collision
        assert ag_bid not in self.get_body_ids(), "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[(ag_bid, ag_link)]
        touching_at_least_two_fingers = len(set(self.finger_link_ids[arm]).intersection(robot_contacts)) >= 2

        # Return None if object cannot be assisted grasped or not touching at least two fingers
        if (not self.simulator.can_assisted_grasp(ag_bid, ag_link)) or (not touching_at_least_two_fingers):
            return None

        return ag_bid, ag_link

    def _handle_release_window(self, arm="default"):
        """
        Handles releasing an object from arm @arm

        :param arm: str, specific arm to handle release window.
        Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        self._ag_release_counter[arm] += 1
        time_since_release = self._ag_release_counter[arm] * self.simulator.render_timestep
        if time_since_release >= RELEASE_WINDOW:
            set_coll_filter(
                self._ag_obj_in_hand[arm],
                self.finger_links[arm],
                enable=True,
            )
            self._ag_obj_in_hand[arm] = None
            self._ag_release_counter[arm] = None

    def _freeze_gripper(self, arm="default"):
        """
        Freezes gripper finger joints - used in assisted grasping.

        :param arm: str, specific arm to freeze gripper.
        Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        for joint_name, j_val in self._ag_freeze_joint_pos[arm].items():
            joint = self._joints[joint_name]
            p.resetJointState(joint.body_id, joint.joint_id, targetValue=j_val, targetVelocity=0.0)

    @property
    def _default_arm_joint_controller_configs(self):
        """
        :return: Dict[str, Any] Dictionary mapping arm appendage name to default controller config to control that
            robot's arm. Uses velocity control by default.
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self.control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": "default",
                "use_delta_commands": False,
            }
        return dic

    @property
    def _default_arm_ik_controller_configs(self):
        """
        :return: Dict[str, Any] Dictionary mapping arm appendage name to default controller config for an
            Inverse kinematics controller to control this robot's arm
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "InverseKinematicsController",
                "base_body_id": self.base_link.body_id,
                "task_link_id": self.eef_links[arm].link_id,
                "task_name": "eef_{}".format(arm),
                "control_freq": self.control_freq,
                "default_joint_pos": self.default_joint_pos,
                "joint_damping": self.joint_damping,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    np.array([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
                ),
                "kv": 2.0,
                "mode": "pose_delta_ori",
                "smoothing_filter_size": 2,
                "workspace_pose_limiter": None,
            }
        return dic

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        """
        :return: Dict[str, Any] Dictionary mapping arm appendage name to default controller config to control
            this robot's multi finger gripper. Assumes robot gripper idx has exactly two elements
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self.control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "mode": "binary",
                "limit_tolerance": 0.001,
            }
        return dic

    @property
    def _default_gripper_joint_controller_configs(self):
        """
        :return: Dict[str, Any] Dictionary mapping arm appendage name to default gripper joint controller config
            to control this robot's gripper
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self.control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "use_delta_commands": False,
            }
        return dic

    @property
    def _default_gripper_null_controller_configs(self):
        """
        :return: Dict[str, Any] Dictionary mapping arm appendage name to default gripper null controller config
            to control this robot's (non-prehensile) gripper i.e. dummy controller
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "NullGripperController",
                "control_freq": self.control_freq,
                "control_limits": self.control_limits,
            }
        return dic

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        arm_ik_configs = self._default_arm_ik_controller_configs
        arm_joint_configs = self._default_arm_joint_controller_configs
        gripper_pj_configs = self._default_gripper_multi_finger_controller_configs
        gripper_joint_configs = self._default_gripper_joint_controller_configs
        gripper_null_configs = self._default_gripper_null_controller_configs

        # Add arm and gripper defaults, per arm
        for arm in self.arm_names:
            cfg["arm_{}".format(arm)] = {
                arm_ik_configs[arm]["name"]: arm_ik_configs[arm],
                arm_joint_configs[arm]["name"]: arm_joint_configs[arm],
            }
            cfg["gripper_{}".format(arm)] = {
                gripper_pj_configs[arm]["name"]: gripper_pj_configs[arm],
                gripper_joint_configs[arm]["name"]: gripper_joint_configs[arm],
                gripper_null_configs[arm]["name"]: gripper_null_configs[arm],
            }

        return cfg

    def _establish_grasp(self, arm="default", ag_data=None):
        """
        Establishes an ag-assisted grasp, if enabled.

        :param arm: str, specific arm to establish grasp.
            Default is "default" which corresponds to the first entry in self.arm_names
        :param ag_data: None or Tuple[int, int], if specified, assisted-grasp object body ID and link ID. Otherwise,
            does a no-op
        """
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return
        ag_bid, ag_link = ag_data

        # Create a p2p joint if it's a child link of a fixed URDF that is connected by a revolute or prismatic joint
        if (
            ag_link != -1
            and p.getJointInfo(ag_bid, ag_link)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
            and ag_bid in self.simulator.scene.objects_by_id
            and hasattr(self.simulator.scene.objects_by_id[ag_bid], "fixed_base")
            and self.simulator.scene.objects_by_id[ag_bid].fixed_base
        ):
            joint_type = p.JOINT_POINT2POINT
        else:
            joint_type = p.JOINT_FIXED

        force_data, _ = self._find_gripper_contacts(arm=arm, return_contact_positions=True)
        contact_pos = None
        for c_bid, c_link, c_contact_pos in force_data:
            if (c_bid, c_link) == ag_data:
                contact_pos = c_contact_pos
                break
        assert contact_pos is not None

        # Joint frame set at the contact point
        joint_frame_pos = contact_pos
        joint_frame_orn = [0, 0, 0, 1]
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        inv_eef_link_pos, inv_eef_link_orn = p.invertTransform(eef_link_pos, eef_link_orn)
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
            inv_eef_link_pos, inv_eef_link_orn, joint_frame_pos, joint_frame_orn
        )
        if ag_link == -1:
            obj_pos, obj_orn = p.getBasePositionAndOrientation(ag_bid)
        else:
            obj_pos, obj_orn = p.getLinkState(ag_bid, ag_link)[:2]
        inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
        child_frame_pos, child_frame_orn = p.multiplyTransforms(
            inv_obj_pos, inv_obj_orn, joint_frame_pos, joint_frame_orn
        )
        self._ag_obj_cid[arm] = p.createConstraint(
            parentBodyUniqueId=self.eef_links[arm].body_id,
            parentLinkIndex=self.eef_links[arm].link_id,
            childBodyUniqueId=ag_bid,
            childLinkIndex=ag_link,
            jointType=joint_type,
            jointAxis=(0, 0, 0),
            parentFramePosition=parent_frame_pos,
            childFramePosition=child_frame_pos,
            parentFrameOrientation=parent_frame_orn,
            childFrameOrientation=child_frame_orn,
        )
        # Modify max force based on user-determined assist parameters
        if joint_type == p.JOINT_FIXED:
            max_force = ASSIST_FORCE
        else:
            max_force = ASSIST_FORCE * ARTICULATED_ASSIST_FRACTION
        p.changeConstraint(self._ag_obj_cid[arm], maxForce=max_force)

        self._ag_obj_cid_params[arm] = {
            "childBodyUniqueId": ag_bid,
            "childLinkIndex": ag_link,
            "jointType": joint_type,
            "maxForce": max_force,
        }
        self._ag_obj_in_hand[arm] = ag_bid
        self._ag_freeze_gripper[arm] = True
        # Disable collisions while picking things up
        set_coll_filter(ag_bid, self.finger_links[arm], enable=False)
        for joint in self.finger_joints[arm]:
            j_val = joint.get_state()[0]
            self._ag_freeze_joint_pos[arm][joint.joint_name] = j_val

    def _handle_assisted_grasping(self, action):
        """
        Handles assisted grasping.

        :param action: Array[action], gripper action to apply. >= 0 is release (open), < 0 is grasp (close).
        """
        # Loop over all arms
        for arm in self.arm_names:
            # Make sure gripper action dimension is only 1
            assert (
                self._controllers["gripper_{}".format(arm)].command_dim == 1
            ), "Gripper {} controller command dim must be 1 to use assisted grasping, got: {}".format(
                arm, self._controllers["gripper_{}".format(arm)].command_dim
            )

            # TODO: Why are we separately checking for complementary conditions?
            threshold = np.mean(self._controllers["gripper_{}".format(arm)].command_input_limits)
            applying_grasp = action[self.controller_action_idx["gripper_{}".format(arm)]] < threshold
            releasing_grasp = action[self.controller_action_idx["gripper_{}".format(arm)]] > threshold

            # Execute gradual release of object
            if self._ag_obj_in_hand[arm]:
                if self._ag_release_counter[arm] is not None:
                    self._handle_release_window(arm=arm)
                else:
                    constraint_violated = (
                        get_constraint_violation(self._ag_obj_cid[arm]) > CONSTRAINT_VIOLATION_THRESHOLD
                    )
                    if constraint_violated or releasing_grasp:
                        self._release_grasp(arm=arm)

            elif applying_grasp:
                self._ag_data[arm] = self._calculate_in_hand_object(arm=arm)
                self._establish_grasp(arm=arm, ag_data=self._ag_data[arm])

    def dump_config(self):
        """Dump robot config"""
        dump = super(ManipulationRobot, self).dump_config()
        dump.update(
            {
                "grasping_mode": self.grasping_mode,
            }
        )
        return dump

    def dump_state(self):
        dump = super(ManipulationRobot, self).dump_state()

        if self.grasping_mode == "physical":
            return dump

        # Recompute child frame pose because it could have changed since the
        # constraint has been created
        ag_dump = {}
        for arm in self.arm_names:
            if self._ag_obj_cid[arm] is not None:
                ag_bid = self._ag_obj_cid_params[arm]["childBodyUniqueId"]
                ag_link = self._ag_obj_cid_params[arm]["childLinkIndex"]
                child_frame_pos, child_frame_orn = get_child_frame_pose(
                    parent_bid=self.eef_links[arm].body_id,
                    parent_link=self.eef_links[arm].link_id,
                    child_bid=ag_bid,
                    child_link=ag_link,
                )
                self._ag_obj_cid_params[arm].update(
                    {
                        "childFramePosition": child_frame_pos,
                        "childFrameOrientation": child_frame_orn,
                    }
                )
            ag_dump.update(
                {
                    "_ag_{}_obj_in_hand".format(arm): self._ag_obj_in_hand[arm],
                    "_ag_{}_release_counter".format(arm): self._ag_release_counter[arm],
                    "_ag_{}_freeze_gripper".format(arm): self._ag_freeze_gripper[arm],
                    "_ag_{}_freeze_joint_pos".format(arm): self._ag_freeze_joint_pos[arm],
                    "_ag_{}_obj_cid".format(arm): self._ag_obj_cid[arm],
                    "_ag_{}_obj_cid_params".format(arm): self._ag_obj_cid_params[arm],
                }
            )

        dump["ManipulationRobot"] = ag_dump

        return dump

    def load_state(self, dump):
        super(ManipulationRobot, self).load_state(dump)

        if self.grasping_mode == "physical":
            return

        # Loop over all arms
        for arm in self.arm_names:
            # Cancel the previous AG if exists
            if self._ag_obj_cid[arm] is not None:
                p.removeConstraint(self._ag_obj_cid[arm])

            if self._ag_obj_in_hand[arm] is not None:
                set_coll_filter(
                    self._ag_obj_in_hand[arm],
                    self.finger_links[arm],
                    enable=True,
                )

            robot_dump = dump["ManipulationRobot"]

            # For backwards compatibility, if the newest version of the string doesn't exist, we try to use the old string
            _ag_obj_in_hand_str = (
                "_ag_{}_obj_in_hand".format(arm) if "_ag_{}_obj_in_hand".format(arm) in robot_dump else "object_in_hand"
            )
            _ag_release_counter_str = (
                "_ag_{}_release_counter".format(arm)
                if "_ag_{}_release_counter".format(arm) in robot_dump
                else "release_counter"
            )
            _ag_freeze_gripper_str = (
                "_ag_{}_freeze_gripper".format(arm)
                if "_ag_{}_freeze_gripper".format(arm) in robot_dump
                else "should_freeze_joints"
            )
            _ag_freeze_joint_pos_str = (
                "_ag_{}_freeze_joint_pos".format(arm)
                if "_ag_{}_freeze_joint_pos".format(arm) in robot_dump
                else "freeze_vals"
            )
            _ag_obj_cid_str = "_ag_{}_obj_cid".format(arm) if "_ag_{}_obj_cid".format(arm) in robot_dump else "obj_cid"
            _ag_obj_cid_params_str = (
                "_ag_{}_obj_cid_params".format(arm)
                if "_ag_{}_obj_cid_params".format(arm) in robot_dump
                else "obj_cid_params"
            )

            self._ag_obj_in_hand[arm] = robot_dump[_ag_obj_in_hand_str]
            self._ag_release_counter[arm] = robot_dump[_ag_release_counter_str]
            self._ag_freeze_gripper[arm] = robot_dump[_ag_freeze_gripper_str]
            self._ag_freeze_joint_pos[arm] = {
                int(key): val for key, val in robot_dump[_ag_freeze_joint_pos_str].items()
            }
            self._ag_obj_cid[arm] = robot_dump[_ag_obj_cid_str]
            self._ag_obj_cid_params[arm] = robot_dump[_ag_obj_cid_params_str]
            if self._ag_obj_cid[arm] is not None:
                self._ag_obj_cid[arm] = p.createConstraint(
                    parentBodyUniqueId=self.eef_links[arm].body_id,
                    parentLinkIndex=self.eef_links[arm].link_id,
                    childBodyUniqueId=robot_dump[_ag_obj_cid_params_str]["childBodyUniqueId"],
                    childLinkIndex=robot_dump[_ag_obj_cid_params_str]["childLinkIndex"],
                    jointType=robot_dump[_ag_obj_cid_params_str]["jointType"],
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, 0),
                    childFramePosition=robot_dump[_ag_obj_cid_params_str]["childFramePosition"],
                    childFrameOrientation=robot_dump[_ag_obj_cid_params_str]["childFrameOrientation"],
                )
                p.changeConstraint(self._ag_obj_cid[arm], maxForce=robot_dump[_ag_obj_cid_params_str]["maxForce"])

            if self._ag_obj_in_hand[arm] is not None:
                set_coll_filter(
                    self._ag_obj_in_hand[arm],
                    self.finger_links[arm],
                    enable=False,
                )

    def can_toggle(self, toggle_position, toggle_distance_threshold):
        # Calculate for any fingers in any arm
        for arm in self.arm_names:
            for link in self.finger_links[arm]:
                link_pos = link.get_position()
                if np.linalg.norm(np.array(link_pos) - np.array(toggle_position)) < toggle_distance_threshold:
                    return True
        return False
