from abc import abstractmethod
from collections import namedtuple

import numpy as np

from igibson import app
from igibson.utils.assets_utils import get_assisted_grasping_categories
import igibson.utils.transform_utils as T
from igibson.controllers import (
    IsGraspingState,
    ControlType,
    JointController,
    ManipulationController,
    GripperController,
    MultiFingerGripperController,
    NullGripperController,
)
# from igibson.external.pybullet_tools.utils import (
#     ContactResult,
#     get_child_frame_pose,
#     get_constraint_violation,
#     get_link_pose,
# )
from igibson.objects.dataset_object import DatasetObject
from igibson.robots.robot_base import BaseRobot
from igibson.utils.python_utils import classproperty, assert_valid_key
from igibson.utils.constants import JointType
from igibson.utils.usd_utils import create_joint

from pxr import Gf


AG_MODES = {
    "physical",
    "assisted",
    "sticky",
}

# Assisted grasping parameters
VISUALIZE_RAYS = False
ASSIST_FRACTION = 1.0
ASSIST_GRASP_OBJ_CATEGORIES = get_assisted_grasping_categories()
ASSIST_GRASP_MASS_THRESHOLD = 10.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
RELEASE_WINDOW = 1 / 30.0  # release window in seconds
GraspingPoint = namedtuple("GraspingPoint", ["link_name", "position"])  # link_name (str), position (x,y,z tuple)


def can_assisted_grasp(obj):
    """
    Check whether an object @obj can be grasped. This is done
    by checking its category to see if is in the allowlist.

    :param obj: (BaseObject), Object targeted for an assisted grasp

    Returns:
        bool: Whether or not this object can be grasped
    """

    if isinstance(obj, DatasetObject) and obj.category != "object":
        # Use manually defined allowlist
        return obj.category in ASSIST_GRASP_OBJ_CATEGORIES
    else:
        # Use fallback based on mass
        mass = obj.mass
        print(f"Mass for AG: obj: {mass}, max mass: {ASSIST_GRASP_MASS_THRESHOLD}, obj: {obj.name}")
        return mass <= ASSIST_GRASP_MASS_THRESHOLD

# # TODO
# def set_coll_filter(target_body_id, source_links, enable):
#     # TODO: mostly shared with behavior robot, can be factored out
#     """
#     Sets collision filters for body - to enable or disable them
#     :param target_body_id: physics body to enable/disable collisions with
#     :param source_links: RobotLink objects to disable collisions with
#     :param enable: whether to enable/disable collisions
#     """
#     target_link_idxs = [-1] + list(range(p.getNumJoints(target_body_id)))
#
#     for link in source_links:
#         for target_link_idx in target_link_idxs:
#             p.setCollisionFilterPair(link.body_id, target_body_id, link.link_id, target_link_idx, 1 if enable else 0)


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
        prim_path,
        name=None,
        category="agent",
        class_id=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
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
        obs_modalities="all",
        proprio_obs="default",

        # Unique to ManipulationRobot
        grasping_mode="physical",

        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            robot abilities and parameters.
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific
            controller configurations for this object. This will override any default values specified by this class.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
            corresponds to all modalities being used.
            Otherwise, valid options should be part of igibson.sensors.ALL_SENSOR_MODALITIES.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store relevant internal vars
        assert_valid_key(key=grasping_mode, valid_keys=AG_MODES, name="grasping_mode")
        self._grasping_mode = grasping_mode

        # Initialize other variables used for assistive grasping
        self._ag_data = {arm: None for arm in self.arm_names}
        self._ag_freeze_joint_pos = {
            arm: {} for arm in self.arm_names
        }  # Frozen positions for keeping fingers held still
        self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
        self._ag_obj_constraints = {arm: None for arm in self.arm_names}
        self._ag_obj_constraint_params = {arm: {} for arm in self.arm_names}
        self._ag_freeze_gripper = {arm: None for arm in self.arm_names}
        self._ag_release_counter = {arm: None for arm in self.arm_names}

        # Call super() method
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            **kwargs,
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

            # We make sure that our gripper controller exists and is a gripper controller
            assert (
                "gripper_{}".format(arm) in self._controllers
            ), "Controller 'gripper_{}' must exist in controllers! Current controllers: {}".format(
                arm, list(self._controllers.keys())
            )
            assert isinstance(
                self._controllers["gripper_{}".format(arm)], GripperController
            ), "Gripper {} controller must be a GripperController!".format(arm)

        # run super
        super()._validate_configuration()

    def is_grasping(self, arm="default", candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        :param arm: str, specific arm to check for grasping. Default is "default" which corresponds to the first entry
        in self.arm_names
        :param candidate_obj: EntityPrim or None, object to check if this robot is currently grasping. If None, then
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
            # Infer from the gripper controller the state
            is_grasping = self._controllers["gripper_{}".format(arm)].is_grasping()
            # If candidate obj is not None, we also check to see if our fingers are in contact with the object
            if is_grasping and candidate_obj is not None:
                grasping_obj = False
                obj_links = {link.prim_path for link in candidate_obj.links.values()}
                finger_links = {link.prim_path for link in self.finger_links[arm]}
                for c in self.contact_list():
                    c_set = {c.body0, c.body1}
                    # Valid grasping of object if one of the set is a finger link and the other is the grasped object
                    if len(c_set - finger_links) == 1 and len(c_set - obj_links) == 1:
                        grasping_obj = True
                        break
                # Update is_grasping
                is_grasping = grasping_obj

        return is_grasping

    # TODO
    def _find_gripper_raycast_collisions(self, arm="default"):
        """
        For arm @arm, calculate any prims that are not part of the robot
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

        :return set[tuple[str[, Array]]], dict[str: set{str}]: first return value is set of unique
            contact prim_paths that are not the robot self-collisions.
            If @return_contact_positions is True,
            then returns (prim_path, pos), where pos is the contact (x,y,z) position
            Note: if no objects that are not the robot itself are intersecting, the set will be empty.

            Second return value is dictionary mapping unique contact objects defined by the contact prim_path to
            set of unique robot link prim_paths that it is in contact with
        """
        arm = self.default_arm if arm == "default" else arm
        robot_contact_links = dict()
        contact_data = set()
        # Find all objects in contact with all finger joints for this arm
        con_results = [con for link in self.finger_links[arm] for con in link.contact_list()]

        # Get robot contact links
        link_paths = set(self.link_prim_paths)

        for con_res in con_results:
            # Only add this contact if it's not a robot self-collision
            other_contact_set = {con_res.body0, con_res.body1} - link_paths
            if len(other_contact_set) == 1:
                link_contact, other_contact = (con_res.body0, con_res.body1) if \
                    list(other_contact_set)[0] == con_res.body1 else (con_res.body1, con_res.body0)
                # Add to contact data
                contact_data.add((other_contact, tuple(con_res.position)) if return_contact_positions else other_contact)
                # Also add robot contact link info
                if other_contact not in robot_contact_links:
                    robot_contact_links[other_contact] = set()
                robot_contact_links[other_contact].add(link_contact)

        return contact_data, robot_contact_links

    def set_position_orientation(self, position=None, orientation=None):
        # Store the original EEF poses.
        original_poses = {}
        for arm in self.arm_names:
            original_poses[arm] = (self.get_eef_position(arm), self.get_eef_orientation(arm))

        # Run the super method
        super().set_position_orientation(position=position, orientation=orientation)

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

    def deploy_control(self, control, control_type, indices=None, normalized=False):
        # We intercept the gripper control and replace it with the current joint position if we're freezing our gripper
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                control[self.gripper_control_idx[arm]] = self._ag_obj_constraint_params[arm]["gripper_pos"] if \
                    self.controllers[f"gripper_{arm}"].control_type == ControlType.POSITION else 0.0

        super().deploy_control(control=control, control_type=control_type, indices=indices, normalized=normalized)

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        :param arm: str, specific arm whose grasp will be released.
            Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm

        # Remove joint and filtered collision restraints
        self._simulator.stage.RemovePrim(self._ag_obj_constraint_params[arm]["ag_joint_prim_path"])
        self._ag_data[arm] = None
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = {}
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
        joints_state = self.get_joints_state(normalized=False)
        for arm in self.arm_names:
            # Add arm info
            dic["arm_{}_qpos".format(arm)] = joints_state.positions[self.arm_control_idx[arm]]
            dic["arm_{}_qpos_sin".format(arm)] = np.sin(joints_state.positions[self.arm_control_idx[arm]])
            dic["arm_{}_qpos_cos".format(arm)] = np.cos(joints_state.positions[self.arm_control_idx[arm]])
            dic["arm_{}_qvel".format(arm)] = joints_state.velocities[self.arm_control_idx[arm]]

            # Add eef and grasping info
            dic["eef_{}_pos_global".format(arm)] = self.get_eef_position(arm)
            dic["eef_{}_quat_global".format(arm)] = self.get_eef_orientation(arm)
            dic["eef_{}_pos".format(arm)] = self.get_relative_eef_position(arm)
            dic["eef_{}_quat".format(arm)] = self.get_relative_eef_orientation(arm)
            dic["grasp_{}".format(arm)] = np.array([self.is_grasping(arm)])
            dic["gripper_{}_qpos".format(arm)] = joints_state.positions[self.gripper_control_idx[arm]]
            dic["gripper_{}_qvel".format(arm)] = joints_state.velocities[self.gripper_control_idx[arm]]

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
    def arm_link_names(self):
        """
        :return dict[str, list[str]]: Dictionary mapping arm appendage name to corresponding arm link names,
            should correspond to specific link names in this robot's underlying model file
        """
        raise NotImplementedError

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
    def arm_links(self):
        """
        :return dict[str, Array[RobotLink]]: Dictionary mapping arm appendage name to robot links corresponding to
            that arm's links
        """
        return {arm: [self._links[link] for link in self.arm_link_names[arm]] for arm in self.arm_names}

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

    # @property
    # # TODO - deprecate
    # def gripper_link_to_grasp_point(self):
    #     """
    #     :return dict[str, Array[float]]: Dictionary mapping arm appendage name to (dx,dy,dz) relative distance from
    #         the gripper link frame to the expected center of the robot's grasping point.
    #         Unique to each robot embodiment.
    #     """
    #     raise NotImplementedError

    @property
    def finger_lengths(self):
        """
        :return dict[str, float]: Dictionary mapping arm appendage name to corresponding length of the fingers in that
            hand defined from the palm (assuming all fingers in one hand are equally long)
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
        eef_link_pose = self.eef_links[arm].get_position_orientation()
        base_link_pose = self.get_position_orientation()
        pose = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
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

        :return None or tuple(BaseObject, RigidPrim): If a valid assisted-grasp object is found, returns the corresponding
            (object, object_link) pair to the contacted in-hand object. Otherwise, returns None
        """
        arm = self.default_arm if arm == "default" else arm

        # If we're not using physical grasping, we check for gripper contact
        if self.grasping_mode != "physical":
            candidates_set, robot_contact_links = self._find_gripper_contacts(arm=arm)
            # If we're using assisted grasping, we further filter candidates via ray-casting
            if self.grasping_mode == "assisted":
                raise NotImplementedError("Not assisted grasp avaialble yet in OmniGibson!")
                candidates_set_raycast = self._find_gripper_raycast_collisions(arm=arm)
                candidates_set = candidates_set.intersection(candidates_set_raycast)
        else:
            raise ValueError("Invalid grasping mode for calculating in hand object: {}".format(self.grasping_mode))

        # Immediately return if there are no valid candidates
        if len(candidates_set) == 0:
            return None

        # Find the closest object to the gripper center
        gripper_center_pos = self.eef_links[arm].get_position()

        candidate_data = []
        for prim_path in candidates_set:
            # Calculate position of the object link
            # Note: this assumes the simulator is playing!
            rb_handle = self._dc.get_rigid_body(prim_path)
            pose = self._dc.get_rigid_body_pose(rb_handle)
            link_pos = np.asarray(pose.p)
            dist = np.linalg.norm(np.array(link_pos) - np.array(gripper_center_pos))
            candidate_data.append((prim_path, dist))

        candidate_data = sorted(candidate_data, key=lambda x: x[-1])

        ag_prim_path, _ = candidate_data[0]

        # Make sure the ag_prim_path is not a self collision
        assert ag_prim_path not in self.link_prim_paths, "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[ag_prim_path]
        touching_at_least_two_fingers = len({link.prim_path for link in self.finger_links[arm]}.intersection(robot_contacts)) >= 2

        # TODO: Better heuristic, hacky, we assume the parent object prim path is the prim_path minus the last "/" item
        ag_obj_prim_path = "/".join(prim_path.split("/")[:-1])
        ag_obj_link_name = prim_path.split("/")[-1]
        ag_obj = self._simulator.scene.object_registry("prim_path", ag_obj_prim_path)
        ag_obj_link = ag_obj.links[ag_obj_link_name]

        # Return None if object cannot be assisted grasped or not touching at least two fingers
        if ag_obj is None or (not can_assisted_grasp(ag_obj)) or (not touching_at_least_two_fingers):
            return None

        # Get object and its contacted link
        return ag_obj, ag_obj_link

    def _handle_release_window(self, arm="default"):
        """
        Handles releasing an object from arm @arm

        :param arm: str, specific arm to handle release window.
        Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        self._ag_release_counter[arm] += 1
        time_since_release = self._ag_release_counter[arm] * self._simulator.render_timestep
        if time_since_release >= RELEASE_WINDOW:
            # Remove filtered collision restraints
            for finger_link in self.finger_links[arm]:
                finger_link.remove_filtered_collision_pair(prim=self._ag_obj_in_hand[arm])
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
            joint.set_pos(pos=j_val, normalized=False, target=False)
            joint.set_vel(vel=0.0, normalized=False, target=False)

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
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": "default",
                "use_delta_commands": False,
            }
        return dic

    # TODO - update
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
                "base_body_id": None, #self.base_link.body_id,
                "task_link_id": None, #self.eef_links[arm].link_id,
                "task_name": "eef_{}".format(arm),
                "control_freq": self._control_freq,
                "default_joint_pos": self.default_joint_pos,
                "joint_damping": None, #self.joint_damping,
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
                "control_freq": self._control_freq,
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
                "control_freq": self._control_freq,
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
                "control_freq": self._control_freq,
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
        :param ag_data: None or (BaseObject, RigidPrim), if specified, assisted-grasp object, link tuple.
            Otherwise, does a no-op
        """
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return
        ag_obj, ag_link = ag_data

        # Create a p2p joint if it's a child link of a fixed URDF that is connected by a revolute or prismatic joint
        joint_type = "FixedJoint"
        if ag_obj.fixed_base:
            # We search up the tree path from the ag_link until we encounter the root (joint == 0) or a non fixed
            # joint (e.g.: revolute or fixed)
            joint_handle = -1
            link_handle = ag_link.handle
            use_spherical = False
            while joint_handle != 0:
                # If this joint type is not fixed, we've encountered a valid moving joint
                # So we create a spherical joint rather than fixed joint
                if self._dc.get_joint_type(joint_handle) != JointType.JOINT_FIXED:
                    joint_type = "SphericalJoint"
                    break
                # Grab the parent link and its parent joint for the link
                link_handle = self._dc.get_joint_parent_body(joint_handle)
                joint_handle = self._dc.get_rigid_body_parent_joint(link_handle)

        force_data, _ = self._find_gripper_contacts(arm=arm, return_contact_positions=True)
        contact_pos = None
        for c_link_prim_path, c_contact_pos in force_data:
            if c_link_prim_path == ag_link.prim_path:
                contact_pos = np.array(c_contact_pos)
                break
        assert contact_pos is not None

        # Joint frame set at the contact point
        # Need to find distance between robot and contact point in robot link's local frame and
        # ag link and contact point in ag link's local frame
        joint_frame_pos = contact_pos
        joint_frame_orn = np.array([0, 0, 0, 1.0])
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        parent_frame_pos, parent_frame_orn = T.relative_pose_transform(joint_frame_pos, joint_frame_orn, eef_link_pos, eef_link_orn)
        obj_link_pos, obj_link_orn = ag_link.get_position_orientation()
        child_frame_pos, child_frame_orn = T.relative_pose_transform(joint_frame_pos, joint_frame_orn, obj_link_pos, obj_link_orn)
        # inv_eef_link_pos, inv_eef_link_orn = p.invertTransform(eef_link_pos, eef_link_orn)
        # parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
        #     inv_eef_link_pos, inv_eef_link_orn, joint_frame_pos, joint_frame_orn
        # )
        # if ag_link == -1:
        #     obj_pos, obj_orn = p.getBasePositionAndOrientation(ag_bid)
        # else:
        #     obj_pos, obj_orn = p.getLinkState(ag_bid, ag_link)[:2]
        # inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
        # child_frame_pos, child_frame_orn = p.multiplyTransforms(
        #     inv_obj_pos, inv_obj_orn, joint_frame_pos, joint_frame_orn
        # )

        # Create the joint
        # self._simulator.pause()
        joint_prim_path = f"{self.eef_links[arm].prim_path}/ag_constraint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=self.eef_links[arm].prim_path,
            body1=ag_link.prim_path,
            enabled=False,
            stage=self._simulator.stage,
        )

        # Set the local pose of this joint
        joint_prim.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*(parent_frame_pos / self.scale)))
        joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*(parent_frame_orn[[3, 0, 1, 2]])))
        joint_prim.GetAttribute("physics:localPos1").Set(Gf.Vec3f(*(child_frame_pos / ag_obj.scale)))
        joint_prim.GetAttribute("physics:localRot1").Set(Gf.Quatf(*(child_frame_orn[[3, 0, 1, 2]])))

        # We have to toggle the joint from off to on after a physics step because of an omni quirk
        # Otherwise the joint transform is very weird
        app.update()
        joint_prim.GetAttribute("physics:jointEnabled").Set(True)
        # self._simulator.play()

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        max_force = ASSIST_FORCE if joint_type == "FixedJoint" else ASSIST_FORCE * ARTICULATED_ASSIST_FRACTION
        # joint_prim.GetAttribute("physics:breakForce").Set(max_force)

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True
        # Disable collisions while picking things up
        # for finger_link in self.finger_links[arm]:
        #     finger_link.add_filtered_collision_pair(prim=ag_obj)
        for joint in self.finger_joints[arm]:
            j_val = joint.get_state()[0][0]
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
                    # constraint_violated = (
                    #     get_constraint_violation(self._ag_obj_cid[arm]) > CONSTRAINT_VIOLATION_THRESHOLD
                    # )
                    # if constraint_violated or releasing_grasp:
                    if releasing_grasp:
                        self._release_grasp(arm=arm)

            elif applying_grasp:
                self._ag_data[arm] = self._calculate_in_hand_object(arm=arm)
                self._establish_grasp(arm=arm, ag_data=self._ag_data[arm])

    def dump_config(self):
        # Grab running config
        cfg = super().dump_config()

        # Add relevant params
        cfg["grasping_mode"] = self.grasping_mode

        return cfg

    def _dump_state(self):
        # Call super first
        state = super()._dump_state()

        # If we're using actual physical grasping, no extra state needed to save
        if self.grasping_mode == "physical":
            return state

        # Recompute child frame pose because it could have changed since the
        # constraint has been created
        ag_dump = {}
        # TODO
        # for arm in self.arm_names:
        #     if self._ag_obj_cid[arm] is not None:
        #         ag_bid = self._ag_obj_cid_params[arm]["childBodyUniqueId"]
        #         ag_link = self._ag_obj_cid_params[arm]["childLinkIndex"]
        #         child_frame_pos, child_frame_orn = get_child_frame_pose(
        #             parent_bid=self.eef_links[arm].body_id,
        #             parent_link=self.eef_links[arm].link_id,
        #             child_bid=ag_bid,
        #             child_link=ag_link,
        #         )
        #         self._ag_obj_cid_params[arm].update(
        #             {
        #                 "childFramePosition": child_frame_pos,
        #                 "childFrameOrientation": child_frame_orn,
        #             }
        #         )
        #     ag_dump.update(
        #         {
        #             "_ag_{}_obj_in_hand".format(arm): self._ag_obj_in_hand[arm],
        #             "_ag_{}_release_counter".format(arm): self._ag_release_counter[arm],
        #             "_ag_{}_freeze_gripper".format(arm): self._ag_freeze_gripper[arm],
        #             "_ag_{}_freeze_joint_pos".format(arm): self._ag_freeze_joint_pos[arm],
        #             "_ag_{}_obj_cid".format(arm): self._ag_obj_cid[arm],
        #             "_ag_{}_obj_cid_params".format(arm): self._ag_obj_cid_params[arm],
        #         }
        #     )
        #
        # state["ag"] = ag_dump

        return state

    def _load_state(self, state):
        super()._load_state(state=state)

        # No additional loading needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return

        # # Loop over all arms
        # # TODO
        # for arm in self.arm_names:
        #     # Cancel the previous AG if exists
        #     if self._ag_obj_cid[arm] is not None:
        #         p.removeConstraint(self._ag_obj_cid[arm])
        #
        #     if self._ag_obj_in_hand[arm] is not None:
        #         set_coll_filter(
        #             self._ag_obj_in_hand[arm],
        #             self.finger_links[arm],
        #             enable=True,
        #         )
        #
        #     robot_dump = state["ag"]
        #
        #     # For backwards compatibility, if the newest version of the string doesn't exist, we try to use the old string
        #     _ag_obj_in_hand_str = (
        #         "_ag_{}_obj_in_hand".format(arm) if "_ag_{}_obj_in_hand".format(arm) in robot_dump else "object_in_hand"
        #     )
        #     _ag_release_counter_str = (
        #         "_ag_{}_release_counter".format(arm)
        #         if "_ag_{}_release_counter".format(arm) in robot_dump
        #         else "release_counter"
        #     )
        #     _ag_freeze_gripper_str = (
        #         "_ag_{}_freeze_gripper".format(arm)
        #         if "_ag_{}_freeze_gripper".format(arm) in robot_dump
        #         else "should_freeze_joints"
        #     )
        #     _ag_freeze_joint_pos_str = (
        #         "_ag_{}_freeze_joint_pos".format(arm)
        #         if "_ag_{}_freeze_joint_pos".format(arm) in robot_dump
        #         else "freeze_vals"
        #     )
        #     _ag_obj_cid_str = "_ag_{}_obj_cid".format(arm) if "_ag_{}_obj_cid".format(arm) in robot_dump else "obj_cid"
        #     _ag_obj_cid_params_str = (
        #         "_ag_{}_obj_cid_params".format(arm)
        #         if "_ag_{}_obj_cid_params".format(arm) in robot_dump
        #         else "obj_cid_params"
        #     )
        #
        #     self._ag_obj_in_hand[arm] = robot_dump[_ag_obj_in_hand_str]
        #     self._ag_release_counter[arm] = robot_dump[_ag_release_counter_str]
        #     self._ag_freeze_gripper[arm] = robot_dump[_ag_freeze_gripper_str]
        #     self._ag_freeze_joint_pos[arm] = {
        #         int(key): val for key, val in robot_dump[_ag_freeze_joint_pos_str].items()
        #     }
        #     self._ag_obj_cid[arm] = robot_dump[_ag_obj_cid_str]
        #     self._ag_obj_cid_params[arm] = robot_dump[_ag_obj_cid_params_str]
        #     if self._ag_obj_cid[arm] is not None:
        #         self._ag_obj_cid[arm] = p.createConstraint(
        #             parentBodyUniqueId=self.eef_links[arm].body_id,
        #             parentLinkIndex=self.eef_links[arm].link_id,
        #             childBodyUniqueId=robot_dump[_ag_obj_cid_params_str]["childBodyUniqueId"],
        #             childLinkIndex=robot_dump[_ag_obj_cid_params_str]["childLinkIndex"],
        #             jointType=robot_dump[_ag_obj_cid_params_str]["jointType"],
        #             jointAxis=(0, 0, 0),
        #             parentFramePosition=(0, 0, 0),
        #             childFramePosition=robot_dump[_ag_obj_cid_params_str]["childFramePosition"],
        #             childFrameOrientation=robot_dump[_ag_obj_cid_params_str]["childFrameOrientation"],
        #         )
        #         p.changeConstraint(self._ag_obj_cid[arm], maxForce=robot_dump[_ag_obj_cid_params_str]["maxForce"])
        #
        #     if self._ag_obj_in_hand[arm] is not None:
        #         set_coll_filter(
        #             self._ag_obj_in_hand[arm],
        #             self.finger_links[arm],
        #             enable=False,
        #         )

    def _serialize(self, state):
        # Call super first
        state_flat = super()._serialize(state=state)

        # No additional serialization needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return state_flat

        # TODO AG
        return state_flat

    def _deserialize(self, state):
        # Call super first
        state_dict, idx = super()._deserialize(state=state)

        # No additional deserialization needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return state_dict, idx

        # TODO AG
        return state_dict, idx

    def can_toggle(self, toggle_position, toggle_distance_threshold):
        # Calculate for any fingers in any arm
        for arm in self.arm_names:
            for link in self.finger_links[arm]:
                link_pos = link.get_position()
                if np.linalg.norm(np.array(link_pos) - np.array(toggle_position)) < toggle_distance_threshold:
                    return True
        return False

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ManipulationRobot")
        return classes
