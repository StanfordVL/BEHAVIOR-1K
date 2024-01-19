from abc import abstractmethod
from collections import namedtuple
import numpy as np
import networkx as nx

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.controllers import InverseKinematicsController, MultiFingerGripperController, OperationalSpaceController
from omnigibson.macros import gm, create_module_macros
from omnigibson.object_states import ContactBodies
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import (
    IsGraspingState,
    ControlType,
    ManipulationController,
    GripperController,
)
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.python_utils import classproperty, assert_valid_key
from omnigibson.utils.geometry_utils import generate_points_in_volume_checker_function
from omnigibson.utils.constants import JointType, PrimType
from omnigibson.utils.usd_utils import create_joint
from omnigibson.utils.teleop_utils import TeleopData
from omnigibson.utils.sampling_utils import raytest_batch

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Assisted grasping parameters
m.ASSIST_FRACTION = 1.0
m.ASSIST_GRASP_MASS_THRESHOLD = 10.0
m.ARTICULATED_ASSIST_FRACTION = 0.7
m.MIN_ASSIST_FORCE = 0
m.MAX_ASSIST_FORCE = 100
m.ASSIST_FORCE = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
m.CONSTRAINT_VIOLATION_THRESHOLD = 0.1
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
        prim_path=None,
        class_id=None,
        uuid=None,
        scale=None,
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
        sensor_config=None,

        # Unique to ManipulationRobot
        grasping_mode="physical",
        grasping_direction="lower",
        disable_grasp_handling=False,

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be at the render frequency by default.
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store relevant internal vars
        assert_valid_key(key=grasping_mode, valid_keys=AG_MODES, name="grasping_mode")
        assert_valid_key(key=grasping_direction, valid_keys=["lower", "upper"], name="grasping direction")
        self._grasping_mode = grasping_mode
        self._grasping_direction = grasping_direction
        self._disable_grasp_handling = disable_grasp_handling

        # Initialize other variables used for assistive grasping
        self._ag_freeze_joint_pos = {
            arm: {} for arm in self.arm_names
        }  # Frozen positions for keeping fingers held still
        self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
        self._ag_obj_constraints = {arm: None for arm in self.arm_names}
        self._ag_obj_constraint_params = {arm: {} for arm in self.arm_names}
        self._ag_freeze_gripper = {arm: None for arm in self.arm_names}
        self._ag_release_counter = {arm: None for arm in self.arm_names}
        self._ag_check_in_volume = {arm: None for arm in self.arm_names}
        self._ag_calculate_volume = {arm: None for arm in self.arm_names}

        # Call super() method
        super().__init__(
            prim_path=prim_path,
            name=name,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
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
            sensor_config=sensor_config,
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

    def _initialize(self):
        super()._initialize()

        if gm.AG_CLOTH:
            for arm in self.arm_names:
                self._ag_check_in_volume[arm], self._ag_calculate_volume[arm] = \
                    generate_points_in_volume_checker_function(obj=self, volume_link=self.eef_links[arm], mesh_name_prefixes="container")

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
                is_grasping = len(candidate_obj.states[ContactBodies].get_value().intersection(finger_links)) > 0

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

    def deploy_control(self, control, control_type, indices=None, normalized=False):
        # We intercept the gripper control and replace it with the current joint position if we're freezing our gripper
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                control[self.gripper_control_idx[arm]] = self._ag_obj_constraint_params[arm]["gripper_pos"] if \
                    self.controllers[f"gripper_{arm}"].control_type == ControlType.POSITION else 0.0

        super().deploy_control(control=control, control_type=control_type, indices=indices, normalized=normalized)

        # Then run assisted grasping
        if self.grasping_mode != "physical" and not self._disable_grasp_handling:
            self._handle_assisted_grasping()

        # Potentially freeze gripper joints
        for arm in self.arm_names:
            if self._ag_freeze_gripper[arm]:
                self._freeze_gripper(arm)

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        Args:
            arm (str): specific arm whose grasp will be released.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm

        # Remove joint and filtered collision restraints
        og.sim.stage.RemovePrim(self._ag_obj_constraint_params[arm]["ag_joint_prim_path"])
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = {}
        self._ag_freeze_gripper[arm] = False
        self._ag_release_counter[arm] = 0

    def release_grasp_immediately(self):
        """
        Magic action to release this robot's grasp for all arms at once.
        As opposed to @_release_grasp, this method would byupass the release window mechanism and immediately release.
        """
        for arm in self.arm_names:
            if self._ag_obj_in_hand[arm] is not None:
                self._release_grasp(arm=arm)
                self._ag_release_counter[arm] = int(np.ceil(m.RELEASE_WINDOW / og.sim.get_rendering_dt()))
                self._handle_release_window(arm=arm)
                assert not self._ag_obj_in_hand[arm], "Object still in ag list after release!"
                # TODO: Verify not needed!
                # for finger_link in self.finger_links[arm]:
                #     finger_link.remove_filtered_collision_pair(prim=self._ag_obj_in_hand[arm])

    def get_control_dict(self):
        # In addition to super method, add in EEF states
        dic = super().get_control_dict()

        for arm in self.arm_names:
            rel_eef_pos, rel_eef_quat = self.get_relative_eef_pose(arm)
            dic[f"eef_{arm}_pos_relative"] = rel_eef_pos
            dic[f"eef_{arm}_quat_relative"] = rel_eef_quat
            dic[f"eef_{arm}_lin_vel_relative"] = self.get_relative_eef_lin_vel(arm)
            dic[f"eef_{arm}_ang_vel_relative"] = self.get_relative_eef_ang_vel(arm)
            # -n_joints because there may be an additional 6 entries at the beginning of the array, if this robot does
            # not have a fixed base (i.e.: the 6DOF --> "floating" joint)
            # see self.get_relative_jacobian() for more info
            eef_link_idx = self._articulation_view.get_body_index(self.eef_links[arm].body_name)
            dic[f"eef_{arm}_jacobian_relative"] = self.get_relative_jacobian()[eef_link_idx, :, -self.n_joints:]

        return dic

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Loop over all arms to grab proprio info
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        for arm in self.arm_names:
            # Add arm info
            dic["arm_{}_qpos".format(arm)] = joint_positions[self.arm_control_idx[arm]]
            dic["arm_{}_qpos_sin".format(arm)] = np.sin(joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qpos_cos".format(arm)] = np.cos(joint_positions[self.arm_control_idx[arm]])
            dic["arm_{}_qvel".format(arm)] = joint_velocities[self.arm_control_idx[arm]]

            # Add eef and grasping info
            dic["eef_{}_pos_global".format(arm)] = self.get_eef_position(arm)
            dic["eef_{}_quat_global".format(arm)] = self.get_eef_orientation(arm)
            dic["eef_{}_pos".format(arm)] = self.get_relative_eef_position(arm)
            dic["eef_{}_quat".format(arm)] = self.get_relative_eef_orientation(arm)
            dic["grasp_{}".format(arm)] = np.array([self.is_grasping(arm)])
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
        Returns:
            int: Number of arms this robot has. Returns 1 by default
        """
        return 1

    @property
    def arm_names(self):
        """
        Returns:
            list of str: List of arm names for this robot. Should correspond to the keys used to index into
                arm- and gripper-related dictionaries, e.g.: eef_link_names, finger_link_names, etc.
                Default is string enumeration based on @self.n_arms.
        """
        return [str(i) for i in range(self.n_arms)]

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
            action_start_idx = sum([self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)])
            arm_action_idx[arm_name] = np.arange(action_start_idx, action_start_idx + self.controllers[f"arm_{arm_name}"].command_dim)
        return arm_action_idx

    @property
    def gripper_action_idx(self):
        gripper_action_idx = {}
        for arm_name in self.arm_names:
            controller_idx = self.controller_order.index(f"gripper_{arm_name}")
            action_start_idx = sum([self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)])
            gripper_action_idx[arm_name] = np.arange(action_start_idx, action_start_idx + self.controllers[f"gripper_{arm_name}"].command_dim)
        return gripper_action_idx

    @property
    @abstractmethod
    def arm_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm link names,
                should correspond to specific link names in this robot's underlying model file
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def arm_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm joint names,
                should correspond to specific joint names in this robot's underlying model file
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def eef_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding name of the EEF link,
                should correspond to specific link name in this robot's underlying model file
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of link names corresponding to
                this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def finger_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of joint names corresponding to
                this robot's fingers
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def arm_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to arm joints.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gripper_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to gripper joints.
        """
        raise NotImplementedError

    @property
    def arm_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's links
        """
        return {arm: [self._links[link] for link in self.arm_link_names[arm]] for arm in self.arm_names}

    @property
    def eef_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot link corresponding to that arm's
                eef link
        """
        return {arm: self._links[self.eef_link_names[arm]] for arm in self.arm_names}

    @property
    def finger_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's finger links
        """
        return {arm: [self._links[link] for link in self.finger_link_names[arm]] for arm in self.arm_names}

    @property
    def finger_joints(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot joints corresponding to
                that arm's finger joints
        """
        return {arm: [self._joints[joint] for joint in self.finger_joint_names[arm]] for arm in self.arm_names}

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
        return {arm: None for arm in self.arm_names}

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
        return {arm: None for arm in self.arm_names}

    @property
    def finger_lengths(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding length of the fingers in that
                hand defined from the palm (assuming all fingers in one hand are equally long)
        """
        raise NotImplementedError

    @property
    def arm_workspace_range(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to a tuple indicating the start and end of the
                angular range of the arm workspace around the Z axis of the robot, where 0 is facing
                forward.
        """
        raise NotImplementedError

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
        return self._links[self.eef_link_names[arm]].get_position()

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
        return self._links[self.eef_link_names[arm]].get_orientation()

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
        base_link_quat = self.get_orientation()
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
        base_link_quat = self.get_orientation()
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
                (object, object_link) (i.e.: (BaseObject, RigidPrim)) pair to the contacted in-hand object.
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
        gripper_center_pos = self.eef_links[arm].get_position()

        candidate_data = []
        for prim_path in candidates_set:
            # Calculate position of the object link. Only allow this for objects currently.
            obj_prim_path, link_name = prim_path.rsplit("/", 1)
            candidate_obj = og.sim.scene.object_registry("prim_path", obj_prim_path, None)
            if candidate_obj is None or link_name not in candidate_obj.links:
                continue
            candidate_link = candidate_obj.links[link_name]
            dist = np.linalg.norm(np.array(candidate_link.get_position()) - np.array(gripper_center_pos))
            candidate_data.append((prim_path, dist))

        if not candidate_data:
            return None

        candidate_data = sorted(candidate_data, key=lambda x: x[-1])
        ag_prim_path, _ = candidate_data[0]

        # Make sure the ag_prim_path is not a self collision
        assert ag_prim_path not in self.link_prim_paths, "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[ag_prim_path]
        touching_at_least_two_fingers = True if self.grasping_mode == "sticky" else len({link.prim_path for link in self.finger_links[arm]}.intersection(robot_contacts)) >= 2

        # TODO: Better heuristic, hacky, we assume the parent object prim path is the prim_path minus the last "/" item
        ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
        ag_obj_link_name = ag_prim_path.split("/")[-1]
        ag_obj = og.sim.scene.object_registry("prim_path", ag_obj_prim_path)

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
        startpoints = []
        endpoints = []
        for grasp_start_point in self.assisted_grasp_start_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_start_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to startpoints
            start_point, _ = T.pose_transform(link_pos, link_orn, grasp_start_point.position, [0, 0, 0, 1])
            startpoints.append(start_point)
        # Repeat for end points
        for grasp_end_point in self.assisted_grasp_end_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_end_point.link_name].get_position_orientation()
            # Calculate grasp start point in world frame and add to endpoints
            end_point, _ = T.pose_transform(link_pos, link_orn, grasp_end_point.position, [0, 0, 0, 1])
            endpoints.append(end_point)
        # Stack the start points and repeat the end points, and add these values to the raycast dicts
        n_startpoints, n_endpoints = len(startpoints), len(endpoints)
        raycast_startpoints = startpoints * n_endpoints
        raycast_endpoints = []
        for endpoint in endpoints:
            raycast_endpoints += [endpoint] * n_startpoints
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
        time_since_release = self._ag_release_counter[arm] * og.sim.get_rendering_dt()
        if time_since_release >= m.RELEASE_WINDOW:
            self._ag_obj_in_hand[arm] = None
            self._ag_release_counter[arm] = None

    def _freeze_gripper(self, arm="default"):
        """
        Freezes gripper finger joints - used in assisted grasping.

        Args:
            arm (str): specific arm to freeze gripper.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        arm = self.default_arm if arm == "default" else arm
        for joint_name, j_val in self._ag_freeze_joint_pos[arm].items():
            joint = self._joints[joint_name]
            joint.set_pos(pos=j_val)
            joint.set_vel(vel=0.0)

    @property
    def robot_arm_descriptor_yamls(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to files path to the descriptor
                of the robot for IK Controller.
        """
        raise NotImplementedError

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
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                Inverse kinematics controller to control this robot's arm
        """
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "InverseKinematicsController",
                "task_name": f"eef_{arm}",
                "robot_description_path": self.robot_arm_descriptor_yamls[arm],
                "robot_urdf_path": self.urdf_path,
                "eef_name": self.eef_link_names[arm],
                "control_freq": self._control_freq,
                "default_joint_pos": self.reset_joint_pos,
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
                "default_joint_pos": self.reset_joint_pos,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    np.array([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
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
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
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
            ag_link (RigidPrim): Link of the object to be grasped

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
            joint = ag_obj.articulation_tree.edges[edge]["joint"]
            if joint.joint_type != JointType.JOINT_FIXED:
                joint_type = "SphericalJoint"
                break

        return joint_type

    def _establish_grasp_rigid(self, arm="default", ag_data=None, contact_pos=None):
        """
        Establishes an ag-assisted grasp, if enabled.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 2-tuple): if specified, assisted-grasp object, link tuple (i.e. :(BaseObject, RigidPrim)).
                Otherwise, does a no-op
            contact_pos (None or np.array): if specified, contact position to use for grasp.
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

        # Create the joint
        joint_prim_path = f"{self.eef_links[arm].prim_path}/ag_constraint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=self.eef_links[arm].prim_path,
            body1=ag_link.prim_path,
            enabled=True,
            joint_frame_in_parent_frame_pos=parent_frame_pos / self.scale,
            joint_frame_in_parent_frame_quat=parent_frame_orn,
            joint_frame_in_child_frame_pos=child_frame_pos / ag_obj.scale,
            joint_frame_in_child_frame_quat=child_frame_orn,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        max_force = m.ASSIST_FORCE if joint_type == "FixedJoint" else m.ASSIST_FORCE * m.ARTICULATED_ASSIST_FRACTION
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
        for joint in self.finger_joints[arm]:
            j_val = joint.get_state()[0][0]
            self._ag_freeze_joint_pos[arm][joint.joint_name] = j_val

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
            threshold = np.mean(np.array(self.control_limits["position"])[:, controlled_joints], axis=0)
            if controller.control is None:
                applying_grasp = False
            elif self._grasping_direction == "lower":
                applying_grasp = np.any(controller.control < threshold)
            else:
                applying_grasp = np.any(controller.control > threshold)
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
                self._establish_grasp(arm=arm, ag_data=self._calculate_in_hand_object(arm=arm))

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
        attachment_point_pos, _ = T.pose_transform(eef_link_pos, eef_link_orn, attachment_point_pos_local, [0, 0, 0, 1])
        joint_prim = self._ag_obj_constraints[arm]
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*attachment_point_pos.astype(float)))

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

        Calculates which object to assisted-grasp for arm @arm. Returns an (BaseObject, RigidPrim, np.ndarray) tuple or
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
                ((BaseObject, RigidPrim, np.ndarray)) to the contacted in-hand object. Otherwise, returns None
        """
        # TODO (eric): Assume joint_pos = 0 means fully closed
        GRIPPER_FINGER_CLOSE_THRESHOLD = 0.03
        gripper_finger_pos = self.get_joint_positions()[self.gripper_control_idx[arm]]
        gripper_finger_close = np.sum(gripper_finger_pos) < GRIPPER_FINGER_CLOSE_THRESHOLD
        if not gripper_finger_close:
            return None

        cloth_objs = og.sim.scene.object_registry("prim_type", PrimType.CLOTH)
        if cloth_objs is None:
            return None

        # TODO (eric): Only AG one cloth at any given moment.
        # Returns the first cloth that overlaps with the "ghost" box volume
        for cloth_obj in cloth_objs:
            attachment_point_pos = cloth_obj.links["attachment_point"].get_position()
            particles_in_volume = self._ag_check_in_volume[arm]([attachment_point_pos])
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
                (object, object_link, attachment_point_position), i.e. ((BaseObject, RigidPrim, np.ndarray)) to the]
                contacted in-hand object
        """
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return

        ag_obj, ag_link, attachment_point_pos = ag_data

        # Find the attachment point position in the eef frame
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        attachment_point_pos_local, _ = \
            T.relative_pose_transform(attachment_point_pos, [0, 0, 0, 1], eef_link_pos, eef_link_orn)

        # Create the joint
        joint_prim_path = f"{ag_link.prim_path}/ag_constraint"
        joint_type = "FixedJoint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=ag_link.prim_path,
            body1=None,
            enabled=False,
            joint_frame_in_child_frame_pos=attachment_point_pos,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        max_force = m.ASSIST_FORCE
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
        for joint in self.finger_joints[arm]:
            j_val = joint.get_state()[0][0]
            self._ag_freeze_joint_pos[arm][joint.joint_name] = j_val

    def _dump_state(self):
        # Call super first
        state = super()._dump_state()

        # If we're using actual physical grasping, no extra state needed to save
        if self.grasping_mode == "physical":
            return state

        # Include AG_state
        state["ag_obj_constraint_params"] = self._ag_obj_constraint_params
        return state

    def _load_state(self, state):
        # If there is an existing AG object, remove it
        self.release_grasp_immediately()

        super()._load_state(state=state)

        # No additional loading needed if we're using physical grasping
        if self.grasping_mode == "physical":
            return

        # Include AG_state
        # TODO: currently does not take care of cloth objects
        # TODO: add unit tests
        for arm in state["ag_obj_constraint_params"].keys():
            if len(state["ag_obj_constraint_params"][arm]) > 0:
                data = state["ag_obj_constraint_params"][arm]
                obj = og.sim.scene.object_registry("prim_path", data["ag_obj_prim_path"])
                link = obj.links[data["ag_link_prim_path"].split("/")[-1]]
                self._ag_data[arm] = (obj, link)
                self._establish_grasp(arm=arm, ag_data=self._ag_data[arm], contact_pos=data["contact_pos"])

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

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ManipulationRobot")
        return classes
    
    @property
    def eef_usd_path(self):
        """
        Returns:
            dict(str, str): dict mapping arm name to the path to the eef usd file
        """
        raise NotImplementedError

    @property
    def teleop_rotation_offset(self):
        """
        Rotational offset that will be applied for teleoperation
        such that [0, 0, 0, 1] as action will keep the robot eef pointing at +x axis
        """
        return {arm: np.array([0, 0, 0, 1]) for arm in self.arm_names}

    def teleop_data_to_action(self, teleop_data: TeleopData) -> np.ndarray:
        """
        Generate action data from teleoperation data
        NOTE: This implementation only supports IK/OSC controller for arm and MultiFingerGripperController for gripper. 
        Overwrite this function if the robot is using a different controller.
        Args:
            teleop_data (TeleopData): teleoperation data
        Returns:
            np.ndarray: array of action data for arm and gripper
        """
        action = super().teleop_data_to_action(teleop_data)
        hands = ["left", "right"] if self.n_arms == 2 else ["right"]
        for i, hand in enumerate(hands):
            arm_name = self.arm_names[i]
            if teleop_data.is_valid[hand]:
                # arm action
                assert \
                    isinstance(self._controllers[f"arm_{arm_name}"], InverseKinematicsController) or \
                    isinstance(self._controllers[f"arm_{arm_name}"], OperationalSpaceController), \
                    f"Only IK and OSC controllers are supported for arm {arm_name}!"
                cur_eef_pos, cur_eef_orn = self.links[self.eef_link_names[arm_name]].get_position_orientation()
                if teleop_data.robot_attached:
                    target_pos, target_orn = teleop_data.transforms[hand]
                else:
                    target_pos, target_orn = cur_eef_pos, cur_eef_orn
                # get orientation relative to robot base
                base_pos, base_orn = self.get_position_orientation()
                rel_des_pos, rel_des_orn = T.relative_pose_transform(target_pos, target_orn, base_pos, base_orn)
                rel_cur_pos, _ = T.relative_pose_transform(cur_eef_pos, cur_eef_orn, base_pos, base_orn)
                action[self.arm_action_idx[arm_name]] = np.r_[rel_des_pos - rel_cur_pos, T.quat2axisangle(rel_des_orn)]
                # gripper action
                assert isinstance(self._controllers[f"gripper_{arm_name}"], MultiFingerGripperController), \
                    f"Only MultiFingerGripperController is supported for gripper {arm_name}!"
                action[self.gripper_action_idx[arm_name]] = teleop_data.grippers[hand]
        return action
