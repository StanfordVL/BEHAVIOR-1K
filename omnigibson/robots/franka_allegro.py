import os
import numpy as np
from typing import Dict, Iterable
from omni.isaac.motion_generation import LulaKinematicsSolver

import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import ManipulationRobot, GraspingPoint


class FrankaAllegro(ManipulationRobot):
    """
    Franka Robot with Allegro hand
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
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=True,

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
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self.default_joint_pos will be used instead.
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
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """

        # Run super init
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
            grasping_mode=grasping_mode,
            grasping_direction="upper",
            **kwargs,
        )

        self.allegro_ik_controller = AllegroIKController(self)

    @property
    def model_name(self):
        return "FrankaAllegro"

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Franka does not support discrete actions!")

    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params here
        for i in range(7):
            self.joints[f"panda_joint{i+1}"].damping = 100
            self.joints[f"panda_joint{i+1}"].stiffness = 1000
            self.joints[f"panda_joint{i+1}"].max_effort = 100

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        return controllers
    
    @property
    def _default_arm_ik_controller_configs(self):
        conf = super()._default_arm_ik_controller_configs
        conf[self.default_arm]["mode"] = "pose_absolute_ori"
        conf[self.default_arm]["command_input_limits"] = None
        conf[self.default_arm]["motor_type"] = "position"
        return conf

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        conf = super()._default_gripper_multi_finger_controller_configs
        conf[self.default_arm]["mode"] = "independent"
        conf[self.default_arm]["command_input_limits"] = None
        return conf

    @property
    def default_joint_pos(self):
        # position where the hand is parallel to the ground
        return np.r_[[0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16)]

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.arange(7)}

    @property
    def gripper_control_idx(self):
        # thumb.proximal, ..., thumb.tip, ..., ring.tip
        return {self.default_arm: np.array([8, 12, 16, 20, 10, 14, 18, 22, 9, 13, 17, 21, 7, 11, 15, 19])}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint_{i+1}" for i in range(7)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "base_link"}

    @property
    def finger_link_names(self):
        return {self.default_arm: [f"link_{i}_0" for i in range(16)]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: [f"joint_{i}_0" for i in range(16)]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro.usd")
    
    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro_description.yaml")}
    
    @property
    def robot_gripper_descriptor_yamls(self):
        return {
            finger: os.path.join(gm.ASSET_PATH, f"models/franka/allegro_{finger}_description.yaml")
            for finger in ["thumb", "index", "middle", "ring"]
        }

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro.urdf")
    
    @property
    def gripper_urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/allegro_hand.urdf")
    
    @property
    def disabled_collision_pairs(self):
        return [
            ["link_12_0", "part_studio_link"],
        ]
    
    @property
    def assisted_grasp_start_points(self):
        return {self.default_arm: [
            GraspingPoint(link_name=f"base_link", position=[0.015, 0, -0.03]),
            GraspingPoint(link_name=f"base_link", position=[0.015, 0, -0.08]),
            GraspingPoint(link_name=f"link_15_0_tip", position=[0, 0.015, 0.007]),
        ]}

    @property
    def assisted_grasp_end_points(self):
        return {self.default_arm: [
            GraspingPoint(link_name=f"link_3_0_tip", position=[0.012, 0, 0.007]),
            GraspingPoint(link_name=f"link_7_0_tip", position=[0.012, 0, 0.007]),
            GraspingPoint(link_name=f"link_11_0_tip", position=[0.012, 0, 0.007]),
        ]}

    @property
    def vr_rotation_offset(self):
        return {self.default_arm: T.euler2quat(np.array([0, np.pi / 2, 0]))}
    
    def remap_thumb_to_allegro(self, coord: np.ndarray) -> np.ndarray:
        """
        remap VR thumb tracking data to allegro thumb, based on the bound of the two embodiments
        Args:
            bound (np.ndarray) 3D coord of thumb tracking position in the base_link frame
        """
        hand_bound = np.array([[0.005, 0.095], [-0.026, 0.104], [-0.005, 0.053]])    # bound for hand
        allegro_bound = np.array([[0.017, 0.117], [-0.037, 0.132], [-0.096, -0.009]])    # bound for allegro 
        for i in range(3):
            coord[i] = np.interp(coord[i], hand_bound[i], allegro_bound[i])
        return coord


    def gen_action_from_vr_data(self, vr_data: dict):
        action = np.zeros(22)
        if "hand_data" in vr_data:
            hand_data = vr_data["hand_data"]
            if "right" in hand_data and "raw" in hand_data["right"]:
                # The center of allegro hand lies somewhere between palm and middle proximal
                target_pos = (hand_data["right"]["raw"]["pos"][0] + hand_data["right"]["raw"]["pos"][12]) / 2
                target_orn = T.quat_multiply(hand_data["right"]["raw"]["orn"][0], self.vr_rotation_offset[self.default_arm])
                cur_robot_eef_pos, cur_robot_eef_orn = self.links[self.eef_link_names[self.default_arm]].get_position_orientation()
                base_pos, base_orn = self.get_position_orientation()
                rel_target_pos, rel_target_orn = T.relative_pose_transform(target_pos + [0.06, 0, 0], target_orn, base_pos, base_orn)
                rel_cur_pos = T.relative_pose_transform(cur_robot_eef_pos, cur_robot_eef_orn, base_pos, base_orn)[0]
                action[:6] = np.r_[rel_target_pos - rel_cur_pos, T.quat2axisangle(rel_target_orn)]
                # joint order: index, middle, pinky
                angles = hand_data["right"]["angles"]
                for f_idx in range(3):
                    for j_idx in range(3):
                        action[11 + f_idx * 4 + j_idx] = angles[f_idx + 1][j_idx]
                # specifically, use ik for thumb
                thumb_pos, thumb_orn = hand_data["right"]["raw"]["pos"][5], hand_data["right"]["raw"]["orn"][5]
                local_thumb_pos, local_thumb_orn = T.relative_pose_transform(thumb_pos, thumb_orn, target_pos, target_orn)
                local_thumb_pos = self.remap_thumb_to_allegro(coord=local_thumb_pos)
                target_thumb_pos = T.pose_transform(
                    cur_robot_eef_pos, cur_robot_eef_orn, local_thumb_pos, local_thumb_orn,
                )[0]
                action[6: 10] = self.allegro_ik_controller.solve({"thumb": target_thumb_pos})["thumb"]

        else:
            action_from_controller = super().gen_action_from_vr_data(vr_data)
            action[:6] = action_from_controller[:6]
            action[6:] = action_from_controller[6]
        return action


class AllegroIKController:
    """
    IK controller for Allegro hand, based on the LulaKinematicsSolver
    """
    def __init__(self, robot: FrankaAllegro, max_iter=150) -> None:
        """
        Initializes the IK controller
        Args:
            robot (FrankaAllegro): the Franka Allegro robot
            max_iter (int): maximum number of iterations for the IK solver, default is 100.
        """
        self.robot = robot
        self.fingers = {
            "ring":     ("link_3_0_tip",    np.array([7, 11, 15, 19])),     # finger name, finger tip link name, joint indices
            "middle":   ("link_7_0_tip",    np.array([9, 13, 17, 21])),
            "index":    ("link_11_0_tip",   np.array([10, 14, 18, 22])),
            "thumb":    ("link_15_0_tip",   np.array([8, 12, 16, 20])), 
        }
        self.finger_ik_solvers = {}
        for finger in self.fingers.keys():
            self.finger_ik_solvers[finger]  = LulaKinematicsSolver(
                robot_description_path = robot.robot_gripper_descriptor_yamls[finger],
                urdf_path = robot.gripper_urdf_path
            )
            self.finger_ik_solvers[finger].ccd_max_iterations = max_iter

    def solve(self, target_gripper_pos: Dict[str, Iterable[float]]) -> Dict[str, np.ndarray]:
        """
        compute the joint positions given the position of each finger tip
        Args:
            target_gripper_pos (Dict[str, Iterable[float]]): (finger name, the 3-array of target positions of the finger tips)
        Returns:
            Dict[str, np.ndarray]: (finger name, 4-array of joint positions in order of proximal to tip)
        """
        target_joint_positions = {}
        # get the current finger joint positions
        finger_joint_positions = self.robot.get_joint_positions()
        # get current hand base pose
        hand_base_pos, hand_base_orn = self.robot.links["base_link"].get_position_orientation()
        for finger_name, pos in target_gripper_pos.items():
            target_joint_positions[finger_name] = finger_joint_positions[self.fingers[finger_name][1]]
            # Grab the finger joint positions in order to reach the desired finger pose
            self.finger_ik_solvers[finger_name].set_robot_base_pose(hand_base_pos, T.convert_quat(hand_base_orn, "wxyz"))
            finger_joint_pos, success = self.finger_ik_solvers[finger_name].compute_inverse_kinematics(
                frame_name=self.fingers[finger_name][0],
                target_position=pos,
                target_orientation=None,
                warm_start=finger_joint_positions[self.fingers[finger_name][1]]
            )
            if success:
                target_joint_positions[finger_name] = finger_joint_pos
            
        return target_joint_positions
