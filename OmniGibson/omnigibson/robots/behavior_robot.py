import itertools
import math
import os
from abc import ABC
from collections import OrderedDict
from functools import cached_property
from typing import Iterable, List, Literal, Tuple

import torch as th

import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.python_utils import classproperty

m = create_module_macros(module_path=__file__)

# component suffixes for the 6-DOF arm joint names
m.COMPONENT_SUFFIXES = ["x", "y", "z", "rx", "ry", "rz"]

# Offset between the body and parts
m.HEAD_TO_BODY_OFFSET = th.tensor([0, 0, -0.4], dtype=th.float32)
m.HAND_TO_BODY_OFFSET = {
    "left": th.tensor([0, -0.15, -0.4], dtype=th.float32),
    "right": th.tensor([0, 0.15, -0.4], dtype=th.float32),
}
m.BODY_HEIGHT_OFFSET = 0.45

# Hand parameters
m.HAND_GHOST_HAND_APPEAR_THRESHOLD = 0.15
m.THUMB_2_POS = th.tensor([0, -0.02, -0.05], dtype=th.float32)
m.THUMB_1_POS = th.tensor([0, -0.015, -0.02], dtype=th.float32)
m.PALM_CENTER_POS = th.tensor([0, -0.04, 0.01], dtype=th.float32)
m.PALM_BASE_POS = th.tensor([0, 0, 0.015], dtype=th.float32)
m.FINGER_TIP_POS = th.tensor([0, -0.025, -0.055], dtype=th.float32)

# Hand link index constants
m.PALM_LINK_NAME = "palm"
m.FINGER_MID_LINK_NAMES = ("Tproximal", "Iproximal", "Mproximal", "Rproximal", "Pproximal")
m.FINGER_TIP_LINK_NAMES = ("Tmiddle", "Imiddle", "Mmiddle", "Rmiddle", "Pmiddle")
m.THUMB_LINK_NAME = "Tmiddle"

# joint parameters
m.BASE_JOINT_STIFFNESS = 1e8
m.BASE_JOINT_MAX_EFFORT = 7500
m.ARM_JOINT_STIFFNESS = 1e6
m.ARM_JOINT_MAX_EFFORT = 300
m.FINGER_JOINT_STIFFNESS = 1e3
m.FINGER_JOINT_MAX_EFFORT = 50
m.FINGER_JOINT_MAX_VELOCITY = math.pi * 4


class BehaviorRobot(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    A humanoid robot that can be used in VR as an avatar. It has two hands, a body and a head with two cameras.
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
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
        action_normalize=False,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities="rgb",
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        # Unique to ManipulationRobot
        grasping_mode="assisted",
        finger_static_friction=None,
        finger_dynamic_friction=None,
        # unique to BehaviorRobot
        use_ghost_hands=True,
        **kwargs,
    ):
        """
        Initializes BehaviorRobot
        Args:
            use_ghost_hands (bool): whether to show ghost hand when the robot hand is too far away from the controller
        """

        super(BehaviorRobot, self).__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=True,
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
            grasping_mode=grasping_mode,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            grasping_direction="upper",
            **kwargs,
        )

        # setup eef parts
        self.parts = OrderedDict()
        for arm_name in self.arm_names:
            self.parts[arm_name] = BRPart(
                name=arm_name,
                parent=self,
                relative_prim_path=f"/{arm_name}_palm",
                eef_type="hand",
                offset_to_body=m.HAND_TO_BODY_OFFSET[arm_name],
                **kwargs,
            )
        self.parts["head"] = BRPart(
            name="head",
            parent=self,
            relative_prim_path="/eye",
            eef_type="head",
            offset_to_body=m.HEAD_TO_BODY_OFFSET,
            **kwargs,
        )

        # whether to use ghost hands (visual markers to help visualize current vr hand pose)
        self._use_ghost_hands = use_ghost_hands
        # prim for the world_base_fixed_joint, used to reset the robot pose
        self._world_base_fixed_joint_prim = None
        # whether hand or body is in contact with other objects (we need this since checking contact list is costly)
        self._part_is_in_contact = {hand_name: False for hand_name in self.arm_names + ["body"]}

    @property
    def usd_path(self):
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), "models/behavior_robot/usd/BehaviorRobot.usd")

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @cached_property
    def eef_link_names(self):
        dic = {arm: f"{arm}_{m.PALM_LINK_NAME}" for arm in self.arm_names}
        dic["head"] = "head"
        return dic

    @cached_property
    def arm_link_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {arm: [f"{arm}_{component}" for component in m.COMPONENT_SUFFIXES] for arm in self.arm_names + ["head"]}

    @cached_property
    def finger_link_names(self):
        return {
            arm: [
                f"{arm}_{link_name}" for link_name in itertools.chain(m.FINGER_MID_LINK_NAMES, m.FINGER_TIP_LINK_NAMES)
            ]
            for arm in self.arm_names
        }

    @cached_property
    def base_joint_names(self):
        return [f"base_{component}_joint" for component in m.COMPONENT_SUFFIXES]

    @cached_property
    def camera_joint_names(self):
        return [f"head_{component}_joint" for component in m.COMPONENT_SUFFIXES]

    @cached_property
    def arm_joint_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {
            eef: [f"{eef}_{component}_joint" for component in m.COMPONENT_SUFFIXES] for eef in self.arm_names + ["head"]
        }

    @cached_property
    def finger_joint_names(self):
        return {
            arm: (
                # palm-to-proximal joints.
                [f"{arm}_{to_link}__{arm}_{m.PALM_LINK_NAME}" for to_link in m.FINGER_MID_LINK_NAMES]
                +
                # proximal-to-tip joints.
                [
                    f"{arm}_{to_link}__{arm}_{from_link}"
                    for from_link, to_link in zip(m.FINGER_MID_LINK_NAMES, m.FINGER_TIP_LINK_NAMES)
                ]
            )
            for arm in self.arm_names
        }

    @property
    def _default_joint_pos(self):
        return th.zeros(self.n_joints)

    @property
    def _raw_controller_order(self):
        controllers = ["base", "camera"]
        for arm_name in self.arm_names:
            controllers += [f"arm_{arm_name}", f"gripper_{arm_name}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = {"base": "JointController", "camera": "JointController"}
        controllers.update({f"arm_{arm_name}": "JointController" for arm_name in self.arm_names})
        controllers.update({f"gripper_{arm_name}": "MultiFingerGripperController" for arm_name in self.arm_names})
        return controllers

    @property
    def _default_base_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "motor_type": "position",
            "dof_idx": self.base_control_idx,
            "command_input_limits": None,
        }

    @property
    def _default_arm_joint_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_input_limits": None,
                "use_delta_commands": False,
            }
        return dic

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_input_limits": None,
                "mode": "independent",
                "inverted": True,
            }
        return dic

    @property
    def _default_camera_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "command_input_limits": None,
            "use_delta_commands": False,
        }

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
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_input_limits": None,
                "use_delta_commands": False,
            }
        return dic

    @property
    def _default_controller_config(self):
        controllers = {
            "base": {"JointController": self._default_base_joint_controller_config},
            "camera": {"JointController": self._default_camera_joint_controller_config},
        }
        controllers.update(
            {
                f"arm_{arm_name}": {"JointController": self._default_arm_joint_controller_configs[arm_name]}
                for arm_name in self.arm_names
            }
        )
        controllers.update(
            {
                f"gripper_{arm_name}": {
                    "MultiFingerGripperController": self._default_gripper_multi_finger_controller_configs[arm_name],
                    "JointController": self._default_gripper_joint_controller_configs[arm_name],
                }
                for arm_name in self.arm_names
            }
        )
        return controllers

    def load(self, scene):
        prim = super(BehaviorRobot, self).load(scene)
        for part in self.parts.values():
            part.load(scene)
        return prim

    def _post_load(self):
        super()._post_load()

    def _create_discrete_action_space(self):
        raise ValueError("BehaviorRobot does not support discrete actions!")

    def update_controller_mode(self):
        super().update_controller_mode()
        # set base joint properties
        for joint_name in self.base_joint_names:
            self.joints[joint_name].stiffness = m.BASE_JOINT_STIFFNESS
            self.joints[joint_name].max_effort = m.BASE_JOINT_MAX_EFFORT

        # set arm joint properties
        for arm in self.arm_joint_names:
            for joint_name in self.arm_joint_names[arm]:
                self.joints[joint_name].stiffness = m.ARM_JOINT_STIFFNESS
                self.joints[joint_name].max_effort = m.ARM_JOINT_MAX_EFFORT

        # set finger joint properties
        for arm in self.finger_joint_names:
            for joint_name in self.finger_joint_names[arm]:
                self.joints[joint_name].stiffness = m.FINGER_JOINT_STIFFNESS
                self.joints[joint_name].max_effort = m.FINGER_JOINT_MAX_EFFORT
                self.joints[joint_name].max_velocity = m.FINGER_JOINT_MAX_VELOCITY

    @cached_property
    def base_footprint_link_name(self):
        """
        Name of the actual root link that we are interested in.
        """
        return "base"

    def get_position_orientation(self, frame: Literal["world", "scene"] = "world", clone=True):
        """
        Gets robot's pose with respect to the specified frame.

        Args:
            frame (Literal): frame to get the pose with respect to. Default to world.
                scene frame get position relative to the scene.
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            2-tuple:
                - th.Tensor: (x,y,z) position in the specified frame
                - th.Tensor: (x,y,z,w) quaternion orientation in the specified frame
        """
        return self.base_footprint_link.get_position_orientation(frame=frame, clone=clone)

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Sets behavior robot's pose with respect to the specified frame

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
            frame (Literal): frame to set the pose with respect to, defaults to "world".
                scene frame set position relative to the scene.
        """
        super().set_position_orientation(position, orientation, frame=frame)
        # Move the joint frame for the world_base_joint
        if self._world_base_fixed_joint_prim is not None:
            if position is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
            if orientation is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                    lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
                )

    @property
    def _assisted_grasp_start_points(self):
        side_coefficients = {"left": th.tensor([1, -1, 1]), "right": th.tensor([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_{m.PALM_LINK_NAME}", position=m.PALM_BASE_POS),
                GraspingPoint(
                    link_name=f"{arm}_{m.PALM_LINK_NAME}", position=m.PALM_CENTER_POS * side_coefficients[arm]
                ),
                GraspingPoint(link_name=f"{arm}_{m.THUMB_LINK_NAME}", position=m.THUMB_1_POS * side_coefficients[arm]),
                GraspingPoint(link_name=f"{arm}_{m.THUMB_LINK_NAME}", position=m.THUMB_2_POS * side_coefficients[arm]),
            ]
            for arm in self.arm_names
        }

    @property
    def _assisted_grasp_end_points(self):
        side_coefficients = {"left": th.tensor([1, -1, 1]), "right": th.tensor([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_{finger}", position=m.FINGER_TIP_POS * side_coefficients[arm])
                for finger in m.FINGER_TIP_LINK_NAMES
            ]
            for arm in self.arm_names
        }

    def update_hand_contact_info(self):
        """
        Helper function that updates the contact info for the hands and body.
        Can be used in the future with device haptics to provide collision feedback.
        """
        self._part_is_in_contact["body"] = len(self.links["body"].contact_list()) > 0
        for hand_name in self.arm_names:
            self._part_is_in_contact[hand_name] = len(self.eef_links[hand_name].contact_list()) > 0 or th.any(
                [len(finger.contact_list()) > 0 for finger in self.finger_links[hand_name]]
            )

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        """
        Generates an action for the BehaviorRobot to perform based on teleop action data dict.

        Action space (all non-normalized values that will be clipped if they are too large)
        Body:
        - 6DOF pose - relative to body frame from previous frame
        Eye:
        - 6DOF pose - relative to body frame (where the body will be after applying this frame's action)
        Left hand, right hand (in that order):
        - 6DOF pose - relative to body frame (same as above)
        - 10DOF gripper joint rotation

        Total size: 44
        """
        # Actions are stored as 1D numpy array
        action = th.zeros(self.action_dim)
        # Update body action space
        if teleop_action.is_valid["head"]:
            head_pos, head_orn = teleop_action.head[:3], T.euler2quat(teleop_action.head[3:6])
            des_body_pos = head_pos - th.tensor([0, 0, m.BODY_HEIGHT_OFFSET])
            des_body_rpy = th.tensor([0, 0, T.quat2euler(head_orn)[2][0]])
            des_body_orn = T.euler2quat(des_body_rpy)
        else:
            des_body_pos, des_body_orn = self.get_position_orientation()
            des_body_rpy = th.stack(T.quat2euler(des_body_orn)).squeeze(1)
        action[self.controller_action_idx["base"]] = th.cat((des_body_pos, des_body_rpy))
        # Update action space for other VR objects
        for part_name, eef_part in self.parts.items():
            # Process local transform adjustments
            hand_data = 0
            if teleop_action.is_valid[part_name]:
                des_world_part_pos, des_world_part_orn = (
                    teleop_action[part_name][:3],
                    T.euler2quat(teleop_action[part_name][3:6]),
                )
                if part_name in self.arm_names:
                    # compute gripper action
                    if hasattr(teleop_action, "hand_data"):
                        # hand tracking mode, compute joint rotations for each independent hand joint
                        hand_data = teleop_action.hand_data[part_name]
                        hand_data = hand_data[:, :2].T.reshape(-1)
                    else:
                        # controller mode, map trigger fraction from [0, 1] to [-1, 1] range.
                        hand_data = teleop_action[part_name][6] * 2 - 1
                    action[self.controller_action_idx[f"gripper_{part_name}"]] = hand_data
                    # update ghost hand if necessary
                    if self._use_ghost_hands:
                        self.parts[part_name].update_ghost_hands(des_world_part_pos, des_world_part_orn)
            else:
                des_world_part_pos, des_world_part_orn = eef_part.local_position_orientation

            # Get local pose with respect to the new body frame
            des_local_part_pos, des_local_part_orn = T.relative_pose_transform(
                des_world_part_pos, des_world_part_orn, des_body_pos, des_body_orn
            )
            # apply shoulder position offset to the part transform to get final destination pose
            des_local_part_pos, des_local_part_orn = T.pose_transform(
                eef_part.offset_to_body, [0, 0, 0, 1], des_local_part_pos, des_local_part_orn
            )
            des_part_rpy = th.stack(T.quat2euler(des_local_part_orn)).squeeze(1)
            controller_name = "camera" if part_name == "head" else "arm_" + part_name
            action[self.controller_action_idx[controller_name]] = th.cat((des_local_part_pos, des_part_rpy))
            # If we reset, teleop the robot parts to the desired pose
            if part_name in self.arm_names and teleop_action.reset[part_name]:
                self.parts[part_name].set_position_orientation(position=des_local_part_pos, orientation=des_part_rpy)
        return action


class BRPart(ABC):
    """This is the interface that all BehaviorRobot eef parts must implement."""

    def __init__(
        self, name: str, parent: BehaviorRobot, relative_prim_path: str, eef_type: str, offset_to_body: List[float]
    ) -> None:
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        Args:
            name (str): unique name of this BR part
            parent (BehaviorRobot): the parent BR object
            relative_prim_path (str): relative prim path to the root link of the eef
            eef_type (str): type of eef. One of hand, head
            offset_to_body (List[float]): relative POSITION offset between the rz link and the eef link.
        """
        self.name = name
        self.parent = parent
        self.relative_prim_path = relative_prim_path
        self.eef_type = eef_type
        self.offset_to_body = offset_to_body

        self.ghost_hand = None
        self._root_link = None

    def load(self, scene) -> None:
        self.scene = scene
        self._root_link = self.parent.links[self.relative_prim_path.replace("/", "")]
        # setup ghost hand
        if self.eef_type == "hand" and self.parent._use_ghost_hands:
            gh_name = f"ghost_hand_{self.name}"
            self.ghost_hand = USDObject(
                relative_prim_path=f"/{gh_name}",
                usd_path=os.path.join(
                    get_dataset_path("omnigibson-robot-assets"), f"models/behavior_robot/usd/{gh_name}.usd"
                ),
                name=gh_name,
                scale=0.001,
                visible=False,
                visual_only=True,
            )
            self.scene.add_object(self.ghost_hand)

    @property
    def local_position_orientation(self) -> Tuple[Iterable[float], Iterable[float]]:
        """
        Get local position and orientation w.r.t. to the body
        Returns:
            Tuple[Array[x, y, z], Array[x, y, z, w]]

        """
        return T.relative_pose_transform(*self.get_position_orientation(), *self.parent.get_position_orientation())

    def get_position_orientation(self, clone=True) -> Tuple[Iterable[float], Iterable[float]]:
        """
        Get position and orientation in the world space

        Args:
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            Tuple[Array[x, y, z], Array[x, y, z, w]]
        """
        return self._root_link.get_position_orientation(clone=clone)

    def set_position_orientation(self, pos: Iterable[float], orn: Iterable[float]) -> None:
        """
        Call back function to set the base's position
        """
        self.parent.joints[f"{self.name}_x_joint"].set_pos(pos[0], drive=False)
        self.parent.joints[f"{self.name}_y_joint"].set_pos(pos[1], drive=False)
        self.parent.joints[f"{self.name}_z_joint"].set_pos(pos[2], drive=False)
        self.parent.joints[f"{self.name}_rx_joint"].set_pos(orn[0], drive=False)
        self.parent.joints[f"{self.name}_ry_joint"].set_pos(orn[1], drive=False)
        self.parent.joints[f"{self.name}_rz_joint"].set_pos(orn[2], drive=False)

    def update_ghost_hands(self, pos: Iterable[float], orn: Iterable[float]) -> None:
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        Args:
            pos (Iterable[float]): list of positions [x, y, z]
            orn (Iterable[float]): list of rotations [x, y, z, w]
        """
        assert self.eef_type == "hand", "ghost hand is only valid for BR hand!"
        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position_orientation(position=pos, orientation=orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = th.norm(pos - self.get_position_orientation()[0])
        should_visible = dist_to_real_controller > m.HAND_GHOST_HAND_APPEAR_THRESHOLD

        # Only toggle visibility if we are transition from hidden to unhidden, or the other way around
        if self.ghost_hand.visible is not should_visible:
            self.ghost_hand.visible = should_visible
