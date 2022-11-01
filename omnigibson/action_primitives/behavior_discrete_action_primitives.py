import copy
import logging
import random
import time
from enum import IntEnum
import bddl

import gym
import numpy as np
# import pybullet as p
import omnigibson as ig

from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from omnigibson.controllers import ControlType, JointController
from omnigibson.object_states.pose import Pose
from omnigibson.robots.manipulation_robot import IsGraspingState
from omnigibson.utils.motion_planning_utils import MotionPlanner
from omnigibson.utils.transform_utils import mat2euler, quat2mat

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


skill_object_offset_params = {
    0: {  # skill id: move
        "printer.n.03_1": [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
        "table.n.02_1": [0, -0.6, 0, 0.5 * np.pi],
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi],
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi],
        "countertop.n.01_1": [0.0, -0.8, 0, 0.5 * np.pi],  # [0.1, 0.5, 0.8 1.0]
        # 'countertop.n.01_1': [[0.0, -0.8, 0, 0.1 * np.pi], [0.0, -0.8, 0, 0.5 * np.pi], [0.0, -0.8, 0, 0.8 * np.pi],],  # [0.1, 0.5, 0.8 1.0]
        # # Ihlen_1_int, 0
        # 'hamburger.n.01_1': [0, 0.8, 0, -0.5 * np.pi],
        # 'hamburger.n.01_2': [0, 0.8, 0, -0.5 * np.pi],
        # 'hamburger.n.01_3': [-0.2, 0.7, 0, -0.6 * np.pi],
        # 'ashcan.n.01_1': [-0.2, -0.5, 0, 0.4 * np.pi],
        # 'countertop.n.01_1': [-0.5, -0.6, 0, 0.5 * np.pi],
        # putting_away_Halloween_decorations
        "pumpkin.n.02_1": [0.5, 0.0, 0.0, 0.7 * np.pi],
        "pumpkin.n.02_2": [0.3, -0.45, 0, 0.5 * np.pi],
        # "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi],
        "cabinet.n.01_1": [0.7, -0.83, 0, 0.5 * np.pi],
        # "cabinet.n.01_1": [-0.1, -1.5, 0, 0.5 * np.pi],
    },
    1: {  # pick
        "printer.n.03_1": [-0.2, 0.0, 0.2],  # dx, dy, dz
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0.0, 0.0, 0.025],
        "hamburger.n.01_2": [
            0.0,
            0.0,
            0.025,
        ],
        "hamburger.n.01_3": [
            0.0,
            0.0,
            0.025,
        ],
        # putting_away_Halloween_decorations
        # "pumpkin.n.02_1": [0.0,0.0,0.0,],
        # "pumpkin.n.02_2": [0.0,0.0,0.0,],
        # "pumpkin.n.02_1": [0.0,0.0,-0.1,],
        # "pumpkin.n.02_2": [0.0,0.0,-0.1,],
        # "cabinet.n.01_1": [0.35, -0.65, -0.1],  # dx, dy, dz
        "pumpkin.n.02_1": [0.0,0.0,0,0],
        "pumpkin.n.02_2": [0.0,0.0,0,0],
    },
    2: {  # place
        "table.n.02_1": [0, 0, 0.5],  # dx, dy, dz
        # Pomaria_1_int, 2
        # 'ashcan.n.01_1': [0, 0, 0.5],
        # Ihlen_1_int, 0
        "ashcan.n.01_1": [0, 0, 0.5],
        # putting_away_Halloween_decorations
        #"cabinet.n.01_1": [0.3, -0.55, 0.25],
        "cabinet.n.01_1": [0.2, -0.58, 0.5],
    },
    3: {  # toggle
        "printer.n.03_1": [-0.3, -0.25, 0.23],  # dx, dy, dz
    },
    4: {  # pull
        "cabinet.n.01_1": [0.35, -0.3, 0.35, -1, 0, 0],  # dx, dy, dz
    },
    5: {  # push open
        "cabinet.n.01_1": [0.2, -0.58, 0.5, -1, 0, 0],  # dx, dy, dz
    },
    6: {  # vis_pick
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025],
        "hamburger.n.01_2": [
            0,
            -0.7,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        "hamburger.n.01_3": [
            0,
            -0.8,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        # vis: putting_away_Halloween_decorations
        "pumpkin.n.02_1": [
            0.4,
            0.0,
            0.0,
            1.0 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        "pumpkin.n.02_2": [
            0,
            -0.5,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
    },
    7: {  # vis_place
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi, 0, 0, 0.5],
        # vis: putting_away_Halloween_decorations
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi, 0.3, -0.60, 0.25],
    },
    8: {  # vis pull
        "cabinet.n.01_1": [0.3, -0.55, 0.35],  # dx, dy, dz
    },
    9: {  # vis push
        "cabinet.n.01_1": [0.3, -0.8, 0.35],  # dx, dy, dz
    },
}

action_list_installing_a_printer = [
    [0, "printer.n.03_1"],  # skill id, target_obj
    [1, "printer.n.03_1"],
    [0, "table.n.02_1"],
    [2, "table.n.02_1"],
    [3, "printer.n.03_1"],
]

action_list_throwing_away_leftovers_v1 = [
    [0, "hamburger.n.01_1"],
    [1, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [2, "ashcan.n.01_1"],  # place
    [0, "hamburger.n.01_2"],
    [1, "hamburger.n.01_2"],
    [0, "hamburger.n.01_3"],
    [1, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers_discrete = [
    [0, "countertop.n.01_1", 0],
    [0, "countertop.n.01_1", 1],
    [0, "countertop.n.01_1", 2],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers = [
    [0, "countertop.n.01_1", 0],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_putting_leftovers_away = [
    [0, "pasta.n.02_1"],
    [1, "pasta.n.02_1"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2"],
    [1, "pasta.n.02_2"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_3"],
    [1, "pasta.n.02_2_3"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_4"],
    [1, "pasta.n.02_2_4"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
]


# vis version: full set
action_list_putting_away_Halloween_decorations = [
    [0, "cabinet.n.01_1"],  # navigate_to 0
    [5, "cabinet.n.01_1"],  # push open 1
    [0, "pumpkin.n.02_1"],  # navigate_to 2
    [1, "pumpkin.n.02_1"],  # pick 3
    [2, "cabinet.n.01_1"],  # place 4
    # [0, "pumpkin.n.02_2"],  # navigate_to 5
    # [1, "pumpkin.n.02_2"],  # pick 6
]

action_list_room_rearrangement = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    # [7, 'cabinet.n.01_1'],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]

action_dict = {
    "installing_a_printer": action_list_installing_a_printer,
    "throwing_away_leftovers": action_list_throwing_away_leftovers,
    "putting_leftovers_away": action_list_putting_leftovers_away,
    "putting_away_Halloween_decorations": action_list_putting_away_Halloween_decorations,
    "throwing_away_leftovers_discrete": action_list_throwing_away_leftovers_discrete,
    "room_rearrangement": action_list_room_rearrangement,
}


class B1KActionPrimitive(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH_OPEN = 5
    DUMMY = 10


class BehaviorActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, env, task, scene, robot, arm=None, execute_free_space_motion=False, mode=''):
        """ """
        super().__init__(env, task, scene, robot)

        if self.env.task_config["activity_name"] == "putting_away_Halloween_decorations":
            pumpkins = self.env.scene.object_registry("category", "pumpkin")
            for pumpkin in pumpkins:
                for link in pumpkin.links.values():
                    link.mass = 0.5

        if arm is None:
            arm = self.robot.default_arm
        self.arm = arm
        logger.info("Using arm: {}".format(self.arm))

        if robot.model_name in ["Tiago", "Fetch"]:
            assert not robot.rigid_trunk, "The APs will use the trunk of Fetch/Tiago, it can't be rigid"
        assert isinstance(
            robot._controllers["arm_" + self.arm], JointController
        ), "The arm to use with the primitives must be controlled in joint space"
        assert (
            robot._controllers["arm_" + self.arm].control_type == ControlType.POSITION
        ), "The arm to use with the primitives must be controlled in absolute positions"
        assert not robot._controllers[
            "arm_" + self.arm
        ].use_delta_commands, "The arm to use with the primitives cannot be controlled with deltas"

        self.controller_functions = {
            B1KActionPrimitive.NAVIGATE_TO: self._navigate_to,
            B1KActionPrimitive.PICK: self._pick,
            B1KActionPrimitive.PLACE: self._place,
            B1KActionPrimitive.TOGGLE: self._toggle,
            B1KActionPrimitive.PULL: self._pull,
            B1KActionPrimitive.PUSH_OPEN: self._push_open,
            B1KActionPrimitive.DUMMY: self._dummy,
        }

        if self.env.config["task"] == "throwing_away_leftovers":
            self.action_list = action_dict["throwing_away_leftovers_discrete"]
            skill_object_offset_params[0]["countertop.n.01_1"] = [
                [0.0, -0.8, 0, 0.1 * np.pi],
                [0.0, -0.8, 0, 0.5 * np.pi],
                [0.0, -0.8, 0, 0.8 * np.pi],
            ]
        else:
            self.action_list = action_dict[self.env.task_config["activity_name"]]
        self.num_discrete_action = len(self.action_list)
        self.initial_pos_dict = {}  # for checking object pos change/movement
        full_observability_2d_planning = True
        collision_with_pb_2d_planning = True
        self.planner = MotionPlanner(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=False,
            visualize_2d_result=False,
            check_selfcollisions_in_path=True,
            check_collisions_with_env_in_path=False,
            mode=mode,
        )
        self.default_direction = np.array((0.0, 0.0, -1.0))  # default hit normal
        self.execute_free_space_motion = execute_free_space_motion

        # Whether we check if the objects have moved from the previous navigation attempt and do not try to navigate
        # to them if they have (this avoids moving to objects after pick and place)
        self.obj_pose_check = True
        self.task_obj_list = self.env.task.object_scope
        self.print_log = True
        self.skip_base_planning = True # False  # True
        self.skip_arm_planning = True # False # True
        self.is_grasping = False
        self.fast_execution = False
        self.is_close_to_cabinet = False
        self.ag_name_mapping = {'pumpkin_0': 'pumpkin.n.02_1', 'pumpkin_1': 'pumpkin.n.02_2',}
        self.agent_location = 2
        self.view_location = {
            'cabinet.n.01_1': np.array([0.7746856808093053, -2.1980548994181603, 1.6858723008716936e-06]),
            'pumpkin.n.02_1': np.array([0.5010705671572231, -2.8976973969861453, 4.691743125169298]),
            'pumpkin.n.02_2': np.array([0.11446791576161452, 0.44583753843399887, 3.24755534800043])}


    def get_action_space(self):
        return gym.spaces.Discrete(self.num_discrete_action)

    def apply(self, action_index):
        if action_index == 10:
            yield from self._dummy()
        else:
            primitive_obj_pair = self.action_list[action_index]
            yield from self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1])

    def _get_obj_in_hand(self):
        return self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.

    def _get_still_action(self):
        # The camera and arm(s) and any other absolution joint position controller will be controlled with absolute positions
        joint_positions = self.robot.get_joint_positions()
        action = np.zeros(self.robot.action_dim)
        for controller_name, controller in self.robot.controllers.items():
            if (
                isinstance(controller, JointController)
                and controller.control_type == ControlType.POSITION
                and not controller.use_delta_commands
                # and not controller.use_constant_goal_position
            ):
                action_idx = self.robot.controller_action_idx[controller_name]
                joint_idx = self.robot.controller_joint_idx[controller_name]
                logger.debug(
                    "Setting action to absolute position for {} in action dims {} corresponding to joint idx {}".format(
                        controller_name, action_idx, joint_idx
                    )
                )
                action[action_idx] = joint_positions[joint_idx]
        if self.robot.is_grasping() == IsGraspingState.TRUE:
            # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
        return action

    def _execute_ee_path(
        self, path, ignore_failure=False, stop_on_contact=False, reverse_path=False, while_grasping=False
    ):
        full_body_action = self._get_still_action()
        for arm_action in path if not reverse_path else reversed(path):
            if stop_on_contact and len(self.robot._find_gripper_contacts(arm=self.arm)[0]) != 0:
                return
            # This assumes the arms are called "arm_"+self.arm. Maybe some robots do not follow this convention
            arm_controller_action_idx = self.robot.controller_action_idx["arm_" + self.arm]
            full_body_action[arm_controller_action_idx] = arm_action

            if while_grasping:
                # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
                full_body_action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
            yield full_body_action

        if stop_on_contact:
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "No contact was made.")


    def _execute_grasp(self):
        action = self._get_still_action().tolist()
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        action[int(self.robot.controller_action_idx["gripper_" + self.arm][0])] = -1.0
        action = np.asarray(action)

        grasping_steps = 5 if self.fast_execution else 9 # 10
        for i in range(grasping_steps):
            yield action
        grasped_object = self._get_obj_in_hand()
        if grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )
        else:
            logger.info("Execution of grasping ended with grasped object {}".format(grasped_object))

    def _execute_ungrasp(self):
        action = self._get_still_action().tolist()

        action[int(self.robot.controller_action_idx["gripper_" + self.arm][0])] = 1.0

        action = np.asarray(action)

        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        ungrasping_steps = 5 if self.fast_execution else 9 # 10
        for i in range(ungrasping_steps):
            yield action


        grasped_object = self._get_obj_in_hand()
        if not grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object detected in hand after executing ungrasp.",
            )
        else:
            logger.info("Execution of ungrasping ended with grasped object None")

    def _dummy(self):
        logger.info("Dummy")
        obj = self._get_obj_in_hand()
        self.planner.visualize_arm_path(
            [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
            grasped_obj=obj,
        )
        yield self._get_still_action()
        logger.info("Finished dummy")

    def _navigate_to(self, object_name):
        obj = self._get_obj_in_hand()
        if not obj is None:
            if obj.name in self.ag_name_mapping and object_name == self.ag_name_mapping[obj.name]:
                logger.info("Cannot navigate to object {}, already in hand".format(object_name))
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "Cannot navigate to object {}, already in hand".format(object_name),
                    {"object_to_navigate": object_name},
                )

        logger.info("Navigating to object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.NAVIGATE_TO][object_name]

        moved_distance_threshold = 1e-1
        if self.obj_pose_check:
            obj_pos = self.env.task.object_scope[object_name].states[Pose].get_value()[0]
            if object_name in ["pumpkin.n.02_1", "pumpkin.n.02_2"]:
                if object_name not in self.initial_pos_dict:
                    self.initial_pos_dict[object_name] = obj_pos
                else:
                    moved_distance = np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos))
                    if moved_distance > moved_distance_threshold:
                        raise ActionPrimitiveError(
                            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                            "Object moved from its initial location",
                            {"object_to_navigate": object_name},
                        )

        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        # acquire the base direction
        euler = mat2euler(mat)
        target_yaw = euler[-1] + params[3]

        objects_to_ignore = []
        if self.robot.is_grasping() == IsGraspingState.TRUE:
            objects_to_ignore = [self.robot._ag_obj_in_hand[self.arm]]

        # Plan a 2D motion
        plan = self.planner.plan_base_motion(
            [obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw],
            plan_full_base_motion=not self.skip_base_planning,
            objects_to_ignore=objects_to_ignore,
        )

        # If the planner suceeded, we visualize it and keep the last location
        if plan is not None and len(plan) > 0:
            self.planner.visualize_base_path(plan, keep_last_location=True)
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No base path found to object",
                {"object_to_navigate": object_name},
            )

        # Stop the robot after teleporting it
        self.robot.keep_still()
        # This action will get executed by the wrapping environment. It should keep the robot still
        still_action = self._get_still_action()
        yield still_action

        # We manually step the simulator to make the robot settle down
        for i in range(10):
            self.robot.keep_still()  # angel velocity
            yield still_action  # position

        if object_name == 'cabinet.n.01_1':
            self.is_close_to_cabinet = True
        else:
            self.is_close_to_cabinet = False

        current_location = copy.deepcopy(np.array([obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw]))

        distance_thres = 1e-1
        if np.sum(np.abs(current_location - self.view_location['cabinet.n.01_1'])) < distance_thres:
            self.agent_location = 0
        elif np.sum(np.abs(current_location - self.view_location['pumpkin.n.02_1'])) < distance_thres:
            self.agent_location = 1
        elif np.sum(np.abs(current_location - self.view_location['pumpkin.n.02_2'])) < distance_thres:
            self.agent_location = 2
        else:
            self.agent_location = 3
        print("Finished navigating to object: {}, self.agent_location: {}".format(object_name, self.agent_location))

    def _pick(self, object_name):
        obj = self._get_obj_in_hand()
        if not obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot pick object {}, something already in hand".format(object_name),
                {"object_to_navigate": object_name},
            )

        logger.info("Picking object {}".format(object_name))
        # Don't do anything if the object is already grasped.
        obj = self.task_obj_list[object_name]
        robot_is_grasping = self.robot.is_grasping(candidate_obj=None)

        robot_is_grasping_obj = self.robot.is_grasping(candidate_obj=obj)
        if robot_is_grasping == IsGraspingState.TRUE:
            if robot_is_grasping_obj == IsGraspingState.TRUE:
                logger.warning("Robot already grasping the desired object")
                yield np.zeros(self.robot.action_dim)
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": object_name},
                )

        # Process the offset from object frame to world frame
        params = skill_object_offset_params[B1KActionPrimitive.PICK][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])
        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        # If we are skipping arm planning, we do not need to plan for a pre-grasp, only the feasibility of the grasping
        # location
        pre_grasping_distance = 0.1 if not self.skip_arm_planning else 0.0
        plan_full_pre_grasp_motion = not self.skip_arm_planning
        picking_direction = self.default_direction

        # If the params have been prepared for any robot, we use the finger size as offset
        if len(params) > 3:
            logger.warning(
                "The number of params indicate that this picking position was made robot-agnostic."
                "Adding finger offset."
            )
            finger_size = self.robot.finger_lengths[self.arm]
            pick_place_pos -= picking_direction * finger_size

        pre_pick_path, interaction_pick_path = self.planner.plan_ee_pick(
            pick_place_pos,
            grasping_direction=picking_direction,
            pre_grasping_distance=pre_grasping_distance,
            plan_full_pre_grasp_motion=plan_full_pre_grasp_motion,
        )
        print('pre_pick_path: {}, interaction_pick_path: {}, pre_grasping_distance: {}'.format(pre_pick_path, interaction_pick_path, pre_grasping_distance))  # [] None 0.1
        # If the pre-pick path planning of the interaction path planning failed, we raise an error
        if (
            pre_pick_path is None
            or len(pre_pick_path) == 0
            or interaction_pick_path is None
            or (len(interaction_pick_path) == 0 and pre_grasping_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pick object",
                {"object_to_pick": object_name},
            )
        # If we continue is because we found both paths and we will executed them

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pick path")
        self.planner.visualize_arm_path(pre_pick_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()

        # If we are requesting some interaction path
        if pre_grasping_distance != 0:
            # Then, execute the interaction_pick_path stopping if there is a contact
            logger.info("Executing interaction-pick path")
            yield from self._execute_ee_path(interaction_pick_path, stop_on_contact=True)

        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()

        for i in range(10):
            ig.sim.step()
        if pre_grasping_distance != 0:
            logger.info("Executing retracting path")
            yield from self._execute_ee_path(
                interaction_pick_path, stop_on_contact=False, reverse_path=True, while_grasping=True
            )
            for i in range(5):
                ig.sim.step()
        if plan_full_pre_grasp_motion:
            self.planner.visualize_arm_path(
                pre_pick_path, arm=self.arm, reverse_path=True, grasped_obj=obj, keep_last_location=True
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj=obj,
            )

        still_action = self._get_still_action()
        for i in range(5):
            self.robot.keep_still()
            yield still_action
        print("Pick action completed")


    def _place(self, object_name):
        logger.info("Placing on object {}".format(object_name))
        obj = self._get_obj_in_hand()
        if not obj is None:

            # Process the offset from object frame to world frame
            params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]

            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])
            place_pos = copy.deepcopy(obj_pos)
            place_pos[0] += vector[0]
            place_pos[1] += vector[1]
            place_pos[2] += vector[2]

            # If we are skipping arm planning, we do not need to plan for a pre-place, only the feasibility of the grasping
            # location
            plan_full_pre_place_motion = not self.skip_arm_planning  # True
            pre_place_path, pre_place_above_path = self.planner.plan_ee_place(
                place_pos,
                plan_full_pre_place_motion=plan_full_pre_place_motion,
            )

            # If the pre-place path planning of the interaction path planning failed, we raise an error
            if (
                    pre_place_path is None
                    or len(pre_place_path) == 0
                    or pre_place_above_path is None
                    or (len(pre_place_above_path) == 0)
            ):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No arm path found to push object",
                    {"object_to_pick": object_name},
                )

            # If we continue is because we found both paths and we will executed them
            # First, teleport the robot to the beginning of the pre-pick path

            self.planner.visualize_arm_path(pre_place_path, arm=self.arm, keep_last_location=True)
            yield self._get_still_action()

            self.planner.visualize_arm_path(pre_place_above_path, arm=self.arm, keep_last_location=True)

            yield from self._execute_ungrasp()
            still_action = self._get_still_action()
            for i in range(10):
                self.robot.keep_still()
                yield still_action

            self.planner.visualize_arm_path(
                pre_place_above_path, arm=self.arm,
                reverse_path=True, keep_last_location=True
            )

            self.planner.visualize_arm_path(
                [self.robot.tucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )

        still_action = self._get_still_action()
        for i in range(5):
            self.robot.keep_still()
            yield still_action

        logger.info("Place action completed")

    def _toggle(self, object_name):
        logger.info("Toggling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.TOGGLE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        toggle_pos = copy.deepcopy(obj_pos)
        toggle_pos[0] += vector[0]
        toggle_pos[1] += vector[1]
        toggle_pos[2] += vector[2]

        pre_toggling_distance = 0.0
        plan_full_pre_toggle_motion = not self.skip_arm_planning

        pre_toggle_path, toggle_interaction_path = self.planner.plan_ee_toggle(
            toggle_pos,
            -np.array(self.default_direction),
            pre_toggling_distance=pre_toggling_distance,
            plan_full_pre_toggle_motion=plan_full_pre_toggle_motion,
        )

        if (
            pre_toggle_path is None
            or len(pre_toggle_path) == 0
            or toggle_interaction_path is None
            or (len(toggle_interaction_path) == 0 and pre_toggling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to toggle object",
                {"object_to_toggle": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-toggle path")
        self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing interaction-toggle path")
        yield from self._execute_ee_path(toggle_interaction_path, stop_on_contact=True)

        logger.info("Executing retracting path")
        if plan_full_pre_toggle_motion:
            self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, reverse_path=True, keep_last_location=True)
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()

        logger.info("Toggle action completed")

    def _pull(self, object_name):
        if not self.is_close_to_cabinet:
            logger.info("Cannot do magic pull, not close to the cabinet")
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot do magic pull, not close to the cabinet",
                {"object_to_navigate": object_name},
            )
        logger.info("Pushing object {}".format(object_name))

        jnt = self.task_obj_list[object_name].joints['joint_0']

        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        jnt.set_pos(max_pos - 0.1, target=False)

        still_action = self._get_still_action()
        for i in range(5):
            yield still_action
            ig.sim.step()

    def _pull_open(self, object_name):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        pre_pulling_distance = 0.10
        pulling_distance = 0.30

        plan_full_pre_pull_motion = not self.skip_arm_planning
        plan_ee_pull_time = time.time()
        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pick_place_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=4,
        )

        if (
            pre_pull_path is None
            or len(pre_pull_path) == 0
            or approach_interaction_path is None
            or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
            or pull_interaction_path is None
            or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object",
                {"object_to_pull": object_name},
            )

        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        yield from self._execute_grasp()
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        yield from self._execute_ungrasp()
        yield self._get_still_action()


    def _push(self, object_name):
        logger.info("Pushing object {}".format(object_name))

        jnt = self.task_obj_list[object_name].joints['joint_0']
        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit
        jnt.set_pos(min_pos, target=False)
        still_action = self._get_still_action()
        for i in range(5):
            yield still_action
            ig.sim.step()

    def _push_open(self, object_name):
        obj = self._get_obj_in_hand()
        if not obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Cannot push_open object {}, something already in hand".format(object_name),
                {"object_to_navigate": object_name},
            )

        jnt = self.task_obj_list[object_name].joints['joint_1']
        min_pos, max_pos = jnt.lower_limit, jnt.upper_limit

        robot_position = self.robot.get_position()
        robot_pos_before_cabinet = np.array([7.74685502e-01, -2.19805408e+00, -7.16727482e-06])
        if np.linalg.norm(robot_position - robot_pos_before_cabinet) > 1e-1:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Robot is not in the front of the cabinet",
                {"push open": object_name},
            )
        else:
            if self.skip_arm_planning:
                jnt.set_pos(max_pos - 0.33, target=False)
                still_action = self._get_still_action()
                for i in range(5):
                    yield still_action
                    ig.sim.step()
            else:
                jnt.set_pos(min_pos + (max_pos - min_pos) * 0.2, target=False)
                still_action = self._get_still_action()
                for i in range(5):
                    yield still_action
                    ig.sim.step()

                # Process the offset from object frame to world frame
                params = skill_object_offset_params[B1KActionPrimitive.PUSH_OPEN][object_name]

                obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
                obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
                mat = quat2mat(obj_rot_XYZW)
                vector = mat @ np.array(params[:3])
                push_pos = copy.deepcopy(obj_pos)
                push_pos[0] += vector[0]
                push_pos[1] += vector[1]
                push_pos[2] += vector[2]

                plan_full_pre_push_motion = True
                pushing_direction = np.array(params[3:])

                pre_push_path, pre_insert_path, insert_path, interaction_push_path = self.planner.plan_ee_push(
                    push_pos,
                    pushing_direction,
                    plan_full_pre_push_motion=plan_full_pre_push_motion,
                    pushing_steps=10,
                )

                # If the pre-pick path planning of the interaction path planning failed, we raise an error
                if (
                    pre_push_path is None
                    or len(pre_push_path) == 0
                    or interaction_push_path is None
                    or (len(interaction_push_path) == 0)
                ):
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PLANNING_ERROR,
                        "No arm path found to push object",
                        {"object_to_pick": object_name},
                    )
                # If we continue is because we found both paths and we will executed them

                # First, teleport the robot to the beginning of the pre-pick path
                logger.info("Visualizing pre-push path")
                self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
                yield self._get_still_action()

                logger.info("Moving above the drawer")
                self.planner.visualize_arm_path(
                    pre_insert_path, arm=self.arm, keep_last_location=True
                )

                logger.info("Inserting the ee")
                yield from self._execute_ee_path(insert_path)

                logger.info("Executing interaction-push path")
                self.planner.visualize_arm_path(
                    interaction_push_path, arm=self.arm, keep_last_location=True
                )

                logger.info("Executing retracting path")
                self.planner.visualize_arm_path(
                    pre_push_path + pre_insert_path + insert_path + interaction_push_path, arm=self.arm, reverse_path=True, keep_last_location=True
                )

                still_action = self._get_still_action()
                for i in range(5):
                    self.robot.keep_still()
                    yield still_action  # self._get_still_action()
                    # print('yield self._get_still_action()')

                logger.info("Push-open action completed")

