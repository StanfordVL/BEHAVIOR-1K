"""
WARNING!
The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example.
It currently only works with BehaviorRobot with its JointControllers set to absolute mode.
See provided behavior_robot_mp_behavior_task.yaml config file for an example. See examples/action_primitives for
runnable examples.
"""
import inspect
import logging
import random
from enum import IntEnum
from math import ceil
import cv2
from matplotlib import pyplot as plt

import gym
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import PhysxSchema

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
# from igibson.external.pybullet_tools.utils import set_joint_position
# from omnigibson.object_states.on_floor import RoomFloor
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.object_states.utils import get_center_extent
# from igibson.objects.articulated_object import URDFObject
from omnigibson.objects.object_base import BaseObject
# from igibson.robots import BaseRobot, behavior_robot
from omnigibson.robots import BaseRobot
# from igibson.robots.behavior_robot import DEFAULT_BODY_OFFSET_FROM_FLOOR, BehaviorRobot
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    detect_robot_collision,
    detect_hand_collision
)
# from igibson.utils.grasp_planning_utils import get_grasp_poses_for_object, get_grasp_position_for_open
# from igibson.utils.utils import restoreState

import omnigibson.utils.transform_utils as T
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.grasping_planning_utils import (
    get_grasp_poses_for_object_sticky
)
from omnigibson.objects import DatasetObject
from omnigibson.controllers.controller_base import ControlType

# Fake imports
p = None
set_joint_position = None
RoomFloor = None
URDFObject = None
DEFAULT_BODY_OFFSET_FROM_FLOOR = 0.05
behavior_robot = None
BehaviorRobot = None
get_grasp_poses_for_object = None
get_grasp_position_for_open = None
plan_hand_motion_br = None
get_pose3d_hand_collision_fn = None
restoreState = None

KP_LIN_VEL = 0.3
KP_ANGLE_VEL = 0.2

MAX_STEPS_FOR_HAND_MOVE = 100
MAX_STEPS_FOR_HAND_MOVE_WHEN_OPENING = 30
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10
MAX_STEPS_FOR_WAYPOINT_NAVIGATION = 200

MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 60
MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM = 60

BIRRT_SAMPLING_CIRCLE_PROBABILITY = 0.5
HAND_SAMPLING_DOMAIN_PADDING = 1  # Allow 1m of freedom around the sampling range.
PREDICATE_SAMPLING_Z_OFFSET = 0.2
JOINT_CHECKING_RESOLUTION = np.pi / 18

GRASP_APPROACH_DISTANCE = 0.2
OPEN_GRASP_APPROACH_DISTANCE = 0.2
# HAND_DISTANCE_THRESHOLD = 0.9 * behavior_robot.HAND_DISTANCE_THRESHOLD
HAND_DISTANCE_THRESHOLD = 0.9

ACTIVITY_RELEVANT_OBJECTS_ONLY = False

DEFAULT_DIST_THRESHOLD = 0.05
DEFAULT_ANGLE_THRESHOLD = 0.05

LOW_PRECISION_DIST_THRESHOLD = 0.1
LOW_PRECISION_ANGLE_THRESHOLD = 0.2

logger = logging.getLogger(__name__)


def indented_print(msg, *args, **kwargs):
    logger.debug("  " * len(inspect.stack()) + str(msg), *args, **kwargs)


def is_close(start_pose, end_pose, angle_threshold, dist_threshold):
    start_pos, start_orn = start_pose
    start_rot = Rotation.from_quat(start_orn)

    end_pos, end_orn = end_pose
    end_rot = Rotation.from_quat(end_orn)

    diff_rot = end_rot * start_rot.inv()
    diff_pos = np.array(end_pos) - np.array(start_pos)
    indented_print(
        "Position difference to target: %s, Rotation difference: %s", np.linalg.norm(diff_pos), diff_rot.magnitude()
    )
    return diff_rot.magnitude() < angle_threshold, np.linalg.norm(diff_pos) < dist_threshold, diff_rot.as_euler('xyz')[2], np.linalg.norm(diff_pos)


class UndoableContext(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.state = og.sim.dump_state(serialized=False)
        og.sim._physics_context.set_gravity(value=0.0)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
            obj.keep_still()

    def __exit__(self, *args):
        og.sim.load_state(self.state, serialized=False)

        og.sim._physics_context.set_gravity(value=-9.81)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(True)



class StarterSemanticActionPrimitive(IntEnum):
    GRASP = 0
    PLACE_ON_TOP = 1
    PLACE_INSIDE = 2
    OPEN = 3
    CLOSE = 4
    NAVIGATE_TO = 5  # For mostly debugging purposes.


class StarterSemanticActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, task, scene, robot):
        logger.warning(
            "The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example. "
            "It currently only works with BehaviorRobot with its JointControllers set to absolute mode. "
            "See provided behavior_robot_mp_behavior_task.yaml config file for an example. "
            "See examples/action_primitives for runnable examples."
        )
        super().__init__(task, scene, robot)
        self.controller_functions = {
            StarterSemanticActionPrimitive.GRASP: self.grasp,
            StarterSemanticActionPrimitive.PLACE_ON_TOP: self.place_on_top,
            StarterSemanticActionPrimitive.PLACE_INSIDE: self.place_inside,
            StarterSemanticActionPrimitive.OPEN: self.open,
            StarterSemanticActionPrimitive.CLOSE: self.close,
            StarterSemanticActionPrimitive.NAVIGATE_TO: self._navigate_to_obj,
        }
        self.arm = self.robot.default_arm

    def get_action_space(self):
        if ACTIVITY_RELEVANT_OBJECTS_ONLY:
            assert isinstance(self.task, BehaviorTask), "Activity relevant objects can only be used for BEHAVIOR tasks."
            self.addressable_objects = [
                item
                for item in self.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
            ]
        else:
            self.addressable_objects = set(self.scene.objects_by_name.values())
            if isinstance(self.task, BehaviorTask):
                self.addressable_objects.update(self.task.object_scope.values())
            self.addressable_objects = list(self.addressable_objects)

        # Filter out the robots.
        self.addressable_objects = [obj for obj in self.addressable_objects if not isinstance(obj, BaseRobot)]

        self.num_objects = len(self.addressable_objects)
        return gym.spaces.Tuple(
            [gym.spaces.Discrete(self.num_objects), gym.spaces.Discrete(len(StarterSemanticActionPrimitive))]
        )

    def get_action_from_primitive_and_object(self, primitive: StarterSemanticActionPrimitive, obj: BaseObject):
        assert obj in self.addressable_objects
        primitive_int = int(primitive)
        return primitive_int, self.addressable_objects.index(obj)

    def _get_obj_in_hand(self):
        obj_in_hand = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        return obj_in_hand

    def apply(self, action):
        # Decompose the tuple
        action_idx, obj_idx = action

        # Find the target object.
        target_obj = self.addressable_objects[obj_idx]

        # Find the appropriate action generator.
        action = StarterSemanticActionPrimitive(action_idx)
        return self.controller_functions[action](target_obj)

    def open(self, obj):
        yield from self._open_or_close(obj, True)

    def close(self, obj):
        yield from self._open_or_close(obj, False)

    def _open_or_close(self, obj, should_open):
        hand_collision_fn = get_pose3d_hand_collision_fn(
            self.robot, None, self._get_collision_body_ids(include_robot=True)
        )

        # Open the hand first
        yield from self._execute_release()

        # Don't do anything if the object is already closed and we're trying to close.
        if not should_open and not obj.states[object_states.Open].get_value():
            return

        grasp_data = get_grasp_position_for_open(self.robot, obj, should_open)
        if grasp_data is None:
            if should_open and obj.states[object_states.Open].get_value():
                # It's already open so we're good
                return
            else:
                # We were trying to do something but didn't have the data.
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample grasp position for object.",
                    {"object": obj},
                )

        grasp_pose, target_poses, object_direction, joint_info, grasp_required = grasp_data
        with UndoableContext(self.robot):
            if hand_collision_fn(grasp_pose):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Rejecting grasp pose due to collision.",
                    {"object": obj, "grasp_pose": grasp_pose},
                )

        # Prepare data for the approach later.
        approach_pos = grasp_pose[0] + object_direction * OPEN_GRASP_APPROACH_DISTANCE
        approach_pose = (approach_pos, grasp_pose[1])

        # If the grasp pose is too far, navigate
        [bid] = obj.get_body_ids()  # TODO: Fix this!
        check_joint = (bid, joint_info)
        yield from self._navigate_if_needed(obj, pos_on_obj=approach_pos, check_joint=check_joint)
        yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0], check_joint=check_joint)

        yield from self._move_hand(grasp_pose)

        # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
        # It's okay if we can't go all the way because we run into the object.
        indented_print("Performing grasp approach for open.")

        try:
            yield from self._move_hand_direct(approach_pose, ignore_failure=True, stop_on_contact=True)
        except ActionPrimitiveError:
            # An error will be raised when contact fails. If this happens, let's retreat back to the grasp pose.
            yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
            raise

        if grasp_required:
            try:
                yield from self._execute_grasp()
            except ActionPrimitiveError:
                # Retreat back to the grasp pose.
                yield from self._execute_release()
                yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
                raise

        for target_pose in target_poses:
            yield from self._move_hand_direct(
                target_pose, ignore_failure=True, max_steps_for_hand_move=MAX_STEPS_FOR_HAND_MOVE_WHEN_OPENING
            )

        # Moving to target pose often fails. Let's get the hand to apply the correct actions for its current pos
        # This prevents the hand from jerking into its desired position when we do a release.
        yield from self._move_hand_direct(
            self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True
        )

        yield from self._execute_release()
        yield from self._reset_hand()

        if obj.states[object_states.Open].get_value() == should_open:
            return

    def grasp(self, obj):
        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": obj, "object_in_hand": obj_in_hand},
                )
            
        if self._get_obj_in_hand() != obj:
            # Open the hand first
            yield from self._execute_release()

            # Allow grasping from suboptimal extents if we've tried enough times.
            force_allow_any_extent = np.random.rand() < 0.5
            grasp_poses = get_grasp_poses_for_object_sticky(obj, force_allow_any_extent=force_allow_any_extent)
            grasp_pose, object_direction = random.choice(grasp_poses)
            # with UndoableContext():
            #     joint_pos, control_idx = self._convert_cartesian_to_joint_space(grasp_pose)
            #     if detect_hand_collision(self.robot, joint_pos, control_idx):
            #         raise ActionPrimitiveError(
            #             ActionPrimitiveError.Reason.SAMPLING_ERROR,
            #             "Rejecting grasp pose candidate due to collision",
            #             {"grasp_pose": grasp_pose},  # TODO: Add more info about collision.
            #         )
            # Prepare data for the approach later.
            approach_pos = grasp_pose[0] + object_direction * GRASP_APPROACH_DISTANCE
            approach_pose = (approach_pos, grasp_pose[1])

            # If the grasp pose is too far, navigate.
            # yield from self._navigate_if_needed(obj, pos_on_obj=approach_pos)
            # yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0])
            yield from self._move_hand(grasp_pose)

            # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            # It's okay if we can't go all the way because we run into the object.
            indented_print("Performing grasp approach.")
            try:
                yield from self._move_hand_direct_cartesian(approach_pose, stop_on_contact=True)
            except ActionPrimitiveError:
                # An error will be raised when contact fails. If this happens, let's retry.
                # Retreat back to the grasp pose.
                yield from self._move_hand_direct_cartesian(grasp_pose)
                raise

            indented_print("Grasping.")
            try:
                yield from self._execute_grasp()
            except ActionPrimitiveError:
                # Retreat back to the grasp pose.
                yield from self._move_hand_direct_cartesian(grasp_pose)
                raise

        indented_print("Moving hand back to neutral position.")
        yield from self._reset_hand()
        # yield from self._execute_release()

        if self._get_obj_in_hand() == obj:
            return

    def place_on_top(self, obj):
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def place_inside(self, obj):
        yield from self._place_with_predicate(obj, object_states.Inside)

    def toggle_on(self, obj):
        yield from self._toggle(obj, True)

    def toggle_off(self, obj):
        yield from self._toggle(obj, False)

    def _toggle(self, obj, value):
        if obj.states[object_states.ToggledOn].get_value() == value:
            return

        # Put the hand in the toggle marker.
        toggle_state = obj.states[object_states.ToggledOn]
        toggle_position = toggle_state.get_link_position()
        yield from self._navigate_if_needed(obj, toggle_position)

        hand_orientation = self.robot.eef_links[self.arm].get_orientation()  # Just keep the current hand orientation.
        desired_hand_pose = (toggle_position, hand_orientation)

        yield from self._move_hand_direct(desired_hand_pose, ignore_failure=True)

        # Put hand back where it was.
        yield from self._reset_hand()

        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR, "Failed to toggle object.", {"object": object}
            )

    def _place_with_predicate(self, obj, predicate):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "Cannot place object if not holding one."
            )
        
        # yield from self._execute_release()
        obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
        # obj_in_hand.set_position([1.0, 1.0, 0.5])
        # obj_pose = ([1.0, 1.0, 0.7], [0.0, 0.0, 0.0, 1.0])
        # print("obj_pose")
        # print(obj_pose)
        # print(obj_in_hand.get_position())
        # while True:
        #     print(obj_in_hand.get_position())
        #     og.sim.step()
        # return
        hand_pose = self._get_hand_pose_for_object_pose(obj_pose)
        # print("hand pose")
        # print(hand_pose)
        print("navigating")
        yield from self._navigate_if_needed(obj, pos_on_obj=hand_pose[0])
        print("done navigating")
        # with UndoableContext():
        #     obj_in_hand.set_position_orientation(*hand_pose)
        #     for i in range(1000):
        #         og.sim.step()
        # pose = [1.17965, 0.38005, 1.8925]
        # robot_pose = self._get_robot_pose_from_2d_pose((pose[0], pose[1], pose[2]))
        # self.robot.set_position_orientation(robot_pose[0], robot_pose[1])
        # og.sim.step()
        # yield from self._navigate_to_pose(pose)
        print(self.robot.get_position_orientation())
        print(hand_pose)
        yield from self._move_hand(hand_pose)
        yield from self._execute_release()
        yield from self._reset_hand()

        if obj_in_hand.states[predicate].get_value(obj):
            return

    def _convert_cartesian_to_joint_space(self, target_pose):
        joint_pos, control_idx = self._ik_solver_cartesian_to_joint_space(target_pose)
        if joint_pos is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find joint positions for target pose",
                {"target_pose": target_pose},
            )
        return joint_pos, control_idx
    
    def _target_in_reach_of_robot(self, target_pose):
        joint_pos, _ = self._ik_solver_cartesian_to_joint_space(target_pose)
        # print(joint_pos)
        return False if joint_pos is None else True

    def _ik_solver_cartesian_to_joint_space(self, target_pose):
        relative_target_pose = self._get_pose_in_robot_frame(target_pose)
        # print(relative_target_pose)
        # print("----")
        control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])
        ik_solver = IKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[self.arm],
            robot_urdf_path=self.robot.urdf_path,
            default_joint_pos=self.robot.get_joint_positions()[control_idx],
            eef_name=self.robot.eef_link_names[self.arm],
        )
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=relative_target_pose[0],
            target_quat=relative_target_pose[1],
            max_iterations=100,
        )
        # Check the joint values returned by IKSolver are within joint limits
        joints = np.array([joint for joint in self.robot.joints.values()])
        arm_joints = joints[control_idx]
        for i, joint in enumerate(arm_joints):
            if joint_pos[i] < joint.lower_limit or joint_pos[i] > joint.upper_limit:
                return None, control_idx
            
        return joint_pos, control_idx

    def _move_hand(self, target_pose):
        joint_pos, control_idx = self._convert_cartesian_to_joint_space(target_pose)

        with UndoableContext():
            plan = plan_arm_motion(
                robot=self.robot,
                obj_in_hand=self._get_obj_in_hand(),
                end_conf=joint_pos
            )

        if plan is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not make a hand motion plan.",
                {"target_pose": target_pose},
            )

        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, joint_pos in enumerate(plan):
            indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
            yield from self._move_hand_direct_joint(joint_pos, control_idx)

    def _move_hand_direct_joint(self, joint_pos, control_idx, stop_on_contact=False):
        action = self._empty_action()
        controller_name = "arm_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = joint_pos

        while True:
            current_joint_pos = self.robot.get_joint_positions()[control_idx]
            diff_joint_pos = np.absolute(np.array(current_joint_pos) - np.array(joint_pos))
            if max(diff_joint_pos) < 0.005:
                return
            if stop_on_contact and detect_robot_collision(self.robot):
                return
            yield action

    def _move_hand_direct_cartesian(self, target_pose, **kwargs):
        joint_pos, control_idx = self._convert_cartesian_to_joint_space(target_pose)
        yield from self._move_hand_direct_joint(joint_pos, control_idx, **kwargs)

    def _execute_grasp(self):
        action = self._empty_action()
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = -1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield self._empty_action()

        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )

    def _execute_release(self):
        action = self._empty_action()
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = 1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # Otherwise, keep applying the action!
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield self._empty_action()

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object still detected in hand after executing release.",
                {"object_in_hand": self._get_obj_in_hand()},
            )
        
    def _empty_action(self):
        action = np.zeros(self.robot.action_dim)
        for name, controller in self.robot._controllers.items():
            joint_idx = controller.dof_idx
            action_idx = self.robot.controller_action_idx[name]
            if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
                action[action_idx] = self.robot.get_joint_positions()[joint_idx]

        return action

    def _reset_hand(self):
        control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])
        default_pose = self.robot.tucked_default_joint_pos[control_idx]
        yield from self._move_hand_direct_joint(default_pose, control_idx)

    def _navigate_to_pose(self, pose_2d):
        with UndoableContext():
            obj_in_hand = self._get_obj_in_hand()
            plan = plan_base_motion(
                robot=self.robot,
                obj_in_hand=obj_in_hand,
                end_conf=pose_2d,
            )

        if plan is None:
            # TODO: Would be great to produce a more informative error.
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, "Could not make a navigation plan.")

        self.draw_plan(plan)
        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, pose_2d in enumerate(plan):
            indented_print("Executing navigation plan step %d/%d", i + 1, len(plan))
            low_precision = True if i < len(plan) - 1 else False
            yield from self._navigate_to_pose_direct(pose_2d, low_precision=low_precision)

    def draw_plan(self, plan):
        SEARCHED = []
        trav_map = og.sim.scene._trav_map
        for q in plan:
            # The below code is useful for plotting the RRT tree.
            SEARCHED.append(np.flip(trav_map.world_to_map((q[0], q[1]))))

            fig = plt.figure()
            plt.imshow(trav_map.floor_map[0])
            plt.scatter(*zip(*SEARCHED), 5)
            fig.canvas.draw()

            # Convert the canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # Convert to BGR for cv2-based viewing.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("SceneGraph", img)
            cv2.waitKey(1)

    def _navigate_if_needed(self, obj, pos_on_obj=None, **kwargs):
        if pos_on_obj is not None:
            pose_for_obj = (pos_on_obj, [0.0, 0.0, 0.0, 1.0])
            if self._target_in_reach_of_robot(pose_for_obj):
                # No need to navigate.
                return
        elif self._target_in_reach_of_robot(obj.get_position_orientation()):
            return

        yield from self._navigate_to_obj(obj, pos_on_obj=pos_on_obj, **kwargs)

    def _navigate_to_obj(self, obj, pos_on_obj=None, **kwargs):
        # if isinstance(obj, RoomFloor):
        #     # TODO(lowprio-MP): Pos-on-obj for the room navigation?
        #     pose = self._sample_pose_in_room(obj.room_instance)
        # else:
        #     pose = self._sample_pose_near_object(obj, pos_on_obj=pos_on_obj, **kwargs)
        # pos_on_obj = [1.1077145,  0.96007216, 0.41105416]

        pose = self._sample_pose_near_object(obj, pos_on_obj=pos_on_obj, **kwargs)
        # # pose = [0.98078451, 0.61068289, 1.01672635]
        # print(self._test_pose(pose, np.array(pos_on_obj)))
        # pose_2d = self._get_robot_pose_from_2d_pose(pose)
        # self.robot.set_position_orientation(*pose_2d)
        # og.sim.step()
        # print(detect_robot_collision(self.robot))
        
        yield from self._navigate_to_pose(pose)

    def _navigate_to_pose_direct(self, pose_2d, low_precision=False):
        pose = self._get_robot_pose_from_2d_pose(pose_2d)

        dist_threshold = LOW_PRECISION_DIST_THRESHOLD if low_precision else DEFAULT_DIST_THRESHOLD
        angle_threshold = LOW_PRECISION_ANGLE_THRESHOLD if low_precision else DEFAULT_ANGLE_THRESHOLD
        arrived_at_pos = False
        while True:
            body_target_pose = self._get_pose_in_robot_frame(pose)

            # Accumulate the actions in the correct order.
            action = self._empty_action()

            is_angle_close, is_dist_close, angle, dist = is_close(([0, 0, 0], [0, 0, 0, 1]), body_target_pose, angle_threshold, dist_threshold)
            if is_dist_close: arrived_at_pos = True

            angle_to_waypoint = T.vecs2axisangle([1, 0, 0], [body_target_pose[0][0], body_target_pose[0][1], 0.0])[2]
            # angle_to_target_pose = T.quat2euler(body_target_pose[1])[2]

            lin_vel = 0.0
            ang_vel = 0.0

            if arrived_at_pos:
                direction = -1.0 if angle < 0.0 else 1.0
                ang_vel = KP_ANGLE_VEL * abs(angle) * direction
                ang_vel = 0.2 * direction
            else:
                if abs(angle_to_waypoint) > DEFAULT_ANGLE_THRESHOLD:
                    direction = -1.0 if angle_to_waypoint < 0 else 1.0
                    ang_vel = KP_ANGLE_VEL * direction
                else:
                    lin_vel = KP_LIN_VEL

            base_action = [lin_vel, ang_vel]
            action[self.robot.controller_action_idx["base"]] = base_action

            # Return None if no action is needed.
            if is_angle_close and arrived_at_pos:
                indented_print("Move is complete.")
                return

            yield action

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.EXECUTION_ERROR,
            "Could not move robot to desired waypoint.",
            {"target_pose": pose, "current_pose": self.robot.get_position_orientation()},
        )

    def _sample_pose_near_object(self, obj, pos_on_obj=None, **kwargs):
        if pos_on_obj is None:
            pos_on_obj = self._sample_position_on_aabb_face(obj)

        pos_on_obj = np.array(pos_on_obj)
        obj_rooms = obj.in_rooms if obj.in_rooms else [self.scene._seg_map.get_room_instance_by_point(pos_on_obj[:2])]
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            distance = np.random.uniform(0.2, 1.0)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose_2d = np.array(
                [pos_on_obj[0] + distance * np.cos(yaw), pos_on_obj[1] + distance * np.sin(yaw), yaw + np.pi]
            )
            # pose_2d = [1.21189, 0.625961, 1.87304]
            # pos_on_obj = [1.0, 0.7, 0.5]

            # Check room
            if self.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                indented_print("Candidate position is in the wrong room.")
                continue

            if not self._test_pose(pose_2d, pos_on_obj=pos_on_obj, **kwargs):
                continue

            return pose_2d
 
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position near object."
        )

    @staticmethod
    def _sample_position_on_aabb_face(target_obj):
        aabb_center, aabb_extent = get_center_extent(target_obj.states)
        # We want to sample only from the side-facing faces.
        face_normal_axis = random.choice([0, 1])
        face_normal_direction = random.choice([-1, 1])
        face_center = aabb_center + np.eye(3)[face_normal_axis] * aabb_extent * face_normal_direction
        face_lateral_axis = 0 if face_normal_axis == 1 else 1
        face_lateral_half_extent = np.eye(3)[face_lateral_axis] * aabb_extent / 2
        face_vertical_half_extent = np.eye(3)[2] * aabb_extent / 2
        face_min = face_center - face_vertical_half_extent - face_lateral_half_extent
        face_max = face_center + face_vertical_half_extent + face_lateral_half_extent
        return np.random.uniform(face_min, face_max)

    def _sample_pose_in_room(self, room: str):
        # TODO(MP): Bias the sampling near the agent.
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM):
            _, pos = self.scene.get_random_point_by_room_instance(room)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose = (pos[0], pos[1], yaw)
            if self._test_pose(pose):
                return pose

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position in room.", {"room": room}
        )

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj):
        with UndoableContext():
            self.robot._release_grasp()
            pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
            result = sample_kinematics(
                pred_map[predicate],
                held_obj,
                target_obj,
                use_ray_casting_method=True,
                max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
                skip_falling=True,
                z_offset=PREDICATE_SAMPLING_Z_OFFSET,
            )

            if not result:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample position with object and predicate.",
                    {"target_object": target_obj, "held_object": held_obj, "predicate": pred_map[predicate]},
                )

            pos, orn = held_obj.get_position_orientation()
            return pos, orn

    def _test_pose(self, pose_2d, pos_on_obj=None, check_joint=None):
        with UndoableContext():
            self.robot.set_position_orientation(*self._get_robot_pose_from_2d_pose(pose_2d))
            og.sim.step(render=False)
            if pos_on_obj is not None:
                pose_on_obj = (pos_on_obj, [0.0, 0.0, 0.0, 1.0])
                if not self._target_in_reach_of_robot(pose_on_obj):
                    return False
            
            if detect_robot_collision(self.robot):
                indented_print("Candidate position failed collision test.")
                return False

            # if check_joint is not None:
            #     body_id, joint_info = check_joint

            #     # Check at different positions of the joint.
            #     joint_range = joint_info.jointUpperLimit - joint_info.jointLowerLimit
            #     turn_steps = int(ceil(abs(joint_range) / JOINT_CHECKING_RESOLUTION))
            #     for i in range(turn_steps):
            #         joint_pos = (i + 1) / turn_steps * joint_range + joint_info.jointLowerLimit
            #         set_joint_position(body_id, joint_info.jointIndex, joint_pos)

            #         if detect_robot_collision():
            #             indented_print("Candidate position failed joint-move collision test.")
            #             return False

            return True

    @staticmethod
    def _get_robot_pose_from_2d_pose(pose_2d):
        pos = np.array([pose_2d[0], pose_2d[1], DEFAULT_BODY_OFFSET_FROM_FLOOR])
        orn = T.euler2quat([0, 0, pose_2d[2]])
        return pos, orn

    def _get_pose_in_robot_frame(self, pose):
        body_pose = self.robot.get_position_orientation()
        return T.relative_pose_transform(*pose, *body_pose)

    def _get_hand_pose_for_object_pose(self, desired_pose):
        obj_in_hand = self._get_obj_in_hand()

        assert obj_in_hand is not None

        # Get the object pose & the robot hand pose
        obj_in_world = obj_in_hand.get_position_orientation()
        hand_in_world = self.robot.eef_links[self.arm].get_position_orientation()

        # Get the hand pose relative to the obj pose
        hand_in_obj = T.relative_pose_transform(*hand_in_world, *obj_in_world)

        # Now apply desired obj pose.
        # desired_hand_pose = T.pose_transform(*hand_in_obj, *desired_pose)
        desired_hand_pose = T.pose_transform(*desired_pose, *hand_in_obj)

        return desired_hand_pose