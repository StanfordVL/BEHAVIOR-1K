"""
WARNING!
The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example.
It currently only works with Tiago and R1 with their JointControllers set to absolute position mode.
See provided tiago_primitives.yaml config file for an example. See examples/action_primitives for
runnable examples.
"""

import inspect
import math

import time
import random
from typing import Any

import cv2
import gymnasium as gym
import torch as th
import numpy as np
from aenum import IntEnum, auto
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as tf

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
    BaseActionPrimitiveSet,
)
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.controllers import (
    InverseKinematicsController,
    HolonomicBaseJointController,
    DifferentialDriveController,
)
from omnigibson.macros import create_module_macros
from omnigibson.macros import macros
from omnigibson.objects.object_base import BaseObject
from omnigibson.robots import (
    R1,
    R1Pro,
    BaseRobot,
    BehaviorRobot,
    Fetch,
    Freight,
    Husky,
    Locobot,
    Stretch,
    Tiago,
    Turtlebot,
)
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.grasping_planning_utils import get_grasp_poses_for_object_sticky, get_grasp_position_for_open
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.object_state_utils import sample_cuboid_for_predicate
from omnigibson.utils.python_utils import multi_dim_linspace
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.sampling_utils import raytest
from omnigibson.utils.ui_utils import draw_line, clear_debug_drawing


m = create_module_macros(module_path=__file__)

m.KP_LIN_VEL = {
    Tiago: 0.3,
    Fetch: 0.2,
    Stretch: 0.5,
    Turtlebot: 0.3,
    Husky: 0.05,
    Freight: 0.2,
    Locobot: 1.5,
    BehaviorRobot: 0.3,
    R1: 0.3,
    R1Pro: 0.3,
}
m.KP_ANGLE_VEL = {
    Tiago: 0.2,
    Fetch: 0.1,
    Stretch: 0.7,
    Turtlebot: 0.2,
    Husky: 0.05,
    Freight: 0.1,
    Locobot: 1.0,
    BehaviorRobot: 0.2,
    R1: 0.2,
    R1Pro: 0.2,
}

m.DEFAULT_COLLISION_ACTIVATION_DISTANCE = 0.02
m.MAX_PLANNING_ATTEMPTS = 200
m.MAX_IK_FAILURES_BEFORE_RETURN = 50

m.MAX_STEPS_FOR_SETTLING = 500

m.MAX_STEPS_FOR_JOINT_MOTION = 10

m.MAX_CARTESIAN_HAND_STEP = 0.002
m.MAX_STEPS_FOR_HAND_MOVE_JOINT = 500
m.MAX_STEPS_FOR_HAND_MOVE_IK = 1000
m.MAX_STEPS_FOR_GRASP_OR_RELEASE = 250
m.MAX_STEPS_FOR_WAYPOINT_NAVIGATION = 500
m.MAX_ATTEMPTS_FOR_OPEN_CLOSE = 20

m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 200
m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_FOR_CORRECT_ROOM = 20
m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM = 60
m.MAX_ATTEMPTS_FOR_SAMPLING_PLACE_POSE = 50
m.PREDICATE_SAMPLING_Z_OFFSET = 0.02
m.BASE_POSE_SAMPLING_LOWER_BOUND = 0.4
m.BASE_POSE_SAMPLING_UPPER_BOUND = 1.0

m.GRASP_APPROACH_DISTANCE = 0.01
m.OPEN_GRASP_APPROACH_DISTANCE = 0.4

m.HAND_DIST_THRESHOLD = 0.002
m.DEFAULT_DIST_THRESHOLD = 0.01  # originally 0.005
m.DEFAULT_ANGLE_THRESHOLD = 0.03
m.LOW_PRECISION_DIST_THRESHOLD = 0.1
m.LOW_PRECISION_ANGLE_THRESHOLD = 0.2

m.JOINT_POS_DIFF_THRESHOLD = 0.01  # originally 0.005
m.LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD = 0.05
m.JOINT_CONTROL_MIN_ACTION = 0.0
m.MAX_ALLOWED_JOINT_ERROR_FOR_LINEAR_MOTION = math.radians(45)
m.TIME_BEFORE_JOINT_STUCK_CHECK = 1.0

m.MAX_EYES_SAMPLING_ANGLE = math.radians(30)

log = create_module_logger(module_name=__name__)


def indented_print(msg, *args, **kwargs):
    print("  " * len(inspect.stack()) + str(msg), *args, **kwargs)


class StarterSemanticActionPrimitiveSet(IntEnum):
    _init_ = "value __doc__"
    GRASP = auto(), "Grasp an object"
    PLACE_ON_TOP = auto(), "Place the currently grasped object on top of another object"
    PLACE_INSIDE = auto(), "Place the currently grasped object inside another object"
    OPEN = auto(), "Open an object"
    CLOSE = auto(), "Close an object"
    NAVIGATE_TO = auto(), "Navigate to an object (mostly for debugging purposes - other primitives also navigate first)"
    RELEASE = (
        auto(),
        "Release an object, letting it fall to the ground. You can then grasp it again, as a way of reorienting your grasp of the object.",
    )
    TOGGLE_ON = auto(), "Toggle an object on"
    TOGGLE_OFF = auto(), "Toggle an object off"


# (TODO) execution failure: overwrite_head_action might move the head that causes self-collision with curobo-generated motion plans (at the time of curobo planning, it assumes the head doesn't move).
# (TODO) R1 improvement: create a new CuRoboEmbodimentSelection type that only allows for arm movement, and not torso movement.


class StarterSemanticActionPrimitives(BaseActionPrimitiveSet):
    def __init__(
        self,
        env,
        robot,
        enable_head_tracking=True,
        always_track_eef=False,
        task_relevant_objects_only=False,
        curobo_batch_size=4,
        curobo_use_cuda_graph=True,
        curobo_embodiment_types=None,
        debug_visual_marker=None,
        skip_curobo_initilization=False,
        use_base_pose_hack=False,
        real_robot_mode=False,
    ):
        """
        Initializes a StarterSemanticActionPrimitives generator.

        Args:
            env (Environment): The environment that the primitives will run on.
            robot (BaseRobot): The robot that the primitives will run on.
            enable_head_tracking (bool): Whether to enable head tracking. Defaults to True.
            always_track_eef (bool, optional): Whether to always track the end effector, as opposed
                to switching between target object and end effector based on context. Defaults to False.
            task_relevant_objects_only (bool): Whether to only consider objects relevant to the task
                when computing the action space. Defaults to False.
            curobo_batch_size (int): The batch size for curobo motion planning and collision checking. Defaults to 3.
            debug_visual_marker (PrimitiveObject): The object to use for debug visual markers. Defaults to None.
            skip_curobo_initilization (bool): Whether to skip curobo initialization. Defaults to False.
        """
        log.warning(
            "The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example. "
            "It currently only works with Tiago and R1 with their HolonomicBaseJointController/JointControllers set to absolute position mode."
        )
        super().__init__(env, robot)
        self.controller_functions = {
            StarterSemanticActionPrimitiveSet.GRASP: self._grasp,
            StarterSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
            StarterSemanticActionPrimitiveSet.PLACE_INSIDE: self._place_inside,
            StarterSemanticActionPrimitiveSet.OPEN: self._open,
            StarterSemanticActionPrimitiveSet.CLOSE: self._close,
            StarterSemanticActionPrimitiveSet.NAVIGATE_TO: self._navigate_to_obj,
            StarterSemanticActionPrimitiveSet.RELEASE: self._execute_release,
            StarterSemanticActionPrimitiveSet.TOGGLE_ON: self._toggle_on,
            StarterSemanticActionPrimitiveSet.TOGGLE_OFF: self._toggle_off,
        }

        if isinstance(env.scene, og.scenes.interactive_traversable_scene.InteractiveTraversableScene):
            scene_model = env.scene.scene_model.lower()
        else:
            scene_model = "empty"

        self._motion_generator = (
            None
            if skip_curobo_initilization
            else CuRoboMotionGenerator(
                robot=self.robot,
                batch_size=curobo_batch_size,
                use_cuda_graph=curobo_use_cuda_graph,
                collision_activation_distance=m.DEFAULT_COLLISION_ACTIVATION_DISTANCE,
                scene_model=scene_model,
                embodiment_types=curobo_embodiment_types,
                use_eyes_targets=enable_head_tracking,
                restrict_torso_joint_limits=real_robot_mode,
            )
        )

        self._task_relevant_objects_only = task_relevant_objects_only

        self._enable_head_tracking = enable_head_tracking
        self._always_track_eef = always_track_eef
        self._tracking_object = None

        # Store the current position of the arm as the arm target
        control_dict = self.robot.get_control_dict()
        self._arm_targets = {}
        self._reset_eef_pose = {}
        if isinstance(self.robot, ManipulationRobot):
            for arm_name in self.robot.arm_names:
                eef = f"eef_{arm_name}"
                arm = f"arm_{arm_name}"
                arm_ctrl = self.robot.controllers[arm]
                if isinstance(arm_ctrl, InverseKinematicsController):
                    pos_relative = cb.to_torch(control_dict[f"{eef}_pos_relative"])
                    quat_relative = cb.to_torch(control_dict[f"{eef}_quat_relative"])
                    quat_relative_axis_angle = T.quat2axisangle(quat_relative)
                    self._arm_targets[arm] = (pos_relative, quat_relative_axis_angle)
                else:
                    arm_target = cb.to_torch(control_dict["joint_position"])[arm_ctrl.dof_idx]
                    self._arm_targets[arm] = arm_target

                self._reset_eef_pose[arm_name] = self.robot.get_relative_eef_pose(arm_name)

        self._curobo_batch_size = curobo_batch_size
        self.debug_visual_marker = debug_visual_marker

        self.mp_err = "None"
        self.target_eyes_pose = None
        self.target_eyes_pose_arr = []
        self.attached_obj_info = {"attached_obj": None, "attached_obj_scale": None}
        self.use_base_pose_hack = use_base_pose_hack
        self.base_sampling_time = 0.0
        self.base_mp_planning_time = 0.0
        self.real_robot_mode = real_robot_mode
        # Action primitives uses assisted grasping; we ignore delayed AG here
        with macros.unlocked():
            macros.robots.manipulation_robot.GRASP_WINDOW = 0.0

    @property
    def arm(self):
        assert isinstance(self.robot, ManipulationRobot), "Cannot use arm for non-manipulation robot"
        return self.robot.default_arm

    def _postprocess_action(self, action):
        """Postprocesses action by applying head tracking."""
        if self._enable_head_tracking and isinstance(self.robot, Tiago):
            action = self._overwrite_head_action(action)
        return action

    def get_action_space(self):
        # TODO: Figure out how to implement what happens when the set of objects in scene changes.
        if self._task_relevant_objects_only:
            assert isinstance(
                self.env.task, BehaviorTask
            ), "Activity relevant objects can only be used for BEHAVIOR tasks"
            self.addressable_objects = sorted(set(self.env.task.object_scope.values()), key=lambda obj: obj.name)
        else:
            self.addressable_objects = sorted(set(self.env.scene.objects_by_name.values()), key=lambda obj: obj.name)

        # Filter out the robots.
        self.addressable_objects = [obj for obj in self.addressable_objects if not isinstance(obj, BaseRobot)]

        self.num_objects = len(self.addressable_objects)
        return gym.spaces.Tuple(
            [gym.spaces.Discrete(self.num_objects), gym.spaces.Discrete(len(StarterSemanticActionPrimitiveSet))]
        )

    def get_action_from_primitive_and_object(self, primitive: StarterSemanticActionPrimitiveSet, obj: BaseObject):
        assert obj in self.addressable_objects
        primitive_int = int(primitive)
        return primitive_int, self.addressable_objects.index(obj)

    def _get_obj_in_hand(self):
        """
        Get object in the robot's hand

        Returns:
            StatefulObject or None: Object if robot is holding something or None if it is not
        """
        obj_in_hand = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        return obj_in_hand

    def apply(self, action):
        # Decompose the tuple
        action_idx, obj_idx = action

        # Find the target object.
        target_obj = self.addressable_objects[obj_idx]

        # Find the appropriate action generator.
        action = StarterSemanticActionPrimitiveSet(action_idx)
        return self.apply_ref(action, target_obj)

    def apply_ref(self, primitive, *args: Any, attempts=5):
        """
        Yields action for robot to execute the primitive with the given arguments.

        Args:
            primitive (StarterSemanticActionPrimitiveSet): Primitive to execute
            args: Arguments for the primitive
            attempts (int): Number of attempts to make before raising an error

        Yields:
            th.tensor or None: Action array for one step for the robot to execute the primitve or None if primitive completed

        Raises:
            ActionPrimitiveError: If primitive fails to execute
        """
        ctrl = self.controller_functions[primitive]

        errors = []
        for _ in range(attempts):
            # Attempt
            success = False
            try:
                yield from ctrl(*args)
                success = True
            except ActionPrimitiveError as e:
                errors.append(e)

            try:
                # If we're not holding anything, release the hand so it doesn't stick to anything else.
                if not self._get_obj_in_hand():
                    yield from self._execute_release()
            except ActionPrimitiveError:
                pass

            try:
                # Make sure we retract the arms after every step
                yield from self._reset_robot()
            except ActionPrimitiveError:
                pass

            try:
                # Settle before returning.
                yield from self._settle_robot()
            except ActionPrimitiveError:
                pass

            # Stop on success
            if success:
                return

        raise ActionPrimitiveErrorGroup(errors)

        # # One-attempt debug version
        # ctrl = self.controller_functions[primitive]
        # yield from ctrl(*args)
        # if not self._get_obj_in_hand():
        #     yield from self._execute_release()
        # yield from self._reset_robot()
        # yield from self._settle_robot()

    def _open(self, obj):
        yield from self._open_or_close(obj, True)

    def _close(self, obj):
        yield from self._open_or_close(obj, False)

    def _open_or_close(self, obj, should_open):
        raise NotImplementedError("Open/close is not implemented correctly yet.")

        # Update the tracking to track the eef.
        self._tracking_object = self.robot

        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand().name},
            )

        # Open the hand first
        yield from self._execute_release()

        for _ in range(m.MAX_ATTEMPTS_FOR_OPEN_CLOSE):
            try:
                # TODO: This needs to be fixed. Many assumptions (None relevant joint, 3 waypoints, etc.)
                if should_open:
                    grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None)
                else:
                    grasp_data = get_grasp_position_for_open(self.robot, obj, should_open, None, num_waypoints=3)

                if grasp_data is None:
                    # We were trying to do something but didn't have the data.
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.SAMPLING_ERROR,
                        "Could not sample grasp position for target object",
                        {"target object": obj.name},
                    )

                relevant_joint, grasp_pose, target_poses, object_direction, grasp_required, pos_change = grasp_data
                if abs(pos_change) < 0.1:
                    indented_print("Yaw change is small and done,", pos_change)
                    return

                # Prepare data for the approach later.
                approach_pos = grasp_pose[0] + object_direction * m.OPEN_GRASP_APPROACH_DISTANCE
                approach_pose = (approach_pos, grasp_pose[1])

                # If the grasp pose is too far, navigate
                yield from self._navigate_if_needed(obj, eef_pose=grasp_pose)

                yield from self._move_hand(grasp_pose, stop_if_stuck=True)

                # We can pre-grasp in sticky grasping mode only for opening
                if should_open:
                    yield from self._execute_grasp()

                # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
                # It's okay if we can't go all the way because we run into the object.
                yield from self._navigate_if_needed(obj, eef_pose=approach_pose)

                if should_open:
                    yield from self._move_hand_linearly_cartesian(
                        approach_pose, ignore_failure=False, stop_on_contact=True, stop_if_stuck=True
                    )
                else:
                    yield from self._move_hand_linearly_cartesian(
                        approach_pose, ignore_failure=False, stop_if_stuck=True
                    )

                # Step once to update
                empty_action = self._empty_action()
                yield self._postprocess_action(empty_action)

                for i, target_pose in enumerate(target_poses):
                    yield from self._move_hand_linearly_cartesian(target_pose, ignore_failure=False, stop_if_stuck=True)

                # Moving to target pose often fails. This might leave the robot's motors with torques that
                # try to get to a far-away position thus applying large torques, but unable to move due to
                # the sticky grasp joint. Thus if we release the joint, the robot might suddenly launch in an
                # arbitrary direction. To avoid this, we command the hand to apply torques with its current
                # position as its target. This prevents the hand from jerking into some other position when we do a release.
                yield from self._move_hand_linearly_cartesian(
                    self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True, stop_if_stuck=True
                )

                if should_open:
                    yield from self._execute_release()
                    yield from self._move_base_backward()

            except ActionPrimitiveError as e:
                indented_print(e)
                if should_open:
                    yield from self._execute_release()
                    yield from self._move_base_backward()
                else:
                    yield from self._move_hand_backward()

        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Despite executing the planned trajectory, the object did not open or close as expected. Maybe try again",
                {"target object": obj.name, "is it currently open": obj.states[object_states.Open].get_value()},
            )

    # TODO: Figure out how to generalize out of this "backing out" behavior.
    def _move_base_backward(self, steps=5, speed=0.2):
        """
        Yields action for the robot to move base so the eef is in the target pose using the planner

        Args:
            steps (int): steps to move base
            speed (float): base speed

        Returns:
            th.tensor or None: Action array for one step for the robot to move base or None if its at the target pose
        """
        for _ in range(steps):
            action = self._empty_action()
            action[self.robot.controller_action_idx["gripper_{}".format(self.arm)]] = 1.0
            action[self.robot.base_control_idx[0]] = -speed
            yield self._postprocess_action(action)

    def _move_hand_backward(self, steps=5, speed=0.2):
        """
        Yields action for the robot to move its base backwards.

        Args:
            steps (int): steps to move eef
            speed (float): eef speed

        Returns:
            th.tensor or None: Action array for one step for the robot to move hand or None if its at the target pose
        """
        for _ in range(steps):
            action = self._empty_action()
            action[self.robot.controller_action_idx["gripper_{}".format(self.arm)]] = 1.0
            action[self.robot.controller_action_idx["arm_{}".format(self.arm)][0]] = -speed
            yield self._postprocess_action(action)

    def _move_hand_upward(self, steps=5, speed=0.1):
        """
        Yields action for the robot to move hand upward.

        Args:
            steps (int): steps to move eef
            speed (float): eef speed

        Returns:
            th.tensor or None: Action array for one step for the robot to move hand or None if its at the target pose
        """
        # TODO: Combine these movement functions.
        for _ in range(steps):
            action = self._empty_action()
            action[self.robot.controller_action_idx["gripper_{}".format(self.arm)]] = 1.0
            action[self.robot.controller_action_idx["arm_{}".format(self.arm)][2]] = speed
            yield self._postprocess_action(action)

    def _sample_grasp_pose(self, obj):
        """
        Samples an eef grasp pose for the object

        Args:
            obj (BaseObject): Object to sample grasp pose for

        Returns:
            Tuple[th.tensor, th.tensor]: Pregrasp pose and grasp pose of the robot eef in the world frame
        """
        indented_print("Sampling grasp pose")
        grasp_poses = get_grasp_poses_for_object_sticky(obj)
        grasp_pos, grasp_quat = random.choice(grasp_poses)

        # Identity quaternion for top-down grasping (x-forward, y-right, z-down)
        approach_dir = T.quat2mat(grasp_quat) @ th.tensor([0.0, 0.0, -1.0])

        avg_finger_offset = th.mean(
            th.tensor([length for length in self.robot.eef_to_fingertip_lengths[self.arm].values()])
        )
        pregrasp_offset = avg_finger_offset + m.GRASP_APPROACH_DISTANCE

        pregrasp_pos = grasp_pos - approach_dir * pregrasp_offset

        # The sampled grasp pose is robot-agnostic
        # We need to multiply by the quaternion of the robot's eef frame of top-down grasping (x-forward, y-right, z-down)
        grasp_quat = T.quat_multiply(grasp_quat, th.tensor([1.0, 0.0, 0.0, 0.0]))

        pregrasp_pose = (pregrasp_pos, grasp_quat)
        grasp_pose = (grasp_pos, grasp_quat)

        return pregrasp_pose, grasp_pose

    def _grasp(self, obj):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            StatefulObject: Object for robot to grasp

        Returns:
            th.tensor or None: Action array for one step for the robot to grasp or None if grasp completed
        """
        if obj is None or not isinstance(obj, BaseObject):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to provide an object to grasp",
                {"provided object": obj},
            )
        if self.robot.grasping_mode not in ["sticky", "assisted"]:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Grasping mode not supported",
                {"grasping mode": self.robot.grasping_mode},
            )

        # Update the tracking to track the object.
        self._tracking_object = obj

        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when your hand is already full",
                    {"target object": obj.name, "object currently in hand": obj_in_hand.name},
                )

        # Open the hand first
        indented_print("Opening hand before grasping")
        yield from self._execute_release()

        pregrasp_pose, grasp_pose = self._sample_grasp_pose(obj)

        # If the pre-grasp pose is too far, navigate.
        indented_print("Navigating to grasp pose if needed")
        yield from self._navigate_if_needed(obj, eef_pose=pregrasp_pose)

        indented_print("Moving hand to grasp pose")
        yield from self._move_hand(pregrasp_pose)

        if self.robot.grasping_mode == "sticky":
            indented_print("Sticky grasping: close gripper")
            # Close the gripper
            yield from self._execute_grasp()

            indented_print("Sticky grasping: approach")
            # Only translate in the z-axis of the goal frame (assuming z-axis points out of the gripper)
            # This is the same as requesting the end-effector to move along the approach_dir direction.
            # By default, it's NOT the z-axis of the world frame unless `project_pose_to_goal_frame=False` is set in curobo.
            # For sticky grasping, we also need to ignore the object during motion planning because the fingers are already closed.
            yield from self._move_hand(
                grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0], stop_on_ag=True, ignore_objects=[obj]
            )
        elif self.robot.grasping_mode == "assisted":
            indented_print("Assisted grasping: approach")
            # Same as above in terms of moving along the approach_dir direction, but we don't ignore the object.
            # For this approach motion, we expect the fingers to move towards and eventually "wrap" around the object without collisions.
            yield from self._move_hand(grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0])

            # Now we close the fingers to grasp the object with AG.
            indented_print("Assisted grasping: close gripper")
            yield from self._execute_grasp()

        # Step a few times to update
        yield from self._settle_robot()

        indented_print("Checking grasp")
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        indented_print("Done with grasp")

        if self._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )

    def _place_on_top(self, obj):
        """
        Yields action for the robot to navigate to the object if needed, then to place an object on it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on

        Returns:
            th.tensor or None: Action array for one step for the robot to place or None if place completed
        """
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def _place_inside(self, obj):
        """
        Yields action for the robot to navigate to the object if needed, then to place an object in it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on

        Returns:
            th.tensor or None: Action array for one step for the robot to place or None if place completed
        """
        yield from self._place_with_predicate(obj, object_states.Inside)

    def _toggle_on(self, obj):
        yield from self._toggle(obj, True)

    def _toggle_off(self, obj):
        yield from self._toggle(obj, False)

    def _toggle(self, obj, value):
        raise NotImplementedError("Toggle is not implemented correctly yet.")

        if obj.states[object_states.ToggledOn].get_value() == value:
            return

        # Put the hand in the toggle marker.
        toggle_state = obj.states[object_states.ToggledOn]
        toggle_position = toggle_state.get_link_position()
        yield from self._navigate_if_needed(obj, toggle_position)

        # Just keep the current hand orientation.
        hand_orientation = self.robot.eef_links[self.arm].get_position_orientation()[1]
        desired_hand_pose = (toggle_position, hand_orientation)

        yield from self._move_hand(desired_hand_pose)

        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {
                    "target object": obj.name,
                    "is it currently toggled on": obj.states[object_states.ToggledOn].get_value(),
                },
            )

    def _place_with_predicate(self, obj, predicate):
        """
        Yields action for the robot to navigate to the object if needed, then to place it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on
            predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

        Returns:
            th.tensor or None: Action array for one step for the robot to place or None if place completed
        """
        if obj is None or not isinstance(obj, BaseObject):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to provide an object to place the object in your hand on",
                {"provided object": obj},
            )
        # Update the tracking to track the object.
        self._tracking_object = obj

        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )

        self._motion_generator.update_obstacles()
        target_in_reach = False
        valid_navigation_pose = None
        for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_PLACE_POSE):
            # Sample one pose at a time
            obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj, world_aligned=True)
            hand_pose = self._get_hand_pose_for_object_pose(obj_pose)

            # First check if we can directly move the hand there
            # We want to plan with the fingers at their upper (open) limits to avoid collisions
            # because later we will open-loop open the gripper with _execute_release after placing.
            initial_joint_pos = self._get_joint_position_with_fingers_at_limit("upper")
            target_in_reach = self._target_in_reach_of_robot(
                hand_pose, initial_joint_pos=initial_joint_pos, skip_obstacle_update=True
            )
            if target_in_reach:
                yield from self._move_hand(hand_pose)
                break

            # If not, try to find a valid navigation pose for this hand pose
            valid_navigation_pose = self._sample_pose_near_object(
                obj, eef_pose=hand_pose, plan_with_open_gripper=True, sampling_attempts=10, skip_obstacle_update=True
            )

            if valid_navigation_pose is not None:
                yield from self._navigate_to_pose(valid_navigation_pose, skip_obstacle_update=True)
                yield from self._move_hand(hand_pose)
                break

        if not target_in_reach and valid_navigation_pose is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find a valid pose to place the object",
                {"target object": obj.name},
            )

        yield from self._execute_release()

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not release object - the object is still in your hand",
                {"object": self._get_obj_in_hand().name},
            )

        if not obj_in_hand.states[predicate].get_value(obj):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Failed to place object at the desired place (probably dropped). The object was still released, so you need to grasp it again to continue",
                {"dropped object": obj_in_hand.name, "target object": obj.name},
            )

    def _convert_cartesian_to_joint_space(self, target_pose):
        """
        Gets joint positions for the arm so eef is at the target pose

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame

        Returns:
            2-tuple
                - th.tensor or None: Joint positions to reach target pose or None if impossible to reach target pose
                - th.tensor: Indices for joints in the robot
        """
        joint_pos = self._ik_solver_cartesian_to_joint_space(target_pose)
        if joint_pos is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find joint positions for target pose. You cannot reach it. Try again for a new pose",
            )
        return joint_pos

    def sample_eyes_pos(self, source_pose, robot_pose):
        """
        Samples new origins within a cube centered around the given origin.

        :param origin: Tuple (x, y, z) representing the current origin.
        :param cube_size: Side length of the cube.
        :param num_samples: Number of samples to generate.
        :return: NumPy array of shape (num_samples, 3) containing sampled positions.
        """

        eye_pose_wrt_world = source_pose
        robot_pose_wrt_world = robot_pose
        # breakpoint()
        eye_pose_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_wrt_world)) @ T.pose2mat(eye_pose_wrt_world)
        eye_pos_wrt_robot = eye_pose_wrt_robot[:3, 3]

        # TODO (eric - datagen): specific to R1 only
        # Generate random samples within the cube range
        if self.real_robot_mode:
            x_sample = np.random.uniform(eye_pos_wrt_robot[0], eye_pos_wrt_robot[0] + 0.05)
            y_sample = np.random.uniform(eye_pos_wrt_robot[1] - 0.01, eye_pos_wrt_robot[1] + 0.01)
            z_sample = np.random.uniform(eye_pos_wrt_robot[2] - 0.025, eye_pos_wrt_robot[2])
        else:
            x_sample = np.random.uniform(eye_pos_wrt_robot[0], eye_pos_wrt_robot[0] + 0.3)
            y_sample = np.random.uniform(eye_pos_wrt_robot[1] - 0.05, eye_pos_wrt_robot[1] + 0.05)
            # TODO (arpit) Test the upper limit range (0.2) makes sense
            z_sample = np.random.uniform(eye_pos_wrt_robot[2] - 0.2, eye_pos_wrt_robot[2] + 0.1) # originally eye_pos_wrt_robot[2] - 0.4 for lower limit

        sampled_eyes_pos_wrt_robot = np.array([x_sample, y_sample, z_sample])
        sampled_eyes_pos_wrt_world = T.pose2mat(robot_pose_wrt_world) @ np.hstack([sampled_eyes_pos_wrt_robot, 1])

        return sampled_eyes_pos_wrt_world[:3]

    # TODO (eric - datagen): understand this.
    def sample_eyes_orn(
        self,
        V,
        ref_quat=th.tensor([0.406, -0.406, -0.579, 0.579]),
        num_samples=100,
        max_angle=np.radians(30),
        anisotropy=(1, 1),
    ):
        V = np.asarray(V, dtype=np.float64)
        if np.linalg.norm(V) == 0:
            raise ValueError("V must be non-zero")

        z_axis = -V / np.linalg.norm(V)  # Desired local -Z axis

        # If no reference, use identity (X = [1, 0, 0])
        if ref_quat is None:
            ref_x = np.array([1.0, 0.0, 0.0])
        else:
            ref_rot = R.from_quat(ref_quat)
            ref_x = ref_rot.apply([1.0, 0.0, 0.0])  # Local X axis in base frame

        # Project ref_x onto the plane orthogonal to z_axis
        x_axis = ref_x - np.dot(ref_x, z_axis) * z_axis
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            # ref_x was too aligned with z_axis, pick arbitrary orthogonal vector
            x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
        x_axis /= np.linalg.norm(x_axis)

        # Compute Y to complete orthonormal frame
        y_axis = np.cross(z_axis, x_axis)

        # Compose rotation matrix with columns: [x, y, z]
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

        rot = R.from_matrix(rot_matrix)
        adjusted_base_quat = rot.as_quat()

        sampled_quaternions = []
        x_scale, y_scale = anisotropy

        for _ in range(num_samples):
            # To understand this: think of a plane perpendicular to the -z-axis of the eye frame.
            # To obtain gaze that is close to the object, what we want is different vectors
            # that intersect the plane at different points within a circle. This is what apha and beta ensures.
            # beta corresponds to which direction from the origin, along the radius to sample the point (hence it is between 0 and 2pi) and
            # alpha corresponds to how far from the origin, along the chosen direction, to sample the point (hence it is between 0 and max_angle)
            alpha = np.random.uniform(0, max_angle)  # Random perturbation angle
            beta = np.random.uniform(0, 2 * np.pi)  # Direction in x-y plane

            # Perturb only in x and y to restrict roll (rotation around z)
            perturb_axis = np.array([x_scale * np.cos(beta), y_scale * np.sin(beta), 0])
            perturb_axis /= np.linalg.norm(perturb_axis)  # Normalize

            # Generate perturbation quaternion
            perturb_quat = tf.Rotation.from_rotvec(alpha * perturb_axis).as_quat()

            # Apply perturbation to adjusted base quaternion
            new_quat = tf.Rotation.from_quat(adjusted_base_quat) * tf.Rotation.from_quat(perturb_quat)
            new_quat = th.tensor(new_quat.as_quat(), dtype=th.float32)
            sampled_quaternions.append(new_quat)

        return sampled_quaternions

    def quaternion_angular_distance(self, q1, q2):
        """
        Computes the angular distance between two unit quaternions.

        :param q1: First quaternion (w, x, y, z).
        :param q2: Second quaternion (w, x, y, z).
        :return: Angular distance in radians.
        """
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure numerical stability
        return 2 * np.arccos(abs(dot_product))

    def _target_in_reach_of_robot_and_visible(self, target_pose, initial_joint_pos=None, skip_obstacle_update=False, ik_world_collision_check=False, emb_sel=CuRoboEmbodimentSelection.ARM, attach_obj=False, eyes_pose=None):
        """
        Determines whether the eef for the robot can reach the target pose in the world frame

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            skip_obstacle_update (bool): Whether to skip updating obstacles

        Returns:
            bool: Whether the default eef can reach the target pose
        """
        ignore_bodies = []
        for link in self.robot.links:
            ignore_bodies.append(self.robot.links[link].prim_path)

        if eyes_pose is None:
            # sample pos and orn for the eyes link and check if IK solution exists
            robot_joint_names = list(self.robot.joints.keys())
            cu_js = lazy.curobo.types.state.JointState(
                position=self._motion_generator._tensor_args.to_device(initial_joint_pos), joint_names=robot_joint_names
            ).get_ordered_joint_state(self._motion_generator.mg["default"].kinematics.joint_names)
            retval = self._motion_generator.mg["default"].kinematics.compute_kinematics(cu_js, link_name="eyes")
            # TODO: (eric - datagen): eye pose in robot initial pose frame
            eye_pos_for_sampled_base_pos = retval.ee_position.cpu()[0]
            eye_quat_for_sampled_base_pos = retval.ee_quaternion.cpu()[0]
            # convert quat from wxyz (curobo) to xyzw (OG)
            eye_quat_for_sampled_base_pos = th.roll(eye_quat_for_sampled_base_pos, -1)
            eye_pose_for_sampled_base_pose = (eye_pos_for_sampled_base_pos, eye_quat_for_sampled_base_pos)
            # TODO: (eric - datagen): obect pos is in world frame
            obj_pos = self._tracking_object.get_position_orientation()[0]

            robot_pose2d_for_sampled_base = th.tensor([initial_joint_pos[0], initial_joint_pos[1], initial_joint_pos[5]])
            robot_pose_for_sampled_base = self._get_robot_pose_from_2d_pose(robot_pose2d_for_sampled_base)
            # breakpoint()

            num_eyes_pos_samples = 5
            num_eyes_orn_samples = 5
            eyes_obj_min_distance = 0.2
            for i in range(num_eyes_pos_samples):
                # World frame
                sampled_eyes_pos = self.sample_eyes_pos(
                    source_pose=eye_pose_for_sampled_base_pose, robot_pose=robot_pose_for_sampled_base
                )
                eye_to_obj_vec = obj_pos - sampled_eyes_pos

                if eye_to_obj_vec.norm() < eyes_obj_min_distance:
                    print("sampled eyes pos too close to object, skipping sample")
                    continue
                
                raytest_result = raytest(sampled_eyes_pos, obj_pos, only_closest=True, ignore_bodies=ignore_bodies)
                # We don't need this!
                # if not raytest_result["hit"]:
                #     print("raytest from eyes to object does not hit, skipping sample")
                #     continue
                if raytest_result["hit"] and raytest_result["rigidBody"] not in [link.prim_path for link in self._tracking_object.links.values()]:
                    print(
                        f"raytest from eyes to object hits something else, expect {self._tracking_object.root_link.prim_path}, got {raytest_result['rigidBody']}, skipping sample"
                    )
                    # breakpoint()
                    # draw_line(sampled_eyes_pos.tolist(), obj_pos.tolist(), color=(1.0, 0.0, 0.0, 1.0), size=2.0)
                    # for _ in range(5): og.sim.step()                    
                    continue

                # draw_line(sampled_eyes_pos.tolist(), obj_pos.tolist(), color=(0.0, 1.0, 0.0, 1.0), size=2.0)
                # for _ in range(5): og.sim.step()

                eye_to_obj_vec_wrt_robot = th.linalg.inv(T.pose2mat(robot_pose_for_sampled_base))[
                    :3, :3
                ] @ eye_to_obj_vec.to(th.float32)

                # quat = eye_pose_wrt_world[1].numpy()
                sampled_eyes_orn_wrt_robot_arr = self.sample_eyes_orn(
                    eye_to_obj_vec_wrt_robot, num_samples=num_eyes_orn_samples, max_angle=m.MAX_EYES_SAMPLING_ANGLE, anisotropy=(1, 1)
                )
                # breakpoint()
                for j in range(num_eyes_orn_samples):
                    sampled_eyes_orn_wrt_robot = sampled_eyes_orn_wrt_robot_arr[j]
                    sampled_eyes_orn_wrt_world = th.matmul(
                        T.pose2mat(robot_pose_for_sampled_base)[:3, :3], T.quat2mat(sampled_eyes_orn_wrt_robot)
                    )
                    sampled_eyes_orn_wrt_world = T.mat2quat(sampled_eyes_orn_wrt_world)
                    target_pose["eyes"] = (sampled_eyes_pos.to(th.float32), sampled_eyes_orn_wrt_world.to(th.float32))
                    # print("angle: ", np.rad2deg(self.quaternion_angular_distance(eye_quat_for_sampled_base_pos, sampled_eyes_orn_wrt_world)))

                    # TODO: Currently, the batch has same target pose. Each element in a batch does run independently, so it's still useful.
                    # But, check if it's better for speed to have different target poses for each element in a batch.
                    # start = time.time()
                    # breakpoint()
                    retval = self._ik_solver_cartesian_to_joint_space(
                        target_pose,
                        initial_joint_pos=initial_joint_pos,
                        skip_obstacle_update=skip_obstacle_update,
                        ik_world_collision_check=ik_world_collision_check,
                        emb_sel=emb_sel,
                        attach_obj=attach_obj,
                    )
                    # print("eye pos orn ij", i, j)
                    # print("target pose", target_pose)
                    # print("initial joint pos", initial_joint_pos)
                    # print("ik solver", time.time() - start)
                    # print("retval: ", retval)
                    
                    # If reachibility and visibility satisfied (without collision check)
                    if retval is not None:
                        
                        # This ensures given the current joint configuration, I can find a collision-aware
                        # IK solution, using my base virtual joints and torso joints, to reach base target pose
                        # and eye target pose, while keeping my arm joints frozen (this is guaranteed because it's emb_sel=BASE)
                        # ============= Check 1 ================
                        base_target_pose = {
                            "base_link": robot_pose_for_sampled_base,
                            "eyes": target_pose["eyes"],
                        }
                        retval_post_nav = self._ik_solver_cartesian_to_joint_space(
                            base_target_pose,
                            initial_joint_pos=None,
                            skip_obstacle_update=skip_obstacle_update,
                            ik_world_collision_check=True,
                            emb_sel=CuRoboEmbodimentSelection.BASE
                        )
                        if retval_post_nav is None:
                            print("Base IK solution for next navigation phase not found, skipping sample")
                            continue

                        # ============= Check 2 ================
                        query_joint_pos_after_nav = retval_post_nav
                        # Target pose for two (first pose from replay) eefs only
                        target_pose_first_eef = dict()
                        for key, val in target_pose.items():
                            if key != "eyes":
                                target_pose_first_eef[key] = (val[0][0], val[1][0])
                        
                        # start = time.time()
                        # Remember that query_joint_pos_after_nav is (sampled base pose + BASE IK (with collision check) torso + current arm joint positions/BASE IK (with collision check) arm)
                        retval = self._ik_solver_cartesian_to_joint_space(
                            target_pose_first_eef,
                            initial_joint_pos=query_joint_pos_after_nav,
                            skip_obstacle_update=skip_obstacle_update,
                            ik_world_collision_check=True,
                            emb_sel=CuRoboEmbodimentSelection.ARM_NO_TORSO
                        )
                        if retval_post_nav is None:
                            print("Arm-no-torso IK solution for next manipulation phase not found, skipping sample")
                            continue

                        # print("------- target_pose_first_eef: ", target_pose_first_eef)
                        # print("------- query_joint_pos_after_nav: ", query_joint_pos_after_nav)
                        # constraint_satisfied = True
                        self.target_eyes_pose_arr.append(target_pose["eyes"])
                        # clear_debug_drawing()
                        return True

            self.target_eyes_pose_arr.append(None)
            # clear_debug_drawing()
            # breakpoint()
            return False

        # If an eye pose is already given, we don't need to sample and we can directly compute visibility and reachability
        else:
            # 1. Visibility check
            # Camera direction test
            obj_pos = self._tracking_object.get_position_orientation()[0]
            eyes_to_obj_vec = obj_pos - eyes_pose[0]
            eyes_z_vec = -R.from_quat(eyes_pose[1]).as_matrix()[:3, 2]
            dot_product = np.dot(eyes_to_obj_vec, eyes_z_vec)
            norm_eyes_to_obj_vec = np.linalg.norm(eyes_to_obj_vec)
            norm_eyes_z_vec = np.linalg.norm(eyes_z_vec)
            # Ensure the value passed to arccos is within [-1, 1] to avoid NaNs due to floating point errors
            cos_theta = np.clip(dot_product / (norm_eyes_to_obj_vec * norm_eyes_z_vec), -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            print("angle_rad: ", angle_rad)
            # Early return
            if angle_rad > m.MAX_EYES_SAMPLING_ANGLE:
                print("angle between eyes and object is too large, meaning that object is not in view of camera")
                return False
            
            # Raycast test
            # ignore_bodies=[self.robot.links["torso_link4"].prim_path]
            raytest_result = raytest(eyes_pose[0], obj_pos, only_closest=True, ignore_bodies=ignore_bodies)
            if raytest_result["hit"] and raytest_result["rigidBody"] != self._tracking_object.root_link.prim_path:
                print(
                    f"raytest from eyes to object hits something else, expect {self._tracking_object.root_link.prim_path}, got {raytest_result['rigidBody']}, meaning that another obect is probably blocking the camera from look at the object"
                )
                return False

            # 2. Reachability check
            retval = self._ik_solver_cartesian_to_joint_space(
                    target_pose,
                    initial_joint_pos=initial_joint_pos,
                    skip_obstacle_update=skip_obstacle_update,
                    ik_world_collision_check=ik_world_collision_check,
                    emb_sel=emb_sel,
                )
            if retval is None:
                print("IK solver failed, meaning that the object is not reachable")
                return False

            # If passed above checks, object is reachable and visible
            print("All checks passed, meaning that the object is reachable and visible")
            return True         
            

    def _target_in_reach_of_robot(self, target_pose, initial_joint_pos=None, skip_obstacle_update=False):
        """
        Determines whether the eef for the robot can reach the target pose in the world frame

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            skip_obstacle_update (bool): Whether to skip updating obstacles

        Returns:
            bool: Whether the default eef can reach the target pose
        """
        retval = self._ik_solver_cartesian_to_joint_space(
            target_pose,
            initial_joint_pos=initial_joint_pos,
            skip_obstacle_update=skip_obstacle_update,
        )
        if retval is None: 
            self.target_eyes_pose_arr.append(None)
            return False
        else: 
            robot_joint_names = list(self.robot.joints.keys())
            returned_joint_pos = retval[0]
            cu_js = lazy.curobo.types.state.JointState(position=self._motion_generator._tensor_args.to_device(returned_joint_pos), joint_names=robot_joint_names).get_ordered_joint_state(self._motion_generator.mg["default"].kinematics.joint_names)
            retval_eye_pose = self._motion_generator.mg["default"].kinematics.compute_kinematics(cu_js, link_name="eyes")
            retval_eye_pos = retval_eye_pose.ee_position.cpu()[0]
            retval_eye_quat = retval_eye_pose.ee_quaternion.cpu()[0]
            retval_eye_quat = th.roll(retval_eye_quat, -1)
            self.target_eyes_pose_arr.append((retval_eye_pos, retval_eye_quat))
            # breakpoint()
            return True


    def _manipulation_control_idx(self):
        """The appropriate manipulation control idx for the current settings."""
        return th.cat([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])

    def _ik_solver_cartesian_to_joint_space(
        self, target_pose, initial_joint_pos=None, skip_obstacle_update=False, ik_world_collision_check=False, emb_sel=CuRoboEmbodimentSelection.ARM, attach_obj=False,
    ):
        """
        Get joint positions for the arm so eef is at the target pose in the world frame

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            initial_joint_pos (None or th.tensor): If specified, the initial joint positions to set the locked joints.
                Default is the current joint positions of the robot
            skip_obstacle_update (bool): Whether to skip updating obstacles

        Returns:
            2-tuple
                - th.tensor or None: Joint positions to reach target pose or None if impossible to reach the target pose
                - th.tensor: Indices for joints in the robot
        """
        if isinstance(target_pose, tuple):
            target_pose = {self.arm: target_pose}

        batched = len(target_pose[list(target_pose.keys())[0]][0].shape) == 2

        target_pos = {}
        target_quat = {}
        for eef, pose in target_pose.items():
            if eef == "eyes":
                if batched and len(target_pose["eyes"][0].shape) == 1:
                    batch_size = target_pose[list(target_pose.keys())[0]][0].shape[0]
                    target_pos[eef] = pose[0].repeat(batch_size, 1)
                    target_quat[eef] = pose[1].repeat(batch_size, 1)
                else:
                    target_pos[eef] = pose[0]
                    target_quat[eef] = pose[1]
            elif eef in self.robot.eef_link_names:
                target_pos[self.robot.eef_link_names[eef]] = pose[0]
                target_quat[self.robot.eef_link_names[eef]] = pose[1]

        # These parameters are specially for speedig up the IK solving here. Don't use it globally
        ik_max_attempts = 50
        ik_fail_return = 50
        timeout = 60.0

        if not batched:
            target_pos = {
                k: th.stack([v for _ in range(self._motion_generator.batch_size)]) for k, v in target_pos.items()
            }
            target_quat = {
                k: th.stack([v for _ in range(self._motion_generator.batch_size)]) for k, v in target_quat.items()
            }
            max_attempts = math.ceil(ik_max_attempts / self._motion_generator.batch_size)
            success_ratio = 1.0 / self._motion_generator.batch_size
        else:
            max_attempts = ik_max_attempts
            success_ratio = 1.0

        successes, joint_states = self._motion_generator.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=initial_joint_pos,
            is_local=False,
            max_attempts=max_attempts,
            timeout=timeout,
            ik_fail_return=ik_fail_return,
            enable_finetune_trajopt=False,
            finetune_attempts=0,
            return_full_result=False,
            success_ratio=success_ratio,
            attached_obj=None if not attach_obj else self.attached_obj_info["attached_obj"],
            attached_obj_scale=None if not attach_obj else self.attached_obj_info["attached_obj_scale"],
            motion_constraint=None,
            skip_obstacle_update=skip_obstacle_update,
            ik_only=True,
            ik_world_collision_check=ik_world_collision_check,
            emb_sel=emb_sel,
        )
        # print("IK solver successes: ", successes)

        if not batched:
            # Grab the first successful joint state if found
            success_idx = th.where(successes)[0].cpu()
            if len(success_idx) == 0:
                return None
        else:
            # We want all of them to be successful because remeber that the target poses represent all the waypoints in the folloiwing manipulation phase
            if not th.all(successes):
                return None

        if not batched:
            joint_state = joint_states[success_idx[0]]
            joint_pos = self._motion_generator.path_to_joint_trajectory(
                joint_state, get_full_js=False, emb_sel=emb_sel
            )
            return joint_pos.cpu()
        else:
            # TODO: Check if this will work with right hand as manipulation control idx returns for the left hand
            # self.temp_js = joint_states
            return [
                self._motion_generator.path_to_joint_trajectory(
                    joint_state, get_full_js=False, emb_sel=emb_sel
                ).cpu()
                for joint_state in joint_states
            ]

    def _move_hand(
        self,
        target_pose,
        stop_on_contact=False,
        stop_on_ag=False,
        motion_constraint=None,
        low_precision=False,
        lock_auxiliary_arm=False,
        ignore_objects=None,
    ):
        """
        Yields action for the robot to move hand so the eef is in the target pose using the planner

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            stop_on_contact (bool): Whether to stop executing motion plan if contact is detected
            stop_on_ag (bool): Whether to stop executing motion plan if assisted grasping is activated
            motion_constraint (MotionConstraint): Motion constraint for the motion
            low_precision (bool): Whether to use low precision for the motion
            lock_auxiliary_arm (bool): Whether to lock the other arm in place
            ignore_objects (None or list of str): If specified, objects that should be ignored when planning

        Returns:
            th.tensor or None: Action array for one step for the robot to move hand or None if its at the target pose
        """
        if self.debug_visual_marker is not None:
            self.debug_visual_marker.set_position_orientation(*target_pose)

        yield from self._settle_robot()
        # curobo motion generator takes a pose but outputs joint positions
        if not lock_auxiliary_arm:
            target_pos = {
                self.robot.eef_link_names[self.arm]: target_pose[0],
            }
            target_quat = {
                self.robot.eef_link_names[self.arm]: target_pose[1],
            }
        else:
            left_hand_pos, left_hand_quat = target_pose if self.arm == "left" else self.robot.get_eef_pose(arm="left")
            right_hand_pos, right_hand_quat = (
                target_pose if self.arm == "right" else self.robot.get_eef_pose(arm="right")
            )
            target_pos = {
                self.robot.eef_link_names["left"]: left_hand_pos,
                self.robot.eef_link_names["right"]: right_hand_pos,
            }
            target_quat = {
                self.robot.eef_link_names["left"]: left_hand_quat,
                self.robot.eef_link_names["right"]: right_hand_quat,
            }
        q_traj = self._plan_joint_motion(
            target_pos=target_pos,
            target_quat=target_quat,
            embodiment_selection=CuRoboEmbodimentSelection.ARM,
            motion_constraint=motion_constraint,
            ignore_objects=ignore_objects,
        )

        indented_print(f"Plan has {len(q_traj)} steps")
        yield from self._execute_motion_plan(
            q_traj, stop_on_contact=stop_on_contact, stop_on_ag=stop_on_ag, low_precision=low_precision
        )

    def _plan_joint_motion(
        self,
        target_pos,
        target_quat,
        embodiment_selection=CuRoboEmbodimentSelection.DEFAULT,
        motion_constraint=None,
        skip_obstacle_update=False,
        ignore_objects=None,
    ):
        # If an object is grasped, we need to pass it to the motion planner
        # obj_in_hand = self._get_obj_in_hand()
        # attached_obj = {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link} if obj_in_hand is not None else None

        # Aggregate target_pos and target_quat to match batch_size
        target_pos = {k: th.stack([v for _ in range(self._motion_generator.batch_size)]) for k, v in target_pos.items()}
        target_quat = {
            k: th.stack([v for _ in range(self._motion_generator.batch_size)]) for k, v in target_quat.items()
        }

        if not skip_obstacle_update:
            self._motion_generator.update_obstacles(ignore_objects=ignore_objects)

        eyes_target_pos, eyes_target_quat = None, None
        if self._enable_head_tracking and isinstance(self.robot, R1):
            obj_pose = self._tracking_object.get_position_orientation()
            eyes_target_pos = obj_pose[0]
            eyes_target_quat = obj_pose[1]

        # breakpoint()
        start_time = time.time()
        results, traj_paths = self._motion_generator.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=math.ceil(m.MAX_PLANNING_ATTEMPTS / self._motion_generator.batch_size),
            timeout=60.0,
            ik_fail_return=m.MAX_IK_FAILURES_BEFORE_RETURN,
            enable_finetune_trajopt=True,
            finetune_attempts=1,
            return_full_result=True,
            success_ratio=1.0 / self._motion_generator.batch_size,
            attached_obj=self.attached_obj_info["attached_obj"],
            attached_obj_scale=self.attached_obj_info["attached_obj_scale"],
            motion_constraint=motion_constraint,
            skip_obstacle_update=True,
            ik_only=False,
            ik_world_collision_check=True,
            emb_sel=embodiment_selection,
            eyes_target_pos=eyes_target_pos,
            eyes_target_quat=eyes_target_quat,
        )
        self.base_mp_planning_time = round(time.time() - start_time, 2)
        # Grab the first successful trajectory if found
        successes = results[0].success
        print("Base motion planning successes: ", successes)
        # print("len(trajs): ", [len(t) for t in traj_paths])
        success_idx = th.where(successes)[0].cpu()
        if len(success_idx) == 0:
            print("Base motion planning failed", results[0].status)
            print("pos error: ", results[0].position_error)
            print("orn error: ", results[0].rotation_error)
            print("feasible: ", results[0].feasible)

            if results[0].status.value == "IK Fail":
                self.mp_err = "BaseMPIKFailed"
                q_traj = None
            else:
                self.mp_err = "BaseMPFailed"
                q_traj = None
                # breakpoint()
                # # Sort traj by their len and then choose largest to smallest while checking that they satisfy the position and orientation constraint
                # # As this base MP failure mostly happens when the robot needs to go on the other side of the table, larger trajectories should generally be better
                # traj_paths = sorted(traj_paths, key=lambda x: len(x), reverse=True)
                # sorted_indices = [i[0] for i in sorted(enumerate(traj_paths), key=lambda x: len(x[1]), reverse=True)]
                # success_idx = -1
                # for idx in sorted_indices:
                #     if results[0].position_error[idx] < 0.01 and results[0].rotation_error[idx] < 0.1:
                #         success_idx = idx
                #         break

                # print("success_idx: ", success_idx, "len: ", len(traj_paths[success_idx]))
                # if success_idx == -1:
                #     print("No successful trajectory found")
                #     q_traj = None
                # else:
                #     traj_path = traj_paths[success_idx]
                #     q_traj = (
                #         self._motion_generator.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=embodiment_selection)
                #         .cpu()
                #         .float()
                #     )
                #     q_traj = self._motion_generator.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.01)

        else:
            pick_shortest_traj = True
            # Pick the lowest length traj_paths from all the success_idx
            if pick_shortest_traj:
                shortest_traj_idx = success_idx[0]
                shortest_traj_len = len(traj_paths[shortest_traj_idx])
                for idx in success_idx:
                    if len(traj_paths[idx]) < shortest_traj_len:
                        shortest_traj_idx = idx
                        shortest_traj_len = len(traj_paths[idx])
                traj_path = traj_paths[shortest_traj_idx]
            else:
                traj_path = traj_paths[success_idx[0]]

            print("pos error: ", results[0].position_error[success_idx[0]])
            print("orn error: ", results[0].rotation_error[success_idx[0]])
            print("feasible: ", results[0].feasible)
            q_traj = (
                self._motion_generator.path_to_joint_trajectory(
                    traj_path, get_full_js=True, emb_sel=embodiment_selection
                )
                .cpu()
                .float()
            )
            q_traj = self._motion_generator.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.01)
            self.mp_err = "None"

        return q_traj

    def _execute_motion_plan(
        self,
        q_traj,
        stop_on_contact=False,
        stop_on_ag=False,
        ignore_failure=False,
        low_precision=False,
        ignore_physics=False,
    ):
        dist_threshold = m.LOW_PRECISION_DIST_THRESHOLD if low_precision else m.DEFAULT_DIST_THRESHOLD
        angle_threshold = m.LOW_PRECISION_ANGLE_THRESHOLD if low_precision else m.DEFAULT_ANGLE_THRESHOLD

        for i, joint_pos in enumerate(q_traj):
            # indented_print(f"Executing motion plan step {i + 1}/{len(q_traj)}")
            if ignore_physics:
                self.robot.set_joint_positions(joint_pos)
                og.sim.step()
                if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                    return
            else:
                # Convert target joint positions to command
                base_target_reached = False
                articulation_target_reached = False
                collision_detected = False
                articulation_control_idx = [self.robot.trunk_control_idx]
                for arm in self.robot.arm_names:
                    articulation_control_idx.append(self.robot.arm_control_idx[arm])
                articulation_control_idx = th.cat(articulation_control_idx)
                for j in range(m.MAX_STEPS_FOR_JOINT_MOTION):
                    # indented_print(f"Step {j + 1}/{m.MAX_STEPS_FOR_JOINT_MOTION}")

                    # We need to call @q_to_action for every step because as the robot base moves, the same base joint_pos will be
                    # converted to different actions, since HolonomicBaseJointController accepts an action in the robot local frame.
                    action = self.robot.q_to_action(joint_pos)
                    yield self._postprocess_action(action)

                    current_joint_pos = self.robot.get_joint_positions()
                    joint_pos_diff = joint_pos - current_joint_pos
                    base_joint_diff = joint_pos_diff[self.robot.base_control_idx]
                    articulation_joint_diff = joint_pos_diff[articulation_control_idx]  # Gets all non-base joints
                    articulation_threshold = (
                        m.JOINT_POS_DIFF_THRESHOLD if not low_precision else m.LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD
                    )
                    max_articulation_joint_diff = th.max(th.abs(articulation_joint_diff)).item()
                    if max_articulation_joint_diff < articulation_threshold:
                        articulation_target_reached = True

                    max_base_pos_diff = th.max(th.abs(base_joint_diff[:2])).item()
                    max_base_orn_diff = th.abs(wrap_angle(base_joint_diff[2]))
                    if max_base_pos_diff < dist_threshold and max_base_orn_diff < angle_threshold:
                        base_target_reached = True

                    # Early stopping if the base, trunk and arm joints are at the target positions
                    if base_target_reached and articulation_target_reached:
                        break

                    collision_detected = detect_robot_collision_in_sim(self.robot)
                    if stop_on_contact and collision_detected:
                        return

                    if stop_on_ag and self._get_obj_in_hand() is not None:
                        return

                if not ignore_failure:
                    if not base_target_reached:
                        indented_print(f"max base pos diff: {max_base_pos_diff}")
                        indented_print(f"max base angle diff: {max_base_orn_diff}")
                        self.mp_err = "BaseExecutionBaseTargetNotReached"
                        yield None
                        # breakpoint()
                        # raise ActionPrimitiveError(
                        #     ActionPrimitiveError.Reason.EXECUTION_ERROR,
                        #     "Could not reach the target base joint positions. Try again",
                        # )
                        # return
                    if not articulation_target_reached:
                        indented_print(f"max articulation joint diff: {max_articulation_joint_diff}")
                        self.mp_err = "BaseExecutionArmTorsoTargetNotReached"
                        yield None
                        # breakpoint()
                        # raise ActionPrimitiveError(
                        #     ActionPrimitiveError.Reason.EXECUTION_ERROR,
                        #     "Could not reach the target articulation joint positions. Try again",
                        # )
                        # return

    def _add_linearly_interpolated_waypoints(self, plan, max_inter_dist=0.01):
        """
        Adds waypoints to the plan so the distance between values in the plan never exceeds the max_inter_dist.

        Args:
            plan (Array of arrays): Planned path
            max_inter_dist (float): Maximum distance between values in the plan

        Returns:
            Array of arrays: Planned path with additional waypoints
        """
        assert len(plan) > 1, "Plan must have at least 2 waypoints to interpolate"
        interpolated_plan = []
        for i in range(len(plan) - 1):
            # Calculate maximum difference across all dimensions
            max_diff = (plan[i + 1] - plan[i]).abs().max()
            num_intervals = math.ceil(max_diff.item() / max_inter_dist)
            interpolated_plan += multi_dim_linspace(plan[i], plan[i + 1], num_intervals, endpoint=False)

        interpolated_plan.append(plan[-1])
        return interpolated_plan

    def _move_hand_direct_joint(self, joint_pos, stop_on_contact=False, ignore_failure=False):
        """
        Yields action for the robot to move its arm to reach the specified joint positions by directly actuating with no planner

        Args:
            joint_pos (th.tensor): Array of joint positions for the arm
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions

        Returns:
            th.tensor or None: Action array for one step for the robot to move arm or None if its at the joint positions
        """

        # Store the previous eef pose for checking if we got stuck
        prev_eef_pos = th.zeros(3)

        # All we need to do here is save the target joint position so that empty action takes us towards it
        controller_name = f"arm_{self.arm}"
        self._arm_targets[controller_name] = joint_pos

        for i in range(m.MAX_STEPS_FOR_HAND_MOVE_JOINT):
            current_joint_pos = self.robot.get_joint_positions()[self._manipulation_control_idx()]
            diff_joint_pos = joint_pos - current_joint_pos
            if th.max(th.abs(diff_joint_pos)).item() < m.JOINT_POS_DIFF_THRESHOLD:
                return
            if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                return
            # check if the eef stayed in the same pose for sufficiently long
            if (
                og.sim.get_sim_step_dt() * i > m.TIME_BEFORE_JOINT_STUCK_CHECK
                and th.max(th.abs(self.robot.get_eef_position(self.arm) - prev_eef_pos)).item() < 0.0001
            ):
                # We're stuck!
                break

            # Since we set the new joint target as the arm_target, the empty action will take us towards it.
            action = self._empty_action()

            prev_eef_pos = self.robot.get_eef_position(self.arm)
            yield self._postprocess_action(action)

        if not ignore_failure:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Your hand was obstructed from moving to the desired joint position",
            )

    def _move_hand_direct_ik(
        self,
        target_pose,
        stop_on_contact=False,
        ignore_failure=False,
        pos_thresh=0.02,
        ori_thresh=0.4,
        in_world_frame=True,
        stop_if_stuck=False,
    ):
        """
        Moves the hand to a target pose using inverse kinematics.

        Args:
            target_pose (tuple): A tuple of two elements, representing the target pose of the hand as a position and a quaternion.
            stop_on_contact (bool, optional): Whether to stop the movement if the hand collides with an object. Defaults to False.
            ignore_failure (bool, optional): Whether to raise an exception if the movement fails. Defaults to False.
            pos_thresh (float, optional): The position threshold for considering the target pose reached. Defaults to 0.04.
            ori_thresh (float, optional): The orientation threshold for considering the target pose reached. Defaults to 0.4.
            in_world_frame (bool, optional): Whether the target pose is given in the world frame. Defaults to True.
            stop_if_stuck (bool, optional): Whether to stop the movement if the hand is stuck. Defaults to False.

        Yields:
            numpy.ndarray: The action to be executed by the robot controller.

        Raises:
            ActionPrimitiveError: If the movement fails and ignore_failure is False.
        """
        # make sure controller is InverseKinematicsController and in expected mode
        controller_config = self.robot._controller_config["arm_" + self.arm]
        assert (
            controller_config["name"] == "InverseKinematicsController"
        ), "Controller must be InverseKinematicsController"
        assert controller_config["mode"] == "pose_absolute_ori", "Controller must be in pose_absolute_ori mode"
        if in_world_frame:
            target_pose = self._world_pose_to_robot_pose(target_pose)
        target_pos = target_pose[0]
        target_orn = target_pose[1]
        target_orn_axisangle = T.quat2axisangle(target_pose[1])
        prev_pos = prev_orn = None

        # All we need to do here is save the target IK position so that empty action takes us towards it
        controller_name = f"arm_{self.arm}"
        self._arm_targets[controller_name] = (target_pos, target_orn_axisangle)

        for i in range(m.MAX_STEPS_FOR_HAND_MOVE_IK):
            current_pose = self._world_pose_to_robot_pose(
                (self.robot.get_eef_position(self.arm), self.robot.get_eef_orientation(self.arm))
            )
            current_pos = current_pose[0]
            current_orn = current_pose[1]

            delta_pos = target_pos - current_pos
            target_pos_diff = th.norm(delta_pos)
            target_orn_diff = T.get_orientation_diff_in_radian(current_orn, target_orn)
            reached_goal = target_pos_diff < pos_thresh and target_orn_diff < ori_thresh
            if reached_goal:
                return

            if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                return

            # if i > 0 and stop_if_stuck and detect_robot_collision_in_sim(self.robot):
            if i > 0 and stop_if_stuck:
                pos_diff = th.norm(prev_pos - current_pos)
                orn_diff = T.get_orientation_diff_in_radian(current_orn, prev_orn)
                if pos_diff < 0.0003 and orn_diff < 0.01:
                    raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "Hand is stuck")

            prev_pos = current_pos
            prev_orn = current_orn

            # Since we set the new IK target as the arm_target, the empty action will take us towards it.
            action = self._empty_action()
            yield self._postprocess_action(action)

        if not ignore_failure:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Your hand was obstructed from moving to the desired joint position",
            )

    def _move_hand_linearly_cartesian(
        self, target_pose, stop_on_contact=False, ignore_failure=False, stop_if_stuck=False
    ):
        """
        Yields action for the robot to move its arm to reach the specified target pose by moving the eef along a line in cartesian
        space from its current pose

        Args:
            target_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions

        Returns:
            th.tensor or None: Action array for one step for the robot to move arm or None if its at the target pose
        """
        # To make sure that this happens in a roughly linear fashion, we will divide the trajectory
        # into 1cm-long pieces
        start_pos, start_orn = self.robot.eef_links[self.arm].get_position_orientation()
        travel_distance = th.norm(target_pose[0] - start_pos)
        num_poses = int(
            th.max(th.tensor([2, int(travel_distance / m.MAX_CARTESIAN_HAND_STEP) + 1], dtype=th.float32)).item()
        )
        pos_waypoints = multi_dim_linspace(start_pos, target_pose[0], num_poses)

        # Also interpolate the rotations
        t_values = th.linspace(0, 1, num_poses)
        quat_waypoints = [T.quat_slerp(start_orn, target_pose[1], t) for t in t_values]

        controller_config = self.robot._controller_config["arm_" + self.arm]
        if controller_config["name"] == "InverseKinematicsController":
            waypoints = list(zip(pos_waypoints, quat_waypoints))

            for i, waypoint in enumerate(waypoints):
                if i < len(waypoints) - 1:
                    yield from self._move_hand_direct_ik(
                        waypoint,
                        stop_on_contact=stop_on_contact,
                        ignore_failure=ignore_failure,
                        stop_if_stuck=stop_if_stuck,
                    )
                else:
                    yield from self._move_hand_direct_ik(
                        waypoints[-1],
                        pos_thresh=0.01,
                        ori_thresh=0.1,
                        stop_on_contact=stop_on_contact,
                        ignore_failure=ignore_failure,
                        stop_if_stuck=stop_if_stuck,
                    )

                # Also decide if we can stop early.
                current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
                pos_diff = th.norm(current_pos - target_pose[0])
                orn_diff = T.get_orientation_diff_in_radian(target_pose[1], current_orn).item()
                if pos_diff < m.HAND_DIST_THRESHOLD and orn_diff < th.deg2rad(th.tensor([0.1])).item():
                    return

                if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                    return

            if not ignore_failure:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Your hand was obstructed from moving to the desired world position",
                )
        else:
            # Use joint positions
            joint_space_data = [
                self._convert_cartesian_to_joint_space(waypoint) for waypoint in zip(pos_waypoints, quat_waypoints)
            ]
            joints = list(self.robot.joints.values())

            for joint_pos in joint_space_data:
                # Check if the movement can be done roughly linearly.
                current_joint_positions = self.robot.get_joint_positions()[self._manipulation_control_idx()]

                failed_joints = []
                for joint_idx, target_joint_pos, current_joint_pos in zip(
                    self._manipulation_control_idx(), joint_pos, current_joint_positions
                ):
                    if th.abs(target_joint_pos - current_joint_pos) > m.MAX_ALLOWED_JOINT_ERROR_FOR_LINEAR_MOTION:
                        failed_joints.append(joints[joint_idx].joint_name)

                if failed_joints:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.EXECUTION_ERROR,
                        "You cannot reach the target position in a straight line - it requires rotating your arm which might cause collisions. You might need to get closer and retry",
                        {"failed joints": failed_joints},
                    )

                # Otherwise, move the joint
                yield from self._move_hand_direct_joint(
                    joint_pos, stop_on_contact=stop_on_contact, ignore_failure=ignore_failure
                )

                # Also decide if we can stop early.
                current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
                pos_diff = th.norm(current_pos - target_pose[0])
                orn_diff = T.get_orientation_diff_in_radian(target_pose[1], current_orn)
                if pos_diff < 0.001 and orn_diff < th.deg2rad(th.tensor([0.1])).item():
                    return

                if stop_on_contact and detect_robot_collision_in_sim(self.robot):
                    return

            if not ignore_failure:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Your hand was obstructed from moving to the desired world position",
                )

    def _get_joint_position_with_fingers_at_limit(self, limit_type):
        """
        Helper function to get the joint positions when the robot's fingers are at their limit positions.

        Args:
            limit_type (str): Either 'lower' for grasping or 'upper' for releasing.

        Yields:
            th.tensor: Joint positions for the robot with fingers at their limit positions.
        """
        target_joint_positions = self.robot.get_joint_positions()
        gripper_ctrl_idx = self.robot.gripper_control_idx[self.arm]
        if limit_type == self.robot._grasping_direction:
            finger_joint_limits = self.robot.joint_lower_limits[gripper_ctrl_idx]
        else:
            finger_joint_limits = self.robot.joint_upper_limits[gripper_ctrl_idx]
        target_joint_positions[gripper_ctrl_idx] = finger_joint_limits
        return target_joint_positions

    def _move_fingers_to_limit(self, limit_type):
        """
        Helper function to move the robot's fingers to their limit positions.

        Args:
            limit_type (str): Either 'lower' for grasping or 'upper' for releasing.

        Yields:
            th.tensor or None: Action array for one step for the robot to move fingers or None if done.
        """
        target_joint_positions = self._get_joint_position_with_fingers_at_limit(limit_type)
        action = self.robot.q_to_action(target_joint_positions)
        for _ in range(m.MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield self._postprocess_action(action)
            current_joint_positions = self.robot.get_joint_positions()
            if th.allclose(current_joint_positions, target_joint_positions, atol=m.JOINT_POS_DIFF_THRESHOLD):
                break
            elif limit_type == "lower" and self._get_obj_in_hand() is not None:
                # If we are grasping an object, we should stop when object is detected in hand
                break

    def _execute_grasp(self):
        """
        Yields action for the robot to grasp.

        Returns:
            th.tensor or None: Action array for one step for the robot to grasp or None if done grasping.
        """
        yield from self._move_fingers_to_limit("lower")

    def _execute_release(self):
        """
        Yields action for the robot to release its grasp.

        Returns:
            th.tensor or None: Action array for one step for the robot to release or None if done releasing.
        """
        yield from self._move_fingers_to_limit("upper")

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "An object was still detected in your hand after executing release",
                {"object in hand": self._get_obj_in_hand().name},
            )

    def _overwrite_head_action(self, action):
        """
        Overwrites camera control actions to track an object of interest.
        If self._always_track_eef is true, always tracks the end effector of the robot.
        Otherwise, tracks the object of interest or the end effector as specified by the primitive.

        Args:
            action (array) : action array to overwrite
        """
        if self._always_track_eef:
            target_obj_pose = (self.robot.get_eef_position(self.arm), self.robot.get_eef_orientation(self.arm))
        else:
            if self._tracking_object is None:
                return action

            if self._tracking_object == self.robot:
                target_obj_pose = (self.robot.get_eef_position(self.arm), self.robot.get_eef_orientation(self.arm))
            else:
                target_obj_pose = self._tracking_object.get_position_orientation()

        assert isinstance(self.robot, Tiago), "Tracking object with camera is currently only supported for Tiago"

        head_q = self._get_head_goal_q(target_obj_pose)
        head_idx = self.robot.controller_action_idx["camera"]

        config = self.robot._controller_config["camera"]
        assert config["name"] == "JointController", "Camera controller must be JointController"
        assert config["motor_type"] == "position", "Camera controller must be in position control mode"
        use_delta = config["use_delta_commands"]

        if use_delta:
            cur_head_q = self.robot.get_joint_positions()[self.robot.camera_control_idx]
            head_action = head_q - cur_head_q
        else:
            head_action = head_q
        action[head_idx] = head_action
        return action

    def _get_head_goal_q(self, target_obj_pose):
        """
        Get goal joint positions for head to look at an object of interest,
        If the object cannot be seen, return the reset (default) head joint positions.
        """

        # get current head joint positions
        head1_joint = self.robot.joints["head_1_joint"]
        head2_joint = self.robot.joints["head_2_joint"]
        head1_joint_limits = [head1_joint.lower_limit, head1_joint.upper_limit]
        head2_joint_limits = [head2_joint.lower_limit, head2_joint.upper_limit]
        head1_joint_goal = head1_joint.get_state()[0][0]
        head2_joint_goal = head2_joint.get_state()[0][0]

        # grab robot and object poses
        robot_pose = self.robot.get_position_orientation()
        # obj_pose = obj.get_position_orientation()
        obj_in_base = T.relative_pose_transform(*target_obj_pose, *robot_pose)

        # compute angle between base and object in xy plane (parallel to floor)
        theta = th.arctan2(obj_in_base[0][1], obj_in_base[0][0])

        # if it is possible to get object in view, compute both head joint positions
        if head1_joint_limits[0] < theta < head1_joint_limits[1]:
            head1_joint_goal = theta

            # compute angle between base and object in xz plane (perpendicular to floor)
            head2_pose = self.robot.links["head_2_link"].get_position_orientation()
            head2_in_base = T.relative_pose_transform(*head2_pose, *robot_pose)

            phi = th.arctan2(obj_in_base[0][2] - head2_in_base[0][2], obj_in_base[0][0])
            if head2_joint_limits[0] < phi < head2_joint_limits[1]:
                head2_joint_goal = phi

        # if not possible to look at object, return the reset (default) head joint positions
        else:
            default_head_pos = self.robot.reset_joint_pos[self.robot.controller_action_idx["camera"]]
            head1_joint_goal = default_head_pos[0]
            head2_joint_goal = default_head_pos[1]

        return th.tensor([head1_joint_goal, head2_joint_goal])

    def _empty_action(self, follow_arm_targets=True):
        """
        Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

        Args:
            follow_arm_targets (bool): Whether to move the arms to the saved pose targets or keep them still.

        Returns:
            th.tensor or None: Action array for one step for the robot to do nothing
        """
        action = th.zeros(self.robot.action_dim)
        for name, controller in self.robot._controllers.items():
            # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
            if follow_arm_targets and name in self._arm_targets:
                if isinstance(controller, InverseKinematicsController):
                    arm = name.replace("arm_", "")
                    target_pos, target_orn_axisangle = self._arm_targets[name]
                    current_pos, current_orn = self._world_pose_to_robot_pose(
                        (self.robot.get_eef_position(arm), self.robot.get_eef_orientation(arm))
                    )
                    delta_pos = target_pos - current_pos
                    if controller.mode == "pose_delta_ori":
                        delta_orn = T.orientation_error(
                            T.quat2mat(T.axisangle2quat(target_orn_axisangle)), T.quat2mat(current_orn)
                        )
                        partial_action = th.cat((delta_pos, delta_orn))
                    elif controller.mode in "pose_absolute_ori":
                        partial_action = th.cat((delta_pos, target_orn_axisangle))
                    elif controller.mode == "absolute_pose":
                        partial_action = th.cat((target_pos, target_orn_axisangle))
                    else:
                        raise ValueError("Unexpected IK control mode")
                else:
                    target_joint_pos = self._arm_targets[name]
                    current_joint_pos = self.robot.get_joint_positions()[self._manipulation_control_idx()]
                    if controller.use_delta_commands:
                        partial_action = target_joint_pos - current_joint_pos
                    else:
                        partial_action = target_joint_pos
            else:
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
            action_idx = self.robot.controller_action_idx[name]
            action[action_idx] = partial_action
        return action

    def _reset_robot(self):
        """
        Yields action to move both hands to the position optimal for executing subsequent action primitives

        Returns:
            th.tensor or None: Action array for one step for the robot to reset its hands or None if it is done resetting
        """
        indented_print("Resetting robot")
        target_pos = dict()
        target_quat = dict()
        for arm in self.robot.arm_names:
            reset_eef_pose = self._get_reset_eef_pose("world")[arm]
            target_pos[self.robot.eef_link_names[arm]] = reset_eef_pose[0]
            target_quat[self.robot.eef_link_names[arm]] = reset_eef_pose[1]
        q_traj = self._plan_joint_motion(
            target_pos=target_pos,
            target_quat=target_quat,
            embodiment_selection=CuRoboEmbodimentSelection.ARM,
        )
        indented_print(f"Plan has {len(q_traj)} steps")
        yield from self._execute_motion_plan(q_traj)

    def _reset_hand(self):
        """
        Yields action to move the hand to the position optimal for executing subsequent action primitives

        Returns:
            th.tensor or None: Action array for one step for the robot to reset its hand or None if it is done resetting
        """
        indented_print("Resetting hand")
        # TODO: make this work with both hands
        reset_eef_pose = self._get_reset_eef_pose("world")[self.arm]
        if self.debug_visual_marker is not None:
            self.debug_visual_marker.set_position_orientation(*reset_eef_pose)
        yield from self._move_hand(reset_eef_pose)

    def _get_reset_eef_pose(self, frame="robot"):
        """
        Get the reset eef pose for the robot

        Args:
            frame (str): The frame in which the reset eef pose is specified, one of "robot" or "world"

        Returns:
            dict of th.tensor: The reset eef pose for each robot arm
        """
        pose = self._reset_eef_pose
        if frame == "robot":
            return pose
        elif frame == "world":
            return {arm: self._robot_pose_to_world_pose(pose[arm]) for arm in pose}
        else:
            raise ValueError(f"Unsupported frame: {frame}")

    def _navigate_to_pose(self, pose_2d, skip_obstacle_update=False, visibility_constraint=False):
        """
        Yields the action to navigate robot to the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose
            skip_obstacle_update (bool): Determines whether to skip updating the obstacles in the scene

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        pose_3d = self._get_robot_pose_from_2d_pose(pose_2d)
        if self.debug_visual_marker is not None:
            self.debug_visual_marker.set_position_orientation(*pose_3d)
        target_pos = {self.robot.base_footprint_link_name: pose_3d[0]}
        target_quat = {self.robot.base_footprint_link_name: pose_3d[1]}

        # if visibility_constraint:
        assert self.target_eyes_pose is not None, "target_eyes_pose is None"
        if isinstance(self.robot, R1):
            target_pos["eyes"] = self.target_eyes_pose[0]
            target_quat["eyes"] = self.target_eyes_pose[1]

        print("Base motion planning")
        q_traj = self._plan_joint_motion(
            target_pos,
            target_quat,
            embodiment_selection=CuRoboEmbodimentSelection.BASE,
            skip_obstacle_update=skip_obstacle_update,
        )

        # If no valid MP was found, yield None (don't return None, that does not work)
        if q_traj is None:
            yield None

        # # remove later
        # yield None

        # low precision is mainly added for clean pan task
        yield from self._execute_motion_plan(q_traj, low_precision=True)

    # (TODO) add a function to draw curobo-generated plans.
    def _draw_plan(self, plan):
        SEARCHED = []
        trav_map = self.robot.scene._trav_map
        for q in plan:
            # The below code is useful for plotting the RRT tree.
            map_point = trav_map.world_to_map((q[0], q[1]))
            SEARCHED.append(th.flip(map_point, dims=tuple(range(map_point.dim()))))

            fig = plt.figure()
            plt.imshow(trav_map.floor_map[0])
            plt.scatter(*zip(*SEARCHED), 5)
            fig.canvas.draw()

            # Convert the canvas to image
            img = th.frombuffer(fig.canvas.tostring_rgb(), dtype=th.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # Convert to BGR for cv2-based viewing.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow("SceneGraph", img)
            cv2.waitKey(1)

    def _navigate_if_needed(self, obj, eef_pose=None):
        """
        Yields action to navigate the robot to be in range of the object if it not in the range

        Args:
            obj (StatefulObject): Object for the robot to be in range of
            eef_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        self._motion_generator.update_obstacles()

        if eef_pose is None:
            eef_pose, _ = self._sample_grasp_pose(obj)

        if self._target_in_reach_of_robot(eef_pose, skip_obstacle_update=True):
            # No need to navigate.
            return

        yield from self._navigate_to_obj(obj, eef_pose=eef_pose, skip_obstacle_update=True)

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False, visibility_constraint=False):
        """
        Yields action to navigate the robot to be in range of the pose

        Args:
            obj (StatefulObject or list of StatefulObject): object(s) to be in range of
            eef_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate in range or None if it is done navigating
        """
        # Attach object to robot if requested. We do this here to avoid calling attach_objects_to_robot in the motion generator multiple times (it's called multiple
        # times in sampling a base pose) as that leads to a trimesh error (https://github.com/mikedh/trimesh/issues/550).
        emb_sel = CuRoboEmbodimentSelection.ARM
        if self.attached_obj_info["attached_obj"] is not None:
            q_pos = th.stack([self.robot.get_joint_positions()] * self._motion_generator.batch_size, axis=0)
            q_vel = th.stack([self.robot.get_joint_velocities()] * self._motion_generator.batch_size, axis=0)
            q_eff = th.stack([self.robot.get_joint_efforts()] * self._motion_generator.batch_size, axis=0)
            cu_joint_state = lazy.curobo.types.state.JointState(
                position=self._motion_generator._tensor_args.to_device(q_pos),
                velocity=self._motion_generator._tensor_args.to_device(q_vel) * 0.0,
                acceleration=self._motion_generator._tensor_args.to_device(q_eff) * 0.0,
                jerk=self._motion_generator._tensor_args.to_device(q_eff) * 0.0,
                joint_names=self._motion_generator.robot_joint_names,
            )
            cu_js_batch = cu_joint_state.get_ordered_joint_state(
                self._motion_generator.mg[emb_sel].kinematics.joint_names
            )
            attached_info = self._motion_generator._attach_objects_to_robot(
                attached_obj=self.attached_obj_info["attached_obj"],
                attached_obj_scale=self.attached_obj_info["attached_obj_scale"],
                cu_js_batch=cu_js_batch,
                emb_sel=emb_sel,
            )
        start_time = time.time()
        pose = self._sample_pose_near_object(
            obj,
            eef_pose=eef_pose,
            skip_obstacle_update=skip_obstacle_update,
            visibility_constraint=visibility_constraint,
        )
        self.base_sampling_time = round(time.time() - start_time, 2)

        if self.attached_obj_info["attached_obj"] is not None:
            # Detach attached object if it was attached
            self._motion_generator._detach_objects_from_robot(attached_info, emb_sel)

        # # remove later
        # state_dict = og.sim.dump_state()
        # pose3d = self._get_robot_pose_from_2d_pose(pose)
        # self.robot.set_position_orientation(position=pose3d[0], orientation=pose3d[1])
        # for _ in range(50): og.sim.step()
        # breakpoint()
        # og.sim.load_state(state_dict)
        # for _ in range(50): og.sim.step()

        if pose is None:
            print(f"Could not find a valid pose near the object {obj.name}")
            self.mp_err = "BaseSamplingFailed"
            yield None

        print("base pose:", pose)
        print("eyes pose", self.target_eyes_pose)

        # # remove later
        # yield None

        # skip_obstacle_update=True because it's already updated in _sample_pose_near_object
        yield from self._navigate_to_pose(pose, skip_obstacle_update=True, visibility_constraint=visibility_constraint)

    def _navigate_to_pose_direct(self, pose_2d, low_precision=False):
        """
        Yields action to navigate the robot to the 2d pose without planning

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose
            low_precision (bool): Determines whether to navigate to the pose within a large range (low precision) or small range (high precison)

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        dist_threshold = m.LOW_PRECISION_DIST_THRESHOLD if low_precision else m.DEFAULT_DIST_THRESHOLD
        angle_threshold = m.LOW_PRECISION_ANGLE_THRESHOLD if low_precision else m.DEFAULT_ANGLE_THRESHOLD

        end_pose = self._get_robot_pose_from_2d_pose(pose_2d)

        for _ in range(m.MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            body_target_pose = self._world_pose_to_robot_pose(end_pose)
            if th.norm(body_target_pose[0][:2]) < dist_threshold:
                break

            diff_pos = end_pose[0] - self.robot.get_position_orientation()[0]
            intermediate_pose = (
                end_pose[0],
                T.euler2quat(th.tensor([0, 0, math.atan2(diff_pos[1], diff_pos[0])], dtype=th.float32)),
            )
            body_intermediate_pose = self._world_pose_to_robot_pose(intermediate_pose)
            diff_yaw = T.quat2euler(body_intermediate_pose[1])[2].item()
            if abs(diff_yaw) > m.DEFAULT_ANGLE_THRESHOLD:
                yield from self._rotate_in_place(intermediate_pose, angle_threshold=m.DEFAULT_ANGLE_THRESHOLD)
            else:
                action = self._empty_action()
                if isinstance(self.robot.controllers["base"], HolonomicBaseJointController):
                    assert (
                        self.robot.controllers["base"].motor_type == "velocity"
                    ), "Holonomic base controller must be in velocity mode"
                    direction_vec = (
                        body_target_pose[0][:2] / th.norm(body_target_pose[0][:2]) * m.KP_LIN_VEL[type(self.robot)]
                    )
                    base_action = th.tensor([direction_vec[0], direction_vec[1], 0.0], dtype=th.float32)
                    action[self.robot.controller_action_idx["base"]] = base_action
                elif isinstance(self.robot.controllers["base"], DifferentialDriveController):
                    base_action = th.tensor([m.KP_LIN_VEL[type(self.robot)], 0.0], dtype=th.float32)
                    action[self.robot.controller_action_idx["base"]] = base_action
                else:
                    raise ValueError(f"Unsupported base controller: {type(self.robot.controllers['base'])}")

                yield self._postprocess_action(action)
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not navigate to the target position",
                {"target pose": end_pose},
            )

        # Rotate in place to final orientation once at location
        yield from self._rotate_in_place(end_pose, angle_threshold=angle_threshold)

    def _rotate_in_place(self, end_pose, angle_threshold=m.DEFAULT_ANGLE_THRESHOLD):
        """
        Yields action to rotate the robot to the 2d end pose

        Args:
            end_pose (Iterable): (x, y, yaw) 2d pose
            angle_threshold (float): The angle difference between the robot's current and end pose that determines when the robot is done rotating

        Returns:
            th.tensor or None: Action array for one step for the robot to rotate or None if it is done rotating
        """
        for _ in range(m.MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            body_target_pose = self._world_pose_to_robot_pose(end_pose)
            diff_yaw = T.quat2euler(body_target_pose[1])[2].item()
            if abs(diff_yaw) < angle_threshold:
                break

            action = self._empty_action()

            direction = -1.0 if diff_yaw < 0.0 else 1.0
            ang_vel = m.KP_ANGLE_VEL[type(self.robot)] * direction

            base_action = action[self.robot.controller_action_idx["base"]]

            if isinstance(self.robot.controllers["base"], HolonomicBaseJointController):
                assert (
                    self.robot.controllers["base"].motor_type == "velocity"
                ), "Holonomic base controller must be in velocity mode"
                base_action[0] = 0.0
                base_action[1] = 0.0
                base_action[2] = ang_vel
            elif isinstance(self.robot.controllers["base"], DifferentialDriveController):
                base_action[0] = 0.0
                base_action[1] = ang_vel
            else:
                raise ValueError(f"Unsupported base controller: {type(self.robot.controllers['base'])}")

            action[self.robot.controller_action_idx["base"]] = base_action
            yield self._postprocess_action(action)
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not rotate in place to the desired orientation",
                {"target pose": end_pose},
            )

        empty_action = self._empty_action()
        yield self._postprocess_action(empty_action)

    def _sample_pose_near_object(
        self,
        obj,
        eef_pose=None,
        plan_with_open_gripper=False,
        sampling_attempts=m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT,
        skip_obstacle_update=False,
        visibility_constraint=False,
    ):
        """
        Returns a 2d pose for the robot within in the range of the object and where the robot is not in collision with anything

        Args:
            obj (StatefulObject or list of StatefulObject): object(s) to sample a 2d pose near
            eef_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            plan_with_open_gripper (bool): Determines whether to plan with the gripper open even though it might be closed now.
                This is useful for the placing primitive because we will open-loop open the gripper after placing
            sampling_attempts (int): Number of attempts to sample a valid pose
            skip_obstacle_update (bool): Determines whether to skip updating the obstacles in the scene
        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        distance_lo, distance_hi = m.BASE_POSE_SAMPLING_LOWER_BOUND, m.BASE_POSE_SAMPLING_UPPER_BOUND
        yaw_lo, yaw_hi = -math.pi, math.pi

        if eef_pose is None:
            eef_pose, _ = self._sample_grasp_pose(obj)

        if isinstance(eef_pose, tuple):
            eef_pose = {self.arm: eef_pose}

        avg_arm_workspace_range = th.mean(self.robot.arm_workspace_range[self.arm])
        # We want the robot to be facing towards the object in a way that is more conducive to manipulating with the requested arm
        # So, if eef_pose has an arm provided, we use that to guide the yaw of the base
        if eef_pose is not None:
            avg_arm_workspace_range = th.tensor(
                [th.mean(self.robot.arm_workspace_range[arm_side]) for arm_side in eef_pose.keys()]
            ).mean()

        target_pose = eef_pose

        target_position = th.stack([eef_pose[arm_side][0] for arm_side in eef_pose.keys()]).mean(dim=(0, 1))

        for arm in target_pose:
            if len(target_pose[arm][0].shape) == 1:
                target_pose[arm] = (target_pose[arm][0].unsqueeze(0), target_pose[arm][1].unsqueeze(0))

        attempt = 0
        if not skip_obstacle_update:
            # Update obstacle once before sampling
            self._motion_generator.update_obstacles()

        if isinstance(self.robot, Tiago):
            visibility_constraint = False

        self.target_eyes_pose = None
        while attempt < sampling_attempts:
            print("attempt: ", attempt)
            candidate_poses = []
            self.target_eyes_pose_arr = []
            for _ in range(self._curobo_batch_size):
                while True:
                    distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
                    yaw = th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo
                    candidate_2d_pose = th.cat(
                        [
                            target_position[0] + distance * th.cos(yaw),
                            target_position[1] + distance * th.sin(yaw),
                            yaw + math.pi - avg_arm_workspace_range,
                        ]
                    )

                    # Hack to not allow sampling robot base poses on the other side of the table
                    # print("candidate_2d_pose[0]", candidate_2d_pose[0])
                    if self.use_base_pose_hack and candidate_2d_pose[0] > 1.0:
                        continue
                    else:
                        break

                candidate_poses.append(candidate_2d_pose)

            # Normally candidate_poses will have length equal to self._curobo_batch_size
            # In case we are unable to find a valid pose in the room, we will have less than self._curobo_batch_size.
            # We skip the following steps if the list is empty.
            if len(candidate_poses) > 0:
                result = self._validate_poses(
                    candidate_poses,
                    eef_pose=target_pose,
                    plan_with_open_gripper=plan_with_open_gripper,
                    skip_obstacle_update=True,
                    visibility_constraint=visibility_constraint,
                )

                # breakpoint()
                # Return the first valid pose. Note that for efficiency, we break the validation check loop in the _validate_poses function
                # as soon as we find the first success. So, the elements after the first success are not actaully tested and so should not be relied upon.
                for i, res in enumerate(result):
                    if res:
                        indented_print("Found valid position near object.")
                        # if visibility_constraint:
                        self.target_eyes_pose = self.target_eyes_pose_arr[i]
                        return candidate_poses[i]

            attempt += self._curobo_batch_size
        return None

    @staticmethod
    def _sample_position_on_aabb_side(target_obj):
        """
        Returns a position on one of the axis-aligned bounding box (AABB) side faces of the target object.

        Args:
            target_obj (StatefulObject): Object to sample a position on

        Returns:
            3-array: (x,y,z) Position in the world frame
        """
        aabb_center, aabb_extent = target_obj.aabb_center, target_obj.aabb_extent
        # We want to sample only from the side-facing faces.
        face_normal_axis = random.choice([0, 1])
        face_normal_direction = random.choice([-1, 1])
        face_center = aabb_center + th.eye(3)[face_normal_axis] * aabb_extent * face_normal_direction
        face_lateral_axis = 0 if face_normal_axis == 1 else 1
        face_lateral_half_extent = th.eye(3)[face_lateral_axis] * aabb_extent / 2
        face_vertical_half_extent = th.eye(3)[2] * aabb_extent / 2
        face_min = face_center - face_vertical_half_extent - face_lateral_half_extent
        face_max = face_center + face_vertical_half_extent + face_lateral_half_extent
        return th.rand(face_min.size()) * (face_max - face_min) + face_min

    # def _sample_pose_in_room(self, room: str):
    #     """
    #     Returns a pose for the robot within in the room where the robot is not in collision with anything

    #     Args:
    #         room (str): Name of room

    #     Returns:
    #         2-tuple:
    #             - 3-array: (x,y,z) Position in the world frame
    #             - 4-array: (x,y,z,w) Quaternion orientation in the world frame
    #     """
    #     # TODO(MP): Bias the sampling near the agent.
    #     for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM):
    #         _, pos = self.env.scene.get_random_point_by_room_instance(room)
    #         yaw_lo, yaw_hi = -math.pi, math.pi
    #         yaw = (th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo).item()
    #         pose = (pos[0], pos[1], yaw)
    #         if self._test_pose(pose):
    #             return pose

    #     raise ActionPrimitiveError(
    #         ActionPrimitiveError.Reason.SAMPLING_ERROR,
    #         "Could not find valid position in the given room to travel to",
    #         {"room": room}
    #     )

    def _sample_pose_with_object_and_predicate(
        self, predicate, held_obj, target_obj, world_aligned=False, near_poses=None, near_poses_threshold=None
    ):
        """
        Returns a pose for the held object relative to the target object that satisfies the predicate

        Args:
            predicate (object_states.OnTop or object_states.Inside): Relation between held object and the target object
            held_obj (StatefulObject): Object held by the robot
            target_obj (StatefulObject): Object to sample a pose relative to
            world_aligned (bool): Whether to align the current world-aligned bbox to the sampled cuboid on the target_obj
                If True, align the current world-aligned bbox; if False, align the base-link aligned bbox
            near_poses (Iterable of arrays): Poses in the world frame to sample near
            near_poses_threshold (float): The distance threshold to check if the sampled pose is near the poses in near_poses

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}

        for _ in range(m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE):
            if world_aligned:
                # Take the bbox in the world frame because we want to keep the x-y component of the current orientation
                bb_extents = held_obj.aabb_extent
                bb_center = held_obj.aabb_center
                # Compute bbox pose in the object base link frame
                bb_pos_in_base, bb_orn_in_base = T.relative_pose_transform(
                    bb_center, th.tensor([0, 0, 0, 1], dtype=th.float32), *held_obj.get_position_orientation()
                )
            else:
                _, _, bb_extents, bb_pos_in_base = held_obj.get_base_aligned_bbox()
                bb_orn_in_base = th.tensor([0, 0, 0, 1], dtype=th.float32)

            sampling_results = sample_cuboid_for_predicate(pred_map[predicate], target_obj, bb_extents)
            if sampling_results[0][0] is None:
                continue
            sampled_bb_center = sampling_results[0][0] + th.tensor([0, 0, m.PREDICATE_SAMPLING_Z_OFFSET])
            sampled_bb_orn = sampling_results[0][2]

            # Tobj_in_world @ Tbbox_in_obj = Tbbox_in_world
            # Tobj_in_world = Tbbox_in_world @ inv(Tbbox_in_obj)
            sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn)) @ T.pose_inv(
                T.pose2mat((bb_pos_in_base, bb_orn_in_base))
            )

            # Check that the pose is near one of the poses in the near_poses list if provided.
            if near_poses:
                sampled_pos = th.tensor([sampled_obj_pose[0]])
                if not th.any(th.norm(near_poses - sampled_pos, dim=1) < near_poses_threshold):
                    continue

            # Return the pose
            return T.mat2pose(sampled_obj_pose)

        # If we get here, sampling failed.
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR,
            "Could not find a position to put this object in the desired relation to the target object",
            {"target object": target_obj.name, "object in hand": held_obj.name, "relation": pred_map[predicate]},
        )

    def _validate_poses(
        self,
        candidate_poses,
        eef_pose=None,
        plan_with_open_gripper=False,
        skip_obstacle_update=False,
        visibility_constraint=False,
    ):
        """
        Determines whether the robot can reach all poses on the objects and is not in collision at the specified 2d poses

        Args:
            candidate_poses (list of arrays): Candidate 2d poses (x, y, yaw)
            eef_pose (Tuple[th.tensor, th.tensor]): target pose to reach for the default end-effector in the world frame
            plan_with_open_gripper (bool): Determines whether to plan with the gripper open even though it might be closed now.
                This is useful for the placing primitive because we will open-loop open the gripper after placing
            skip_obstacle_update (bool): Whether to skip updating the obstacles in the motion generator

        Returns:
            list of bool: Whether the default arm can reach all poses on the objects and is not in collision
                at the specified 2d poses
        """
        if isinstance(eef_pose, tuple):
            eef_pose = {self.arm: eef_pose}

        # First check collisions for all candidate poses
        candidate_joint_positions = []
        canonical_joint_positions = []
        if plan_with_open_gripper:
            current_joint_pos = self._get_joint_position_with_fingers_at_limit("upper")
        else:
            current_joint_pos = self.robot.get_joint_positions()

        for pose in candidate_poses:
            joint_pos = current_joint_pos.clone()
            joint_pos[self.robot.base_control_idx] = pose
            
            # candidate_joint_position: (sampled base pose + current torso + current arm joint positions)
            candidate_joint_positions.append(joint_pos)

            # Use the canonical torso joint positions for sampling eyes poses instead of the current torso joint positions.
            # Cause the eyes pose sampling is based on hardcoded thresholds so if the current position is such that
            # the robot is bent, the sampled eyes pose would be very wried and could lead to a lot of IK/MP failures

            # Originally (sampled base pose + canonical torso + canonical arm joint positions)
            # canonical_joint_pos = self.robot.reset_joint_pos.clone()
            # canonical_joint_pos[self.robot.base_control_idx] = pose

            # New version (sampled base pose + canonical torso + current arm joint positions)
            canonical_joint_pos = joint_pos.clone()
            canonical_joint_pos[self.robot.trunk_control_idx] = self.robot.reset_joint_pos[
                self.robot.trunk_control_idx
            ].clone()
            canonical_joint_positions.append(canonical_joint_pos)

        candidate_joint_positions = th.stack(candidate_joint_positions)
        canonical_joint_positions = th.stack(canonical_joint_positions)

        # If an object is grasped, we need to pass it to the collision checker
        # obj_in_hand = self._get_obj_in_hand()
        # attached_obj = {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link} if obj_in_hand is not None else None

        # Initial collision check of
        # - base joints: sampled base pose
        # - torso joints: canonical torso joint positions
        # - arm joints: current arm joint positions
        # This will filter out base poses that are in collision with the environment
        # During navigation, torso might move to reach the sampled eye pose, so it's meaningless
        # to check self-collision with the current torso joint positions
        # start = time.time()
        invalid_results = self._motion_generator.check_collisions(
            canonical_joint_positions,
            self_collision_check=False,
            skip_obstacle_update=skip_obstacle_update,
            attached_obj=None,  # if we pass an attaced object, leads to segmentation fault: https://github.com/mikedh/trimesh/issues/550
            attached_obj_scale=None,
        ).cpu()
        # print("initial base pose collision check", time.time() - start)

        # For each candidate that passed collision check, verify reachability
        for i in range(len(candidate_poses)):
            # print(f"candidate base pose {i}: ", candidate_poses[i])

            # # ========= remove later =============
            # temp_state = og.sim.dump_state()
            # self.robot.set_joint_positions(canonical_joint_positions[i])

            # for _ in range(50): og.sim.step()

            # breakpoint()
            # og.sim.load_state(temp_state)
            # for _ in range(20): og.sim.step()
            # # ======================================

            if invalid_results[i].item():
                # print("collision? ", invalid_results[i].item())
                self.target_eyes_pose_arr.append(None)
                continue

            if eef_pose is not None and not visibility_constraint:
                # Use the candidate joint position as the initial joint position to update the lock joints in curobo
                # This effectively moves the robot base in curobo when testing for arm reachability to the target eef_pose
                candidate_joint_position = candidate_joint_positions[i]
                if not self._target_in_reach_of_robot(
                    eef_pose, initial_joint_pos=candidate_joint_position, skip_obstacle_update=skip_obstacle_update
                ):
                    invalid_results[i] = True
                else:
                    break
                    # print("unreachable? ", invalid_results[i].item())
            elif eef_pose is not None and visibility_constraint:
                canonical_joint_position = canonical_joint_positions[i]
                if not self._target_in_reach_of_robot_and_visible(
                    eef_pose,
                    initial_joint_pos=canonical_joint_position,
                    skip_obstacle_update=skip_obstacle_update,
                ):
                    invalid_results[i] = True
                    # print("unreachable? ", invalid_results[i].item())
                # If already got a valid pose, we don't need to check for the rest of the poses
                else:
                    break

        return ~invalid_results

    def get_robot_2d_pose_from_3d_pose(self, pose):
        """
        Gets 2d pose from 3d pose

        Args:
            pose (Iterable): (position, quaternion)

        Returns:
            th.tensor: (x, y, yaw)
        """
        R.from_quat(pose[1]).as_euler("xyz")
        yaw = R.from_quat(pose[1]).as_euler("xyz")[2]
        pose2d = th.tensor([pose[0][0], pose[0][1], yaw])
        return pose2d

    def _get_robot_pose_from_2d_pose(self, pose_2d):
        """
        Gets 3d pose from 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            th.tensor: (x,y,z) Position in the world frame
            th.tensor: (x,y,z,w) Quaternion orientation in the world frame
        """
        if isinstance(self.robot, HolonomicBaseRobot):
            base_joints = self.robot.get_joint_positions()[self.robot.base_idx]
            pos = th.tensor([pose_2d[0], pose_2d[1], base_joints[2]], dtype=th.float32)
            euler_intrinsic_xyz = th.tensor([base_joints[3], base_joints[4], pose_2d[2]], dtype=th.float32)
            mat = T.euler_intrinsic2mat(euler_intrinsic_xyz)
            orn = T.mat2quat(mat)
        else:
            pos = th.tensor([pose_2d[0], pose_2d[1], 0.0], dtype=th.float32)
            orn = T.euler2quat(th.tensor([0.0, 0.0, pose_2d[2]], dtype=th.float32))
        return pos, orn

    def _world_pose_to_robot_pose(self, pose):
        """
        Converts the pose in the world frame to the robot frame

        Args:
            pose (Iterable): (pos, quat) Pose in the world frame

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        body_pose = self.robot.get_position_orientation()
        return T.relative_pose_transform(*pose, *body_pose)

    def _robot_pose_to_world_pose(self, pose):
        """
        Converts the pose in the robot frame to the world frame

        Args:
            pose (Iterable): (pos, quat) Pose in the robot frame

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        body_pose = self.robot.get_position_orientation()
        inverse_body_pose = T.invert_pose_transform(body_pose[0], body_pose[1])
        return T.relative_pose_transform(*pose, *inverse_body_pose)

    def _get_hand_pose_for_object_pose(self, desired_pose):
        """
        Gets the pose of the hand for the desired object pose

        Args:
            desired_pose (Iterable of arrays): Pose of the object in the world frame

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position of the hand in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation of the hand in the world frame
        """
        obj_in_hand = self._get_obj_in_hand()

        assert obj_in_hand is not None

        # Get the object pose & the robot hand pose
        obj_in_world = obj_in_hand.get_position_orientation()
        hand_in_world = self.robot.eef_links[self.arm].get_position_orientation()

        # Get the hand pose relative to the obj pose
        hand_in_obj = T.relative_pose_transform(*hand_in_world, *obj_in_world)

        # Now apply desired obj pose.
        desired_hand_pose = T.pose_transform(*desired_pose, *hand_in_obj)

        return desired_hand_pose

    # Function that is particularly useful for Fetch, where it gives time for the base of robot to settle due to its uneven base.
    def _settle_robot(self):
        """
        Yields a no op action for a few steps to allow the robot and physics to settle

        Returns:
            th.tensor or None: Action array for one step for the robot to do nothing
        """
        for _ in range(50):
            empty_action = self.robot.q_to_action(self.robot.get_joint_positions())
            yield self._postprocess_action(empty_action)

        for _ in range(m.MAX_STEPS_FOR_SETTLING):
            if th.norm(self.robot.get_linear_velocity()) < 0.01:
                break
            empty_action = self.robot.q_to_action(self.robot.get_joint_positions())
            yield self._postprocess_action(empty_action)
