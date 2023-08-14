"""
WARNING!
The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example.
It currently only works with BehaviorRobot with its JointControllers set to absolute mode.
See provided behavior_robot_mp_behavior_task.yaml config file for an example. See examples/action_primitives for
runnable examples.
"""
import copy
import inspect
import logging
import random
from enum import IntEnum
from math import ceil
import cv2
from matplotlib import pyplot as plt

import gym
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from omnigibson.utils.constants import JointType
from pxr import PhysxSchema

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup, BaseActionPrimitiveSet
from omnigibson.utils.object_state_utils import sample_cuboid_for_predicate
from omnigibson.object_states.utils import get_center_extent
from omnigibson.objects.object_base import BaseObject
from omnigibson.robots import BaseRobot
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    set_base_and_detect_collision,
    detect_robot_collision_in_sim
)

import omnigibson.utils.transform_utils as T
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.grasping_planning_utils import (
    get_grasp_poses_for_object_sticky,
    get_grasp_position_for_open
)
from omnigibson.controllers.controller_base import ControlType
from omnigibson.prims import CollisionGeomPrim
from omnigibson.utils.control_utils import FKSolver

from omni.usd.commands import CopyPrimCommand, CreatePrimCommand
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf

import os
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject

DEFAULT_BODY_OFFSET_FROM_FLOOR = 0.05

KP_LIN_VEL = 0.3
KP_ANGLE_VEL = 0.2

MAX_CARTESIAN_HAND_STEP = 0.002
MAX_STEPS_FOR_HAND_MOVE = 500
MAX_STEPS_FOR_HAND_MOVE_WHEN_OPENING = 30
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10
MAX_STEPS_FOR_WAYPOINT_NAVIGATION = 200
MAX_ATTEMPTS_FOR_OPEN_CLOSE = 20

MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 200
MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM = 60

BIRRT_SAMPLING_CIRCLE_PROBABILITY = 0.5
HAND_SAMPLING_DOMAIN_PADDING = 1  # Allow 1m of freedom around the sampling range.
PREDICATE_SAMPLING_Z_OFFSET = 0.02
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

class RobotCopy():
    def __init__(self):
        self.prims = {}
        self.meshes = {}
        self.relative_poses = {}
        self.links_relative_poses = {}
        self.reset_pose = {
            "original": ([0, 0, -5.0], [0, 0, 0, 1]),
            "simplified": ([5, 0, -5.0], [0, 0, 0, 1]),
        }

class UndoableContext(object):
    def __init__(self, robot, robot_copy, robot_copy_type="original"):
        self.robot = robot
        self.robot_copy = robot_copy
        self.robot_copy_type = robot_copy_type if robot_copy_type in robot_copy.prims.keys() else "original"
        self.disabled_collision_pairs_dict = {}

    def __enter__(self):
        self._assemble_robot_copy()
        self._construct_disabled_collision_pairs()
        return self 

    def __exit__(self, *args):
        self._set_prim_pose(self.robot_copy.prims[self.robot_copy_type], self.robot_copy.reset_pose[self.robot_copy_type])

    def _assemble_robot_copy(self):
        fk_descriptor = "combined" if "combined" in self.robot.robot_arm_descriptor_yamls else self.robot.default_arm
        self.fk_solver = FKSolver(
            robot_description_path=self.robot.robot_arm_descriptor_yamls[fk_descriptor],
            robot_urdf_path=self.robot.urdf_path,
        )
        arm_links = self.robot.manipulation_link_names
        joint_combined_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[fk_descriptor]])
        joint_pos = np.array(self.robot.get_joint_positions()[joint_combined_idx])
        link_poses = self.fk_solver.get_link_poses(joint_pos, arm_links)

        # Set position of robot copy root prim
        # self._set_prim_pose(self.robot_copy.prims[self.robot_copy_type], self.robot.get_position_orientation())
        self._set_prim_pose(self.robot_copy.prims[self.robot_copy_type], ([0, 0, 0], [0, 0, 0, 1]))
        # Assemble robot meshes
        for link_name, meshes in self.robot_copy.meshes[self.robot_copy_type].items():
            for mesh_name, copy_mesh in meshes.items():
                # Skip grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                if "grasping_frame" in link_name:
                    continue
                # Set poses of meshes relative to the robot to construct the robot
                link_pose = link_poses[link_name] if link_name in arm_links else self.robot_copy.links_relative_poses[self.robot_copy_type][link_name]
                mesh_copy_pose = T.pose_transform(*link_pose, *self.robot_copy.relative_poses[self.robot_copy_type][link_name][mesh_name])
                self._set_prim_pose(copy_mesh, mesh_copy_pose)

    def _set_prim_pose(self, prim, pose):
        translation = Gf.Vec3d(*np.array(pose[0], dtype=float))
        prim.GetAttribute("xformOp:translate").Set(translation)
        orientation = np.array(pose[1], dtype=float)[[3, 0, 1, 2]]
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation)) 

    def _construct_disabled_collision_pairs(self):
        robot_meshes_copy = self.robot_copy.meshes[self.robot_copy_type]

        # Filter out collision pairs of meshes part of the same link
        for meshes in robot_meshes_copy.values():
            for mesh in meshes.values():
                self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] = [m.GetPrimPath().pathString for m in meshes.values()]

        # Filter out collision pairs of meshes part of disabled collision pairs
        for pair in self.robot.primitive_disabled_collision_pairs:
            link_1 = pair[0]
            link_2 = pair[1]
            if link_1 in robot_meshes_copy.keys() and link_2 in robot_meshes_copy.keys():
                for mesh in robot_meshes_copy[link_1].values():
                    self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [m.GetPrimPath().pathString for m in robot_meshes_copy[link_2].values()]

                for mesh in robot_meshes_copy[link_2].values():
                    self.disabled_collision_pairs_dict[mesh.GetPrimPath().pathString] += [m.GetPrimPath().pathString for m in robot_meshes_copy[link_1].values()]
        
        # Filter out colliders all robot copy meshes should ignore
        disabled_colliders = []

        # Disable original robot colliders so copy can't collide with it
        disabled_colliders += [link.prim_path for link in self.robot.links.values()]
        filter_categories = ["floors"]
        for obj in og.sim.scene.objects:
            if obj.category in filter_categories:
                disabled_colliders += [link.prim_path for link in obj.links.values()]

        # Disable object in hand
        obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm] 
        if obj_in_hand is not None:
            disabled_colliders += [link.prim_path for link in obj_in_hand.links.values()]

        for colliders in self.disabled_collision_pairs_dict.values():
            colliders += disabled_colliders

class StarterSemanticActionPrimitiveSet(IntEnum):
    GRASP = 0
    PLACE_ON_TOP = 1
    PLACE_INSIDE = 2
    OPEN = 3
    CLOSE = 4
    NAVIGATE_TO = 5  # For mostly debugging purposes.
    RELEASE = 6  # For reorienting grasp


class StarterSemanticActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, task, scene, robot, teleport=False):
        logger.warning(
            "The StarterSemanticActionPrimitive is a work-in-progress and is only provided as an example. "
            "It currently only works with BehaviorRobot with its JointControllers set to absolute mode. "
            "See provided behavior_robot_mp_behavior_task.yaml config file for an example. "
            "See examples/action_primitives for runnable examples."
        )
        super().__init__(task, scene, robot)
        self.controller_functions = {
            StarterSemanticActionPrimitiveSet.GRASP: self._grasp,
            StarterSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
            StarterSemanticActionPrimitiveSet.PLACE_INSIDE: self._place_inside,
            StarterSemanticActionPrimitiveSet.OPEN: self._open,
            StarterSemanticActionPrimitiveSet.CLOSE: self._close,
            StarterSemanticActionPrimitiveSet.NAVIGATE_TO: self._navigate_to_obj,
            StarterSemanticActionPrimitiveSet.RELEASE: self._execute_release,
        }
        self.arm = self.robot.default_arm
        self.robot_model = self.robot.model_name
        self.robot_base_mass = self.robot._links["base_link"].mass
        self.teleport = teleport
        self._tracking_object = None

        self.robot_copy = self._load_robot_copy(robot)

        if self.robot_model == "Tiago":
            self._setup_tiago()

    # Disable grasping frame for Tiago robot (Should be cleaned up in the future)
    def _setup_tiago(self):
        for link in self.robot.links.values():
            for mesh in link.collision_meshes.values():
                if "grasping_frame" in link.prim_path:
                    mesh.collision_enabled = False

    @staticmethod
    def _load_robot_copy(robot):
        robot_copy = RobotCopy()

        robots_to_copy = {
            "original": {
                "robot": robot,
                "copy_path": "/World/robot_copy"
            }
        }
        if hasattr(robot, 'simplified_mesh_usd_path'):
            simplified_robot = { 
                "robot": USDObject("simplified_copy", robot.simplified_mesh_usd_path),
                "copy_path": "/World/simplified_robot_copy"         
            }
            robots_to_copy['simplified'] = simplified_robot

        for robot_type, rc in robots_to_copy.items():
            copy_robot = None
            copy_robot_meshes = {}
            copy_robot_meshes_relative_poses = {}
            copy_robot_links_relative_poses = {}

            # Create prim under which robot meshes are nested and set position
            CreatePrimCommand("Xform", rc['copy_path']).do()
            copy_robot = get_prim_at_path(rc['copy_path'])
            reset_pose = robot_copy.reset_pose[robot_type]
            translation = Gf.Vec3d(*np.array(reset_pose[0], dtype=float))
            copy_robot.GetAttribute("xformOp:translate").Set(translation)
            orientation = np.array(reset_pose[1], dtype=float)[[3, 0, 1, 2]]
            copy_robot.GetAttribute("xformOp:orient").Set(Gf.Quatd(*orientation)) 

            robot_to_copy = None
            if robot_type == "simplified":
                robot_to_copy =  rc['robot']
                og.sim.import_object(robot_to_copy)
            else:
                robot_to_copy = rc['robot']

            # Copy robot meshes
            for link in robot_to_copy.links.values():
                link_name = link.prim_path.split("/")[-1]
                for mesh_name, mesh in link.collision_meshes.items():
                    split_path = mesh.prim_path.split("/")
                    # Do not copy grasping frame (this is necessary for Tiago, but should be cleaned up in the future)
                    if "grasping_frame" in link_name:
                        continue

                    copy_mesh_path = rc['copy_path'] + "/" + link_name
                    copy_mesh_path += f"_{split_path[-1]}" if split_path[-1] != "collisions" else ""
                    CopyPrimCommand(mesh.prim_path, path_to=copy_mesh_path).do()
                    copy_mesh = get_prim_at_path(copy_mesh_path)
                    relative_pose = T.relative_pose_transform(*mesh.get_position_orientation(), *link.get_position_orientation())
                    if link_name not in copy_robot_meshes.keys():
                        copy_robot_meshes[link_name] = {mesh_name: copy_mesh}
                        copy_robot_meshes_relative_poses[link_name] = {mesh_name: relative_pose}
                    else:
                        copy_robot_meshes[link_name][mesh_name] = copy_mesh
                        copy_robot_meshes_relative_poses[link_name][mesh_name] = relative_pose

                copy_robot_links_relative_poses[link_name] = T.relative_pose_transform(*link.get_position_orientation(), *robot.get_position_orientation())
            
            if robot_type == "simplified":
                og.sim.remove_object(robot_to_copy)

            robot_copy.prims[robot_type] = copy_robot
            robot_copy.meshes[robot_type] = copy_robot_meshes
            robot_copy.relative_poses[robot_type] = copy_robot_meshes_relative_poses
            robot_copy.links_relative_poses[robot_type] = copy_robot_links_relative_poses

        og.sim.step()
        return robot_copy

    def get_action_space(self):
        if ACTIVITY_RELEVANT_OBJECTS_ONLY:
            assert isinstance(self.task, BehaviorTask), "Activity relevant objects can only be used for BEHAVIOR tasks"
            self.addressable_objects = sorted(set(self.task.object_scope.values()), key=lambda obj: obj.name)
        else:
            self.addressable_objects = sorted(set(self.scene.objects_by_name.values()), key=lambda obj: obj.name)

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
        return self.controller_functions[action](target_obj)
    
    def apply_ref(self, prim, *args, track_object=False, attempts=3):
        """
        Yields action for robot to execute the primitive with the given arguments.

        Args:
            prim (StarterSemanticActionPrimitiveSet): Primitive to execute
            args: Arguments for the primitive
            track_object (bool): Whether to track object while executing the primitive
            attempts (int): Number of attempts to make before raising an error
        
        Returns:
            np.array or None: Action array for one step for the robot tto execute the primitve or None if primitive completed
        
        Raises:
            ActionPrimitiveError: If primitive fails to execute
        """
        assert attempts > 0, "Must make at least one attempt"
        ctrl = self.controller_functions[prim]

        errors = []
        for _ in range(attempts):
            # Attempt
            success = False
            try:
                self._tracking_object = args[0] if track_object and args else None
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
                # Make sure we retract the arm after every step
                yield from self._reset_hand()
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

    def _open(self, obj):
        yield from self._open_or_close(obj, True)

    def _close(self, obj):
        yield from self._open_or_close(obj, False)

    def _open_or_close(self, obj, should_open):
        reset_eef_pose = None

        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand()},
            )

        # Open the hand first
        yield from self._execute_release()

        for _ in range(MAX_ATTEMPTS_FOR_OPEN_CLOSE):
            try:
                print("attempt")
                self.counter = 0
                grasp_data = get_grasp_position_for_open(self.robot, obj, should_open)
                if grasp_data is None:
                    # We were trying to do something but didn't have the data.
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.SAMPLING_ERROR,
                        "Could not sample grasp position for target object",
                        {"target object": obj.name},
                    )

                grasp_pose, target_poses, object_direction, joint_info, grasp_required, yaw_change = grasp_data
                if yaw_change < 0.1:
                    return

                # Prepare data for the approach later.
                approach_pos = grasp_pose[0] + object_direction * OPEN_GRASP_APPROACH_DISTANCE
                approach_pose = (approach_pos, grasp_pose[1])

                # self.set_marker(grasp_pose)
                # self.set_marker(approach_pose)
                # for target_pose in target_poses:
                #     print(target_pose)
                #     self.set_marker(target_pose)
                # for i in range(10000):
                #     og.sim.step()

                # If the grasp pose is too far, navigate
                yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)

                yield from self._move_hand(grasp_pose)

                # We can pre-grasp in sticky grasping mode.
                yield from self._execute_grasp()

                # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
                # It's okay if we can't go all the way because we run into the object.
                indented_print("Performing grasp approach for open")

                yield from self._navigate_if_needed(obj, pose_on_obj=approach_pose)  #, check_joint=check_joint)
                
                self.set_marker(approach_pose)
                yield from self._move_hand_direct_cartesian(approach_pose, ignore_failure=False, stop_on_contact=True)

                # Step once to update
                yield self._empty_action()

                # if grasp_required:
                #     if self._get_obj_in_hand() is None:
                #         raise ActionPrimitiveError(
                #             ActionPrimitiveError.Reason.EXECUTION_ERROR,
                #             "Could not grasp the target object to open or close. Try again",
                #             {"target object": obj.name},
                #         )
                # from IPython import embed; embed()
                
                for i, target_pose in enumerate(target_poses):
                    # print(target_pose)
                    reset_eef_pose = target_poses[i+1]

                    # self.set_marker(target_pose)
                    # from IPython import embed; embed()
                    yield from self._move_hand_direct_cartesian(target_pose, ignore_failure=False)
                # from IPython import embed; embed()

                # Moving to target pose often fails. Let's get the hand to apply the correct actions for its current pos
                # This prevents the hand from jerking into its desired position when we do a release.
                yield from self._move_hand_direct_cartesian(
                    self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True
                )
            except ActionPrimitiveError as e:
                # print(_)
                print(e)
                # Let go - we do not want to be holding anything after return of primitive.
                yield from self._execute_release()
                if reset_eef_pose is not None:
                    yield from self._move_hand_direct_cartesian(reset_eef_pose, ignore_failure=True)


        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Despite executing the planned trajectory, the object did not open or close as expected. Maybe try again",
                {"target object": obj.name, "is it currently open": obj.states[object_states.Open].get_value()},
            )
        
    def set_marker(self, pose):
        self.markers[self.counter].set_position_orientation(*pose)
        self.counter += 1

    def _grasp(self, obj):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            StatefulObject: Object for robot to grasp

        Returns:
            np.array or None: Action array for one step for the robot to grasp or None if grasp completed
        """

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
            
        if self._get_obj_in_hand() != obj:
            # Open the hand first
            yield from self._execute_release()

            # Allow grasping from suboptimal extents if we've tried enough times.
            grasp_poses = get_grasp_poses_for_object_sticky(obj)
            grasp_pose, object_direction = random.choice(grasp_poses)

            # Prepare data for the approach later.
            approach_pos = grasp_pose[0] + object_direction * GRASP_APPROACH_DISTANCE
            approach_pose = (approach_pos, grasp_pose[1])
            # If the grasp pose is too far, navigate.
            yield from self._navigate_if_needed(obj, pose_on_obj=grasp_pose)
            yield from self._move_hand(grasp_pose)

            # We can pre-grasp in sticky grasping mode.
            yield from self._execute_grasp()

            # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            # It's okay if we can't go all the way because we run into the object.
            indented_print("Performing grasp approach")
            yield from self._move_hand_direct_cartesian(approach_pose, stop_on_contact=True)

            # Step once to update
            yield self._empty_action()

            if self._get_obj_in_hand() is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                    "Grasp completed, but no object detected in hand after executing grasp",
                    {"target object": obj.name},
                )
            
            yield from self._reset_hand()

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
            np.array or None: Action array for one step for the robot to place or None if place completed
        """
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def _place_inside(self, obj):
        """
        Yields action for the robot to navigate to the object if needed, then to place an object in it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on
        
        Returns:
            np.array or None: Action array for one step for the robot to place or None if place completed
        """
        yield from self._place_with_predicate(obj, object_states.Inside)

    def _toggle_on(self, obj):
        yield from self._toggle(obj, True)

    def _toggle_off(self, obj):
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

        yield from self._move_hand(desired_hand_pose)

        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {"target object": obj.name, "is it currently toggled on": obj.states[object_states.ToggledOn].get_value()}
            )

    def _place_with_predicate(self, obj, predicate):
        """
        Yields action for the robot to navigate to the object if needed, then to place it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on
            predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

        Returns:
            np.array or None: Action array for one step for the robot to place or None if place completed
        """
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping an object first to place it somewhere."
            )
        
        obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
        hand_pose = self._get_hand_pose_for_object_pose(obj_pose)
        yield from self._navigate_if_needed(obj, pose_on_obj=hand_pose)
        yield from self._move_hand(hand_pose)
        yield from self._execute_release()
        yield from self._settle_robot()

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not release object - the object is still in your hand",
                {"object": self._get_obj_in_hand().name}
            )

        if not obj_in_hand.states[predicate].get_value(obj):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Failed to place object at the desired place (probably dropped). The object was still released, so you need to grasp it again to continue",
                {"dropped object": obj_in_hand.name, "target object": obj.name}
            )

    def _convert_cartesian_to_joint_space(self, target_pose):
        """
        Gets joint positions for the arm so eef is at the target pose

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose for the eef
        
        Returns:
            2-tuple
                - np.array or None: Joint positions to reach target pose or None if impossible to reach target pose
                - np.array: Indices for joints in the robot
        """
        relative_target_pose = self._get_pose_in_robot_frame(target_pose)
        joint_pos, control_idx = self._ik_solver_cartesian_to_joint_space(relative_target_pose)
        if joint_pos is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not find joint positions for target pose. You cannot reach it. Try again for a new pose"
            )
        return joint_pos, control_idx
    
    def _target_in_reach_of_robot(self, target_pose):
        """
        Determines whether the eef for the robot can reach the target pose in the world frame

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for the pose for the eef
        
        Returns:
            bool: Whether eef can reach the target pose
        """
        relative_target_pose = self._get_pose_in_robot_frame(target_pose)
        joint_pos, _ = self._ik_solver_cartesian_to_joint_space(relative_target_pose)
        return False if joint_pos is None else True
    
    def _target_in_reach_of_robot_relative(self, relative_target_pose):
        """
        Determines whether eef for the robot can reach the target pose where the target pose is in the robot frame

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose for the eef
        
        Returns:
            bool: Whether eef can the reach target pose
        """
        joint_pos, _ = self._ik_solver_cartesian_to_joint_space(relative_target_pose)
        return False if joint_pos is None else True

    def _ik_solver_cartesian_to_joint_space(self, relative_target_pose):
        """
        Get joint positions for the arm so eef is at the target pose where the target pose is in the robot frame

        Args:
            relative_target_pose (Iterable of array): Position and orientation arrays in an iterable for pose in the robot frame
        
        Returns:
            2-tuple
                - np.array or None: Joint positions to reach target pose or None if impossible to reach the target pose
                - np.array: Indices for joints in the robot
        """
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
        
        if joint_pos is None:
            return None, control_idx
        else:
            return joint_pos, control_idx

    def _move_hand(self, target_pose):
        """
        Yields action for the robot to move hand so the eef is in the target pose using the planner

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose

        Returns:
            np.array or None: Action array for one step for the robot to move hand or None if its at the target pose
        """
        yield from self._settle_robot()
        joint_pos, control_idx = self._convert_cartesian_to_joint_space(target_pose)
        yield from self._move_hand_joint(joint_pos, control_idx)

    def _move_hand_joint(self, joint_pos, control_idx):
        """
        Yields action for the robot to move arm to reach the specified joint positions using the planner

        Args:
            joint_pos (np.array): Joint positions for the arm
            control_idx (np.array): Indices of the joints to move
        
        Returns:
            np.array or None: Action array for one step for the robot to move arm or None if its at the joint positions
        """
        if self.teleport:
            # Teleport the robot to the joint state
            self.robot.set_joint_positions(joint_pos, control_idx)

            # Yield a bunch of no-ops to give the robot time to settle.
            yield from self._settle_robot()
        else:
            with UndoableContext(self.robot, self.robot_copy, "original") as context:
                plan = plan_arm_motion(
                    robot=self.robot,
                    end_conf=joint_pos,
                    context=context
                )
            # plan = self._add_linearly_interpolated_waypoints(plan, 0.1)
            if plan is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "There is no accessible path from where you are to the desired joint position. Try again"
                )
            
            # Follow the plan to navigate.
            indented_print("Plan has %d steps", len(plan))
            for i, joint_pos in enumerate(plan):
                indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
                yield from self._move_hand_direct_joint(joint_pos, control_idx)

    def _add_linearly_interpolated_waypoints(self, plan, max_inter_dist):
        """
        Adds waypoints to the plan so the distance between values in the plan never exceeds the max_inter_dist.

        Args:
            plan (Array of arrays): Planned path
            max_inter_dist (float): Maximum distance between values in the plan
        
        Returns:
            Array of arrays: Planned path with additional waypoints
        """
        plan = np.array(plan)
        interpolated_plan = []
        for i in range(len(plan) - 1):
            max_diff = max(plan[i+1] - plan[i])
            num_intervals = ceil(max_diff / max_inter_dist)
            interpolated_plan += np.linspace(plan[i], plan[i+1], num_intervals, endpoint=False).tolist()
        interpolated_plan.append(plan[-1].tolist())
        return interpolated_plan

    def _move_hand_direct_joint(self, joint_pos, control_idx, stop_on_contact=False, max_steps_for_hand_move=MAX_STEPS_FOR_HAND_MOVE, ignore_failure=False):
        """
        Yields action for the robot to move its arm to reach the specified joint positions by directly actuating with no planner

        Args:
            joint_pos (np.array): Array of joint positions for the arm
            control_idx (np.array): Indices of the joints to move
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            max_steps_for_hand_move (int): The max number of steps that should be taken to reach the final joint positions
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions
        
        Returns:
            np.array or None: Action array for one step for the robot to move arm or None if its at the joint positions
        """
        action = self._empty_action()
        controller_name = "arm_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = joint_pos
        action = self._overwrite_head_action(action, self._tracking_object) if self._tracking_object is not None else action

        for i in range(max_steps_for_hand_move):
            current_joint_pos = self.robot.get_joint_positions()[control_idx]
            diff_joint_pos = np.absolute(np.array(current_joint_pos) - np.array(joint_pos))
            print(max(diff_joint_pos))
            val = 0.005
            # if ignore_failure:
            #     val = 0.02
            if max(diff_joint_pos) < val:
                return
            if stop_on_contact and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
                return
            print(i)
            yield action

        if not ignore_failure:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Your hand was obstructed from moving to the desired joint position"
            )

    def _move_hand_direct_cartesian(self, target_pose, stop_on_contact=False, ignore_failure=False):
        """
        Yields action for the robot to move its arm to reach the specified target pose by moving the eef along a line in cartesian
        space from its current pose

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose
            stop_on_contact (boolean): Determines whether to stop move once an object is hit
            ignore_failure (boolean): Determines whether to throw error for not reaching final joint positions
        
        Returns:
            np.array or None: Action array for one step for the robot to move arm or None if its at the target pose
        """
        # To make sure that this happens in a roughly linear fashion, we will divide the trajectory
        # into 1cm-long pieces
        start_pos, start_orn = self.robot.eef_links[self.arm].get_position_orientation()
        travel_distance = np.linalg.norm(target_pose[0] - start_pos)
        num_poses = np.max([2, int(travel_distance / MAX_CARTESIAN_HAND_STEP) + 1])
        pos_waypoints = np.linspace(start_pos, target_pose[0], num_poses)

        # Also interpolate the rotations
        combined_rotation = Rotation.from_quat(np.array([start_orn, target_pose[1]]))
        slerp = Slerp([0, 1], combined_rotation)
        orn_waypoints = slerp(np.linspace(0, 1, num_poses))
        quat_waypoints = [x.as_quat() for x in orn_waypoints]

        # Joint positions
        joint_space_data = [self._convert_cartesian_to_joint_space(waypoint) for waypoint in zip(pos_waypoints, quat_waypoints)]
        joints = list(self.robot.joints.values())
        
        for joint_pos, control_idx in joint_space_data:
            # Check if the movement can be done roughly linearly.
            current_joint_positions = self.robot.get_joint_positions()[control_idx]

            failed_joints = []
            for joint_idx, target_joint_pos, current_joint_pos in zip(control_idx, joint_pos, current_joint_positions):
                if np.abs(target_joint_pos - current_joint_pos) > np.rad2deg(45):
                    failed_joints.append(joints[joint_idx].joint_name)

            if failed_joints:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "You cannot reach the target position in a straight line - it requires rotating your arm which might cause collisions. You might need to get closer and retry",
                    {"failed joints": failed_joints}
                )

            # Otherwise, move the joint
            yield from self._move_hand_direct_joint(joint_pos, control_idx, stop_on_contact=stop_on_contact, ignore_failure=ignore_failure)

            # Also decide if we can stop early.
            current_pos, current_orn = self.robot.eef_links[self.arm].get_position_orientation()
            pos_diff = np.linalg.norm(np.array(current_pos) - np.array(target_pose[0]))
            orn_diff = (Rotation.from_quat(current_orn) * Rotation.from_quat(target_pose[1]).inv()).magnitude()
            if pos_diff < 0.001 and orn_diff < np.deg2rad(0.1):
                return
            
            if stop_on_contact and detect_robot_collision_in_sim(self.robot, ignore_obj_in_hand=False):
                return

        if not ignore_failure:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Your hand was obstructed from moving to the desired world position"
            )

    def _execute_grasp(self):
        """
        Yields action for the robot to grasp

        Returns:
            np.array or None: Action array for one step for the robot to grasp or None if its done grasping
        """
        action = self._empty_action()
        action = self._overwrite_head_action(action, self._tracking_object) if self._tracking_object is not None else action
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = -1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield action

        # Do nothing for a bit so that AG can trigger.
        # for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
        #     yield self._empty_action()

    def _execute_release(self):
        """
        Yields action for the robot to release its grasp

        Returns:
            np.array or None: Action array for one step for the robot to release or None if its done releasing
        """
        action = self._empty_action()
        action = self._overwrite_head_action(action, self._tracking_object) if self._tracking_object is not None else action
        controller_name = "gripper_{}".format(self.arm)
        action[self.robot.controller_action_idx[controller_name]] = 1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # Otherwise, keep applying the action!
            yield action

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "An object was still detected in your hand after executing release",
                {"object in hand": self._get_obj_in_hand().name},
            )
        
    def _overwrite_head_action(self, action, obj):
        assert self.robot_model == "Tiago", "Tracking object with camera is currently only supported for Tiago"
        head_q = self._get_head_goal_q(obj)
        head_idx = self.robot.controller_action_idx["camera"]
        action[head_idx] = head_q
        return action

    def _get_head_goal_q(self, obj):
        """
        Get goal joint positions for head to look at an object of interest,
        If the object cannot be seen, return the current head joint positions.
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
        obj_pose = obj.get_position_orientation()
        obj_in_base = T.relative_pose_transform(*obj_pose, *robot_pose)

        # compute angle between base and object in xy plane (parallel to floor)
        theta = np.arctan2(obj_in_base[0][1], obj_in_base[0][0])
        
        # if it is possible to get object in view, compute both head joint positions
        if head1_joint_limits[0] < theta < head1_joint_limits[1]:
            head1_joint_goal = theta
            
            # compute angle between base and object in xz plane (perpendicular to floor)
            head2_pose = self.robot.links["head_2_link"].get_position_orientation()
            head2_in_base = T.relative_pose_transform(*head2_pose, *robot_pose)

            phi = np.arctan2(obj_in_base[0][2] - head2_in_base[0][2], obj_in_base[0][0])
            if head2_joint_limits[0] < phi < head2_joint_limits[1]:
                head2_joint_goal = phi

        # if not possible to look at object, return default head joint positions
        else:
            default_head_pos = self._get_reset_joint_pos()[self.robot.controller_action_idx["camera"]]
            head1_joint_goal = default_head_pos[0]
            head2_joint_goal = default_head_pos[1]

        return [head1_joint_goal, head2_joint_goal]
        
    def _empty_action(self):
        """
        No op action

        Returns:
            np.array or None: Action array for one step for the robot to do nothing
        """
        action = np.zeros(self.robot.action_dim)
        for name, controller in self.robot._controllers.items():
            joint_idx = controller.dof_idx
            action_idx = self.robot.controller_action_idx[name]
            if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx) and not controller.use_delta_commands:
                action[action_idx] = self.robot.get_joint_positions()[joint_idx]

        return action

    def _reset_hand(self):
        """
        Yields action to move the hand to the position optimal for executing subsequent action primitives

        Returns:
            np.array or None: Action array for one step for the robot to reset its hand or None if it is done resetting
        """
        control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.arm]])
        reset_pose = self._get_reset_joint_pos()[control_idx]
        indented_print("Resetting hand")
        try:
            yield from self._move_hand_joint(reset_pose, control_idx)
        except ActionPrimitiveError:
            indented_print("Could not do a planned reset of the hand - probably obj_in_hand collides with body")
            yield from self._move_hand_direct_joint(reset_pose, control_idx, ignore_failure=True)
    
    def _get_reset_joint_pos(self):
        reset_pose_fetch = np.array(
            [
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                -1.0,
                0.0,  # head
                -1.0,
                1.53448,
                2.2,
                0.0,
                1.36904,
                1.90996,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

        # reset_pose_tiago = np.array([
        #     -1.78029833e-04,  
        #     3.20231302e-05, 
        #     -1.85759447e-07, 
        #     -1.16488536e-07,
        #     4.55182843e-08,  
        #     2.36128806e-04,  
        #     0.15,  
        #     0.94,
        #     -1.1,  
        #     0.0, 
        #     -0.9,  
        #     1.47,
        #     0.0,  
        #     2.1,  
        #     2.71,  
        #     1.5,
        #     1.71,  
        #     1.3, 
        #     -1.57, 
        #     -1.4,
        #     1.39,  
        #     0.0,  
        #     0.0,  
        #     0.045,
        #     0.045,
        #     0.045,
        #     0.045,
        # ])
        
        reset_pose_tiago = np.array([
            -1.78029833e-04,  
            3.20231302e-05, 
            -1.85759447e-07,
            0.0, 
            -0.2,
            0.0,  
            0.1, 
            -6.10000000e-01,
            -1.10000000e+00,  
            0.00000000e+00, 
            -1.10000000e+00,  
            1.47000000e+00,
            0.00000000e+00,  
            8.70000000e-01,  
            2.71000000e+00,  
            1.50000000e+00,
            1.71000000e+00, 
            -1.50000000e+00, 
            -1.57000000e+00,  
            4.50000000e-01,
            1.39000000e+00,  
            0.00000000e+00,  
            0.00000000e+00,  
            4.50000000e-02,
            4.50000000e-02,  
            4.50000000e-02,  
            4.50000000e-02
        ])
        return reset_pose_tiago if self.robot_model == "Tiago" else reset_pose_fetch
    
    def _navigate_to_pose(self, pose_2d):
        """
        Yields the action to navigate robot to the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            np.array or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        if self.teleport:
            robot_pose = self._get_robot_pose_from_2d_pose(pose_2d)
            self.robot.set_position_orientation(*robot_pose)
            yield from self._settle_robot()
        else:
            with UndoableContext(self.robot, self.robot_copy, "original") as context:
                plan = plan_base_motion(
                    robot=self.robot,
                    end_conf=pose_2d,
                    context=context,
                )

            if plan is None:
                # TODO: Would be great to produce a more informative error.
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "Could not make a navigation plan to get to the target position"
                )

            self._draw_plan(plan)
            # Follow the plan to navigate.
            indented_print("Plan has %d steps", len(plan))
            for i, pose_2d in enumerate(plan):
                indented_print("Executing navigation plan step %d/%d", i + 1, len(plan))
                low_precision = True if i < len(plan) - 1 else False
                yield from self._navigate_to_pose_direct(pose_2d, low_precision=low_precision)

    def _draw_plan(self, plan):
        SEARCHED = []
        trav_map = self.scene._trav_map
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

    def _navigate_if_needed(self, obj, pose_on_obj=None, **kwargs):
        """
        Yields action to navigate the robot to be in range of the object if it not in the range

        Args:
            obj (StatefulObject): Object for the robot to be in range of
            pose_on_obj (Iterable): (pos, quat) Pose

        Returns:
            np.array or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        if pose_on_obj is not None:
            if self._target_in_reach_of_robot(pose_on_obj):
                # No need to navigate.
                return
        elif self._target_in_reach_of_robot(obj.get_position_orientation()):
            return

        yield from self._navigate_to_obj(obj, pose_on_obj=pose_on_obj, **kwargs)

    def _navigate_to_obj(self, obj, pose_on_obj=None, **kwargs):
        """
        Yields action to navigate the robot to be in range of the pose

        Args:
            obj (StatefulObject): object to be in range of
            pose_on_obj (Iterable): (pos, quat) pose

        Returns:
            np.array or None: Action array for one step for the robot to navigate in range or None if it is done navigating
        """
        pose = self._sample_pose_near_object(obj, pose_on_obj=pose_on_obj, **kwargs)
        yield from self._navigate_to_pose(pose)

    def _navigate_to_pose_direct(self, pose_2d, low_precision=False):
        """
        Yields action to navigate the robot to the 2d pose without planning

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose
            low_precision (bool): Determines whether to navigate to the pose within a large range (low precision) or small range (high precison)

        Returns:
            np.array or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        dist_threshold = LOW_PRECISION_DIST_THRESHOLD if low_precision else DEFAULT_DIST_THRESHOLD
        angle_threshold = LOW_PRECISION_ANGLE_THRESHOLD if low_precision else DEFAULT_ANGLE_THRESHOLD
            
        end_pose = self._get_robot_pose_from_2d_pose(pose_2d)
        body_target_pose = self._get_pose_in_robot_frame(end_pose)
        
        while np.linalg.norm(body_target_pose[0][:2]) > dist_threshold:
            if self.robot_model == "Tiago":
                action = self._empty_action()
                direction_vec = body_target_pose[0][:2] / (np.linalg.norm(body_target_pose[0][:2]) * 5)
                base_action = [direction_vec[0], direction_vec[1], 0.0]
                action[self.robot.controller_action_idx["base"]] = base_action
                action = self._overwrite_head_action(action, self._tracking_object) if self._tracking_object is not None else action
                yield action
            else:
                diff_pos = end_pose[0] - self.robot.get_position()
                intermediate_pose = (end_pose[0], T.euler2quat([0, 0, np.arctan2(diff_pos[1], diff_pos[0])]))
                body_intermediate_pose = self._get_pose_in_robot_frame(intermediate_pose)
                diff_yaw = T.wrap_angle(T.quat2euler(body_intermediate_pose[1])[2])
                if abs(diff_yaw) > DEFAULT_ANGLE_THRESHOLD:
                    yield from self._rotate_in_place(intermediate_pose, angle_threshold=DEFAULT_ANGLE_THRESHOLD)
                else:
                    action = self._empty_action()
                    base_action = [KP_LIN_VEL, 0.0]
                    action[self.robot.controller_action_idx["base"]] = base_action
                    yield action

            body_target_pose = self._get_pose_in_robot_frame(end_pose)

        # Rotate in place to final orientation once at location
        yield from self._rotate_in_place(end_pose, angle_threshold=angle_threshold)

    def _rotate_in_place(self, end_pose, angle_threshold = DEFAULT_ANGLE_THRESHOLD):
        """
        Yields action to rotate the robot to the 2d end pose

        Args:
            end_pose (Iterable): (x, y, yaw) 2d pose
            angle_threshold (float): The angle difference between the robot's current and end pose that determines when the robot is done rotating

        Returns:
            np.array or None: Action array for one step for the robot to rotate or None if it is done rotating
        """
        body_target_pose = self._get_pose_in_robot_frame(end_pose)
        diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
        while abs(diff_yaw) > angle_threshold:
            action = self._empty_action()

            direction = -1.0 if diff_yaw < 0.0 else 1.0
            ang_vel = KP_ANGLE_VEL * direction

            base_action = [0.0, 0.0, ang_vel] if self.robot_model == "Tiago" else [0.0, ang_vel]
            action[self.robot.controller_action_idx["base"]] = base_action
            
            action = self._overwrite_head_action(action, self._tracking_object) if self._tracking_object is not None else action
            yield action

            body_target_pose = self._get_pose_in_robot_frame(end_pose)
            diff_yaw = T.wrap_angle(T.quat2euler(body_target_pose[1])[2])
            
        yield self._empty_action()
            
    def _sample_pose_near_object(self, obj, pose_on_obj=None, **kwargs):
        """
        Returns a 2d pose for the robot within in the range of the object and where the robot is not in collision with anything

        Args:
            obj (StatefulObject): Object to sample a 2d pose near
            pose_on_obj (Iterable of arrays or None): The pose to sample near

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        if pose_on_obj is None:
            pos_on_obj = self._sample_position_on_aabb_face(obj)
            pose_on_obj = np.array([pos_on_obj, [0, 0, 0, 1]])

        with UndoableContext(self.robot, self.robot_copy, "original") as context:
            obj_rooms = obj.in_rooms if obj.in_rooms else [self.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
            for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                distance = np.random.uniform(0.0, 1.0)
                yaw = np.random.uniform(-np.pi, np.pi)
                avg_arm_workspace_range = np.mean(self.robot.arm_workspace_range[self.arm])
                pose_2d = np.array(
                    [pose_on_obj[0][0] + distance * np.cos(yaw), pose_on_obj[0][1] + distance * np.sin(yaw), yaw + np.pi - avg_arm_workspace_range]
                )
                # Check room
                if self.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                    indented_print("Candidate position is in the wrong room.")
                    continue

                if not self._test_pose(pose_2d, context, pose_on_obj=pose_on_obj, **kwargs):
                    continue

                return pose_2d

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position near object."
            )

    @staticmethod
    def _sample_position_on_aabb_face(target_obj):
        """
        Returns a position on the axis-aligned bounding box (AABB) faces of the target object.

        Args:
            target_obj (StatefulObject): Object to sample a position on

        Returns:
            3-array: (x,y,z) Position in the world frame
        """
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
        """
        Returns a pose for the robot within in the room where the robot is not in collision with anything

        Args:
            room (str): Name of room

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        # TODO(MP): Bias the sampling near the agent.
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM):
            _, pos = self.scene.get_random_point_by_room_instance(room)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose = (pos[0], pos[1], yaw)
            if self._test_pose(pose):
                return pose

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR,
            "Could not find valid position in the given room to travel to",
            {"room": room}
        )

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj):
        """
        Returns a pose for the held object relative to the target object that satisfies the predicate

        Args:
            predicate (object_states.OnTop or object_states.Inside): Relation between held object and the target object
            held_obj (StatefulObject): Object held by the robot
            target_obj (StatefulObject): Object to sample a pose relative to

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}

        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE):
            _, _, bb_extents, bb_center_in_base = held_obj.get_base_aligned_bbox()
            sampling_results = sample_cuboid_for_predicate(pred_map[predicate], target_obj, bb_extents)
            sampled_bb_center = sampling_results[0][0] + np.array([0, 0, PREDICATE_SAMPLING_Z_OFFSET])
            sampled_bb_orn = sampling_results[0][2]

            # Get the object pose by subtracting the offset
            sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn)) @ T.pose_inv(T.pose2mat((bb_center_in_base, [0, 0, 0, 1])))

            # Return the pose
            return T.mat2pose(sampled_obj_pose)

        # If we get here, sampling failed.
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR,
            "Could not find a position to put this object in the desired relation to the target object",
            {"target object": target_obj.name, "object in hand": held_obj.name, "relation": pred_map[predicate]},
        )

    def _test_pose(self, pose_2d, context, pose_on_obj=None, check_joint=None):
        """
        Determines whether the robot can reach the pose on the object and is not in collision at the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose
            context (Context): Undoable context reference
            pose_on_obj (Iterable of arrays): Pose on the object in the world frame

        Returns:
            bool: True if the robot is in a valid pose, False otherwise
        """
        pose = self._get_robot_pose_from_2d_pose(pose_2d)
        if pose_on_obj is not None:
            relative_pose = T.relative_pose_transform(*pose_on_obj, *pose)
            if not self._target_in_reach_of_robot_relative(relative_pose):
                return False

        if set_base_and_detect_collision(context, pose):
            indented_print("Candidate position failed collision test.")
            return False
        return True

    @staticmethod
    def _get_robot_pose_from_2d_pose(pose_2d):
        """
        Gets 3d pose from 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        pos = np.array([pose_2d[0], pose_2d[1], DEFAULT_BODY_OFFSET_FROM_FLOOR])
        orn = T.euler2quat([0, 0, pose_2d[2]])
        return pos, orn

    def _get_pose_in_robot_frame(self, pose):
        """
        Converts the pose in the world frame to the robot frame

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            2-tuple:
                - 3-array: (x,y,z) Position in the world frame
                - 4-array: (x,y,z,w) Quaternion orientation in the world frame
        """
        body_pose = self.robot.get_position_orientation()
        return T.relative_pose_transform(*pose, *body_pose)

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
            np.array or None: Action array for one step for the robot to do nothing
        """
        yield from [self._empty_action() for _ in range(10)]
        while np.linalg.norm(self.robot.get_linear_velocity()) > 0.01:
            yield self._empty_action()
