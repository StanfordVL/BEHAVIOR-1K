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
from aenum import IntEnum, auto
from math import ceil
import cv2
from matplotlib import pyplot as plt

import gym
import numpy as np
from omnigibson.transition_rules import REGISTERED_RULES, TransitionRuleAPI
from scipy.spatial.transform import Rotation, Slerp
from omnigibson.utils.constants import JointType
from pxr import PhysxSchema

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup, BaseActionPrimitiveSet
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
from omnigibson.utils.object_state_utils import sample_cuboid_for_predicate
from omnigibson.object_states.utils import get_center_extent
from omnigibson.objects import BaseObject, DatasetObject
from omnigibson.robots import BaseRobot
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.utils.motion_planning_utils import (
    plan_base_motion,
    plan_arm_motion,
    detect_robot_collision,
    detect_robot_collision_in_sim,
    set_base_and_detect_collision
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

MAX_STEPS_FOR_HAND_MOVE = 500
MAX_STEPS_FOR_HAND_MOVE_WHEN_OPENING = 30
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10

MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 200
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_HEAT_SOURCE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM = 60

PREDICATE_SAMPLING_Z_OFFSET = 0.02

ACTIVITY_RELEVANT_OBJECTS_ONLY = False

logger = logging.getLogger(__name__)


def indented_print(msg, *args, **kwargs):
    print("  " * len(inspect.stack()) + str(msg), *args, **kwargs)

class SymbolicSemanticActionPrimitiveSet(IntEnum):
    _init_ = 'value __doc__'
    GRASP = auto(), "Grasp an object"
    PLACE_ON_TOP = auto(), "Place the currently grasped object on top of another object"
    PLACE_INSIDE = auto(), "Place the currently grasped object inside another object"
    OPEN = auto(), "Open an object"
    CLOSE = auto(), "Close an object"
    TOGGLE_ON = auto(), "Toggle an object on"
    TOGGLE_OFF = auto(), "Toggle an object off"
    SOAK_UNDER = auto(), "Soak the currently grasped object under a fluid source."
    SOAK_INSIDE = auto(), "Soak the currently grasped object inside the fluid within a container."
    WIPE = auto(), "Wipe the given object with the currently grasped object."
    CUT = auto(), "Cut (slice or dice) the given object with the currently grasped object."
    PLACE_NEAR_HEATING_ELEMENT = auto(), "Place the currently grasped object near the heating element of another object."
    NAVIGATE_TO = auto(), "Navigate to an object"
    RELEASE = auto(), "Release an object, letting it fall to the ground. You can then grasp it again, as a way of reorienting your grasp of the object."

class SymbolicSemanticActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, task, scene, robot):
        super().__init__(task, scene, robot)
        self.controller_functions = {
          SymbolicSemanticActionPrimitiveSet.GRASP: self._grasp,
          SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
          SymbolicSemanticActionPrimitiveSet.PLACE_INSIDE: self._place_inside,
          SymbolicSemanticActionPrimitiveSet.OPEN: self._open,
          SymbolicSemanticActionPrimitiveSet.CLOSE: self._close,
          SymbolicSemanticActionPrimitiveSet.TOGGLE_ON: self._toggle_on,
          SymbolicSemanticActionPrimitiveSet.TOGGLE_OFF: self._toggle_off,
          SymbolicSemanticActionPrimitiveSet.SOAK_UNDER: self._soak_under,
          SymbolicSemanticActionPrimitiveSet.SOAK_INSIDE: self._soak_inside,
          SymbolicSemanticActionPrimitiveSet.WIPE: self._wipe,
          SymbolicSemanticActionPrimitiveSet.CUT: self._cut,
          SymbolicSemanticActionPrimitiveSet.PLACE_NEAR_HEATING_ELEMENT: self._place_near_heating_element,
          SymbolicSemanticActionPrimitiveSet.NAVIGATE_TO: self._navigate_to_obj,
          SymbolicSemanticActionPrimitiveSet.RELEASE: self._release,
        }
        self.arm = self.robot.default_arm
        self.robot_model = self.robot.model_name
        self.robot_base_mass = self.robot._links["base_link"].mass

        self.robot_copy = StarterSemanticActionPrimitives._load_robot_copy(robot)

        if self.robot_model == "Tiago":
            self._setup_tiago()

    # Disable grasping frame for Tiago robot (Should be cleaned up in the future)
    def _setup_tiago(self):
        for link in self.robot.links.values():
            for mesh in link.collision_meshes.values():
                if "grasping_frame" in link.prim_path:
                    mesh.collision_enabled = False

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
            [gym.spaces.Discrete(self.num_objects), gym.spaces.Discrete(len(SymbolicSemanticActionPrimitiveSet))]
        )

    def get_action_from_primitive_and_object(self, primitive: SymbolicSemanticActionPrimitiveSet, obj: BaseObject):
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
        action = SymbolicSemanticActionPrimitiveSet(action_idx)
        return self.controller_functions[action](target_obj)
    
    def apply_ref(self, prim, *args, attempts=3):
        """
        Yields action for robot to execute the primitive with the given arguments.

        Args:
            prim (SymbolicSemanticActionPrimitiveSet): Primitive to execute
            args: Arguments for the primitive
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
                yield from ctrl(*args)
                success = True
            except ActionPrimitiveError as e:
                errors.append(e)

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
        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand().name},
            )
        
        if object_states.Open not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not openable.",
                {"target object": obj.name}
            )

        # Don't do anything if the object is already closed and we're trying to close.
        if should_open == obj.states[object_states.Open].get_value():
            return
        
        # Set the value
        obj.states[object_states.Open].set_value(should_open)

        # Settle
        yield from self._settle_robot()

        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not open or close as expected. Maybe try again",
                {"target object": obj.name, "is it currently open": obj.states[object_states.Open].get_value()},
            )

    def _grasp(self, obj: DatasetObject):
        """
        Yields action for the robot to navigate to object if needed, then to grasp it

        Args:
            DatasetObject: Object for robot to grasp
        
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
            
        # Get close
        yield from self._navigate_if_needed(obj)

        # Perform forced assisted grasp
        obj.set_position(self.robot.get_eef_position(self.arm))
        self.robot._ag_data[self.arm] = (obj, obj.root_link)
        self.robot._establish_grasp(self.arm, self.robot._ag_data[self.arm], obj.get_position())

        # Execute for a moment
        yield from self._settle_robot()

        # Verify
        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        if self._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )
        
    def _release(self):
        if not self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot release an object if you're not already holding an object",
            )
        
        self.robot.release_grasp_immediately()
        yield from self._settle_robot()

    def _place_on_top(self, obj):
        """
        Yields action for the robot to navigate to the object if needed, then to place it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on
        
        Returns:
            np.array or None: Action array for one step for the robot to place or None if grasp completed
        """
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def _place_inside(self, obj):
        yield from self._place_with_predicate(obj, object_states.Inside)

    def _toggle_on(self, obj):
        yield from self._toggle(obj, True)

    def _toggle_off(self, obj):
        yield from self._toggle(obj, False)

    def _toggle(self, obj, value):
        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot toggle an object while holding an object",
                {"object in hand": self._get_obj_in_hand()},
            )
        
        if object_states.ToggledOn not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not toggleable.",
                {"target object": obj.name}
            )

        if obj.states[object_states.ToggledOn].get_value() == value:
            return

        # Call the setter
        obj.states[object_states.ToggledOn].set_value(value)

        # Check that it actually happened
        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {"target object": obj.name, "is it currently toggled on": obj.states[object_states.ToggledOn].get_value()}
            )

    def _place_with_predicate(self, obj, predicate, near_poses=None, near_poses_threshold=None):
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
        
        # Find a spot to put it
        obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj, near_poses=[], near_poses_threshold=None)

        # Get close, release the object.
        yield from self._navigate_if_needed(obj, pose_on_obj=obj_pose)
        yield from self._release()

        # Actually move the object to the spot and step a bit to settle it.
        obj_in_hand.set_position_orientation(*obj_pose)
        yield from self._settle_robot()

        if not obj_in_hand.states[predicate].get_value(obj):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Failed to place object at the desired place (probably dropped). The object was still released, so you need to grasp it again to continue",
                {"dropped object": obj_in_hand.name, "target object": obj.name}
            )
        
    def _soak_under(self, obj):
        # Check that our current object is a particle remover
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping a soakable object first."
            )
        
        # Check that the target object is a particle source
        if object_states.ParticleSource not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not a particle source, so you can not soak anything under it.",
                {"target object": obj.name}
            )

        # Check if the target object has any particles in it
        producing_systems = {ps for ps in self.scene.systems if obj.states[object_states.ParticleSource].check_conditions_for_system(ps)}
        if not producing_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object currently is not producing any particles - try toggling it on.",
                {"target object": obj.name}
            )

        # Check that the current object can remove those particles
        if object_states.Saturated not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"object in hand": obj_in_hand.name}
            )
        
        supported_systems = {
            x for x in producing_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object only contains particles that this object cannot soak.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is producing": sorted(x.name for x in producing_systems),
                    "particles the grasped object can remove": sorted([x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()])
                }
            )
        
        currently_removable_systems = {
            x for x in supported_systems if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x)
        }
        if not currently_removable_systems:
            # TODO: This needs to be far more descriptive.
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered by some particles that this object can normally soak, but needs to be in a different state to do so (e.g. toggled on, soaked by another fluid first, etc.).",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is producing": sorted(x.name for x in producing_systems),
                }
            )

        # If so, remove the particles.
        for system in currently_removable_systems:
            obj_in_hand.states[object_states.Saturated].set_value(system, True)


    def _soak_inside(self, obj):
        # Check that our current object is a particle remover
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping a soakable object first."
            )
        
        # Check that the target object is fillable
        if object_states.Contains not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not fillable by particles, so you can not soak anything in it.",
                {"target object": obj.name}
            )

        # Check if the target object has any particles in it
        contained_systems = {ps for ps in self.scene.systems if obj.states[object_states.Contains].get_value(ps.states)}
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object currently does not contain any particles.",
                {"target object": obj.name}
            )

        # Check that the current object can remove those particles
        if object_states.Saturated not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"object in hand": obj_in_hand.name}
            )
        
        supported_systems = {
            x for x in contained_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object only contains particles that this object cannot soak.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object contains": sorted(x.name for x in contained_systems),
                    "particles the grasped object can remove": sorted([x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()])
                }
            )
        
        currently_removable_systems = {
            x for x in supported_systems if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x)
        }
        if not currently_removable_systems:
            # TODO: This needs to be far more descriptive.
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered by some particles that this object can normally soak, but needs to be in a different state to do so (e.g. toggled on, soaked by another fluid first, etc.).",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object contains": sorted(x.name for x in contained_systems),
                }
            )

        # If so, remove the particles.
        for system in currently_removable_systems:
            obj_in_hand.states[object_states.Saturated].set_value(system, True)


    def _wipe(self, obj):
        # Check that our current object is a particle remover
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping a wiping tool (particle remover) first to wipe an object."
            )
        
        # Check that the target object is coverable
        if object_states.Covered not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not coverable by any particles, so there is no need to wipe it.",
                {"target object": obj.name}
            )

        # Check if the target object has any particles on it
        covering_systems = {ps for ps in self.scene.systems if obj.states[object_states.Covered].get_value(ps.states)}
        if not covering_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not covered by any particles.",
                {"target object": obj.name}
            )

        # Check that the current object can remove those particles
        if object_states.ParticleRemover not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object is not a particle remover.",
                {"object in hand": obj_in_hand.name}
            )
        
        supported_systems = {
            x for x in covering_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered only by particles that this cleaning tool cannot remove.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is covered by": sorted(x.name for x in covering_systems),
                    "particles the grasped object can remove": sorted([x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()])
                }
            )
        
        currently_removable_systems = {
            x for x in supported_systems if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x)
        }
        if not currently_removable_systems:
            # TODO: This needs to be far more descriptive.
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered by some particles that this cleaning tool can normally remove, but needs to be in a different state to do so (e.g. toggled on, soaked by another fluid first, etc.).",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is covered by": sorted(x.name for x in covering_systems),
                }
            )

        # If so, remove the particles.
        for system in currently_removable_systems:
            obj_in_hand.states[object_states.Covered].set_value(system, False)

    def _cut(self, obj):
        # Check that our current object is a slicer
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping a cutting tool first to slice an object."
            )
        
        if "slicer" not in obj_in_hand._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The current object is not a cutting tool.",
                {"object in hand": obj_in_hand.name}
            )

        # Check that the target object is sliceable
        if "sliceable" not in obj._abilities and "diceable" not in obj._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not sliceable or diceable.",
                {"target object": obj.name}
            )
        
        # Get close
        yield from self._navigate_if_needed(obj)

        # TODO: Do some more validation
        added_obj_attrs = []
        removed_objs = []
        output = REGISTERED_RULES["SlicingRule"].transition({"sliceable": [obj]})
        added_obj_attrs += output.add
        removed_objs += output.remove

        TransitionRuleAPI.execute_transition(added_obj_attrs=added_obj_attrs, removed_objs=removed_objs)

    def _place_near_heating_element(self, heat_source_obj):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "You need to be grasping an object first to place it somewhere."
            )
        
        if object_states.HeatSourceOrSink not in heat_source_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "The target object is not a heat source or sink.", {"target object": heat_source_obj.name}
            )
        
        if heat_source_obj.states[object_states.HeatSourceOrSink].requires_inside:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The heat source object has no explicit heating element, it just requires the cookable object to be placed inside it.",
                {"target object": heat_source_obj.name}
            )

        # Get the position of the heat source on the thing we're placing near
        heating_element_positions = np.array([link.get_position() for link in heat_source_obj.states[object_states.HeatSourceOrSink].links.values()])
        heating_distance_threshold = heat_source_obj.states[object_states.HeatSourceOrSink].distance_threshold

        # Call place-with-predicate
        yield from self._place_with_predicate(heat_source_obj, object_states.OnTop, near_poses=heating_element_positions, near_poses_threshold=heating_distance_threshold)

    def _wait_for_cooked(self, obj):
        # Check that the current object is cookable
        if object_states.Cooked not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "Target object is not cookable.",
                {"target object": obj.name}
            )
        
        # Keep waiting as long as the thing is warming up.
        prev_temp = obj.states[object_states.Temperature].get_value()
        while not obj.states[object_states.Cooked].get_value():
            # Pass some time
            for _ in range(10):
                yield from self._empty_action()

            # Check that we are still heating up
            new_temp = obj.states[object_states.Temperature].get_value()
            if new_temp - prev_temp < 1e-2:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.PRE_CONDITION_ERROR,
                    "Target object is not currently heating up.",
                    {"target object": obj.name}
                )
  
    def _target_in_reach_of_robot(self, target_pose):
        """
        Determines whether the eef for the robot can reach the target pose in the world frame

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for the pose for the eef
        
        Returns:
            bool: Whether eef can reach the target pose
        """
        relative_target_pose = self._get_pose_in_robot_frame(target_pose)
        return self._target_in_reach_of_robot_relative(relative_target_pose)
    
    def _target_in_reach_of_robot_relative(self, relative_target_pose):
        """
        Determines whether eef for the robot can reach the target pose where the target pose is in the robot frame

        Args:
            target_pose (Iterable of array): Position and orientation arrays in an iterable for pose for the eef
        
        Returns:
            bool: Whether eef can the reach target pose
        """
        return np.linalg.norm(relative_target_pose[0]) < 2.0
     
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
            if controller.control_type == ControlType.POSITION and len(joint_idx) == len(action_idx):
                action[action_idx] = self.robot.get_joint_positions()[joint_idx]

        return action

    def _navigate_to_pose(self, pose_2d):
        """
        Yields the action to navigate robot to the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose 

        Returns:
            np.array or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        robot_pose = self._get_robot_pose_from_2d_pose(pose_2d)
        self.robot.set_position_orientation(*robot_pose)
        yield from self._settle_robot()

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
            pose_on_obj = [pos_on_obj, np.array([0, 0, 0, 1])]

        with UndoableContext(self.robot, self.robot_copy, "simplified") as context:
            obj_rooms = obj.in_rooms if obj.in_rooms else [self.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
            for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
                distance = np.random.uniform(0.0, 1.0)
                yaw = np.random.uniform(-np.pi, np.pi)
                pose_2d = np.array(
                    [pose_on_obj[0][0] + distance * np.cos(yaw), pose_on_obj[0][1] + distance * np.sin(yaw), yaw + np.pi]
                )

                # Check room
                # if self.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                #     indented_print("Candidate position is in the wrong room.")
                #     continue

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

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj, near_poses=None, near_poses_threshold=None):
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

        if near_poses:
            assert near_poses_threshold, "A near-pose distance threshold must be provided if near_poses is provided."

        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE):
            _, _, bb_extents, bb_center_in_base = held_obj.get_base_aligned_bbox()
            sampling_results = sample_cuboid_for_predicate(pred_map[predicate], target_obj, bb_extents)
            if sampling_results[0][0] is None:
                continue
            sampled_bb_center = sampling_results[0][0] + np.array([0, 0, PREDICATE_SAMPLING_Z_OFFSET])
            sampled_bb_orn = sampling_results[0][2]

            # Get the object pose by subtracting the offset
            sampled_obj_pose = T.pose2mat((sampled_bb_center, sampled_bb_orn)) @ T.pose_inv(T.pose2mat((bb_center_in_base, [0, 0, 0, 1])))

            # Check that the pose is near one of the poses in the near_poses list if provided.
            if near_poses:
                sampled_pos = np.array([sampled_obj_pose[0]])
                if not np.any(np.linalg.norm(near_poses - sampled_pos, axis=1) < near_poses_threshold):
                    continue

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
    
    # Function that is particularly useful for Fetch, where it gives time for the base of robot to settle due to its uneven base.
    def _settle_robot(self):
        """
        Yields a no op action for a few steps to allow the robot and physics to settle

        Returns:
            np.array or None: Action array for one step for the robot to do nothing
        """
        yield from [self._empty_action() for _ in range(100)]
        while np.linalg.norm(self.robot.get_linear_velocity()) > 0.01:
            yield self._empty_action()
