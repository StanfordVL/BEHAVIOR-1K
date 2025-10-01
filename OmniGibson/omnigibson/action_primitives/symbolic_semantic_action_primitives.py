"""
WARNING!
A set of action primitives that work without executing low-level physics but instead teleporting
objects directly into their post-condition states. Useful for learning high-level methods.
"""

import torch as th
from aenum import IntEnum, auto
from typing import Any

from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, ActionPrimitiveErrorGroup
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.objects import DatasetObject
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.transition_rules import SlicingRule


class SymbolicSemanticActionPrimitiveSet(IntEnum):
    _init_ = "value __doc__"
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
    PLACE_NEAR_HEATING_ELEMENT = (
        auto(),
        "Place the currently grasped object near the heating element of another object.",
    )
    NAVIGATE_TO = auto(), "Navigate to an object"
    RELEASE = (
        auto(),
        "Release an object, letting it fall to the ground. You can then grasp it again, as a way of reorienting your grasp of the object.",
    )


class SymbolicSemanticActionPrimitives(StarterSemanticActionPrimitives):
    def __init__(self, env, robot):
        super().__init__(env, robot, skip_curobo_initilization=True)
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

    def apply_ref(self, primitive, *args: Any, attempts=3):
        """
        Yields action for robot to execute the primitive with the given arguments.

        Args:
            primitive (SymbolicSemanticActionPrimitiveSet): Primitive to execute
            args: Arguments for the primitive
            attempts (int): Number of attempts to make before raising an error

        Returns:
            th.tensor or None: Action array for one step for the robot tto execute the primitve or None if primitive completed

        Raises:
            ActionPrimitiveError: If primitive fails to execute
        """
        assert attempts > 0, "Must make at least one attempt"
        ctrl = self.controller_functions[primitive]

        if any(isinstance(arg, BaseRobot) for arg in args):
            raise ActionPrimitiveErrorGroup(
                [
                    ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                        "Cannot call a symbolic semantic action primitive with a robot as an argument.",
                    )
                ]
            )

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
                {"target object": obj.name},
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
            th.tensor or None: Action array for one step for the robot to grasp or None if grasp completed
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
        # yield from self._navigate_if_needed(obj)

        # Perform forced assisted grasp
        obj.set_position_orientation(position=self.robot.get_eef_position(self.arm))
        self.robot._establish_grasp(self.arm, (obj, obj.root_link), obj.get_position_orientation()[0])

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

        for arm in self.robot.arm_names:
            self.robot.release_grasp_immediately(arm=arm)
        yield from self._settle_robot()

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
                {"target object": obj.name},
            )

        if obj.states[object_states.ToggledOn].get_value() == value:
            return

        # Call the setter
        obj.states[object_states.ToggledOn].set_value(value)

        # Yield some actions
        yield from self._settle_robot()

        # Check that it actually happened
        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not toggle as expected - maybe try again",
                {
                    "target object": obj.name,
                    "is it currently toggled on": obj.states[object_states.ToggledOn].get_value(),
                },
            )

    def _place_with_predicate(self, obj, predicate, near_poses=None, near_poses_threshold=None):
        """
        Yields action for the robot to navigate to the object if needed, then to place it

        Args:
            obj (StatefulObject): Object for robot to place the object in its hand on
            predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

        Returns:
            th.tensor or None: Action array for one step for the robot to place or None if place completed
        """
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )

        # Find a spot to put it
        obj_pose = self._sample_pose_with_object_and_predicate(
            predicate, obj_in_hand, obj, near_poses=near_poses, near_poses_threshold=near_poses_threshold
        )

        # Get close, release the object.
        # yield from self._navigate_if_needed(obj, pose_on_obj=obj_pose)
        yield from self._release()

        # Actually move the object to the spot and step a bit to settle it.
        obj_in_hand.set_position_orientation(*obj_pose)
        yield from self._settle_robot()

        if not obj_in_hand.states[predicate].get_value(obj):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Failed to place object at the desired place (probably dropped). The object was still released, so you need to grasp it again to continue",
                {"dropped object": obj_in_hand.name, "target object": obj.name},
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
                {"target object": obj.name},
            )

        # Check if the target object has any particles in it
        producing_systems = {
            ps
            for ps in obj.scene.system_registry.objects
            if obj.states[object_states.ParticleSource].check_conditions_for_system(ps.name)
        }
        if not producing_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object currently is not producing any particles - try toggling it on.",
                {"target object": obj.name},
            )

        # Check that the current object can remove those particles
        if object_states.Saturated not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"object in hand": obj_in_hand.name},
            )

        supported_systems = {
            x for x in producing_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x.name)
        }

        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object only contains particles that this object cannot soak.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is producing": sorted(x.name for x in producing_systems),
                    "particles the grasped object can remove": sorted(
                        [x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                },
            )

        currently_removable_systems = {
            x
            for x in supported_systems
            if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x.name)
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
                },
            )

        # If so, remove the particles.
        for system in currently_removable_systems:
            obj_in_hand.states[object_states.Saturated].set_value(system, True)

        # Yield some actions
        yield from self._settle_robot()

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
                {"target object": obj.name},
            )

        # Check if the target object has any particles in it
        contained_systems = {
            ps for ps in obj.scene.system_registry.objects if obj.states[object_states.Contains].get_value(ps.states)
        }
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object currently does not contain any particles.",
                {"target object": obj.name},
            )

        # Check that the current object can remove those particles
        if object_states.Saturated not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object cannot soak particles.",
                {"object in hand": obj_in_hand.name},
            )

        supported_systems = {
            x for x in contained_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x.name)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object only contains particles that this object cannot soak.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object contains": sorted(x.name for x in contained_systems),
                    "particles the grasped object can remove": sorted(
                        [x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                },
            )

        currently_removable_systems = {
            x
            for x in supported_systems
            if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x.name)
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
                },
            )

        # If so, remove the particles.
        for system in currently_removable_systems:
            obj_in_hand.states[object_states.Saturated].set_value(system, True)

        # Yield some actions
        yield from self._settle_robot()

    def _wipe(self, obj):
        # Check that our current object is a particle remover
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping a wiping tool (particle remover) first to wipe an object.",
            )

        # Check that the target object is coverable
        if object_states.Covered not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not coverable by any particles, so there is no need to wipe it.",
                {"target object": obj.name},
            )

        # Check if the target object has any particles on it
        covering_systems = {
            ps for ps in obj.scene.system_registry.objects if obj.states[object_states.Covered].get_value(ps)
        }
        if not covering_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not covered by any particles.",
                {"target object": obj.name},
            )

        # Check that the current object can remove those particles
        if object_states.ParticleRemover not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The currently grasped object is not a particle remover.",
                {"object in hand": obj_in_hand.name},
            )

        supported_systems = {
            x for x in covering_systems if obj_in_hand.states[object_states.ParticleRemover].supports_system(x.name)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is covered only by particles that this cleaning tool cannot remove.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is covered by": sorted(x.name for x in covering_systems),
                    "particles the grasped object can remove": sorted(
                        [x for x in obj_in_hand.states[object_states.ParticleRemover].conditions.keys()]
                    ),
                },
            )

        currently_removable_systems = {
            x
            for x in supported_systems
            if obj_in_hand.states[object_states.ParticleRemover].check_conditions_for_system(x.name)
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
                },
            )
        # If so, remove the particles on the target object
        for system in currently_removable_systems:
            obj.states[object_states.Covered].set_value(system, False)

        # Yield some actions
        yield from self._settle_robot()

    def _cut(self, obj):
        # Check that our current object is a slicer
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping a cutting tool first to slice an object.",
            )

        if "slicer" not in obj_in_hand._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The current object is not a cutting tool.",
                {"object in hand": obj_in_hand.name},
            )

        # Check that the target object is sliceable
        if "sliceable" not in obj._abilities and "diceable" not in obj._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not sliceable or diceable.",
                {"target object": obj.name},
            )

        # Get close
        # yield from self._navigate_if_needed(obj)

        # TODO: Do some more validation
        added_obj_attrs = []
        removed_objs = []
        (slicing_rule,) = [rule for rule in obj.scene.transition_rule_api.active_rules if isinstance(rule, SlicingRule)]
        output = slicing_rule.transition({"sliceable": [obj]})
        added_obj_attrs += output.add
        removed_objs += output.remove

        obj.scene.transition_rule_api.execute_transition(added_obj_attrs=added_obj_attrs, removed_objs=removed_objs)
        yield from self._settle_robot()

    def _place_near_heating_element(self, heat_source_obj):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )

        if object_states.HeatSourceOrSink not in heat_source_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not a heat source or sink.",
                {"target object": heat_source_obj.name},
            )

        if heat_source_obj.states[object_states.HeatSourceOrSink].requires_inside:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The heat source object has no explicit heating element, it just requires the cookable object to be placed inside it.",
                {"target object": heat_source_obj.name},
            )

        # Get the position of the heat source on the thing we're placing near
        heating_element_positions = th.tensor(
            [
                link.get_position_orientation()[0]
                for link in heat_source_obj.states[object_states.HeatSourceOrSink].links.values()
            ]
        )
        heating_distance_threshold = heat_source_obj.states[object_states.HeatSourceOrSink].distance_threshold

        # Call place-with-predicate
        yield from self._place_with_predicate(
            heat_source_obj,
            object_states.OnTop,
            near_poses=heating_element_positions,
            near_poses_threshold=heating_distance_threshold,
        )

    def _wait_for_cooked(self, obj):
        # Check that the current object is cookable
        if object_states.Cooked not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Target object is not cookable.",
                {"target object": obj.name},
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
                    {"target object": obj.name},
                )

    def _navigate_to_pose(self, pose_2d):
        """
        Yields the action to navigate robot to the specified 2d pose

        Args:
            pose_2d (Iterable): (x, y, yaw) 2d pose

        Returns:
            th.tensor or None: Action array for one step for the robot to navigate or None if it is done navigating
        """
        robot_pose = self._get_robot_pose_from_2d_pose(pose_2d)
        self.robot.set_position_orientation(*robot_pose)
        yield from self._settle_robot()
