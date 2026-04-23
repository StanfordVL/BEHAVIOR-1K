import random

import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.constants import JointType
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ArticulatedObjectViewAPI

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Joint position threshold before a joint is considered open.
# Should be a number in the range [0, 1] which will be transformed
# to a position in the joint's min-max range.
m.JOINT_THRESHOLD_BY_TYPE = {
    JointType.JOINT_REVOLUTE: 0.05,
    JointType.JOINT_PRISMATIC: 0.05,
}
m.OPEN_SAMPLING_ATTEMPTS = 5

m.METADATA_FIELD = "openable_joint_ids"
m.BOTH_SIDES_METADATA_FIELD = "openable_both_sides"


def _compute_joint_threshold(joint, joint_direction):
    """
    Computes the joint threshold for opening and closing

    Args:
        joint (JointPrim): Joint to calculate threshold for
        joint_direction (int): If 1, assumes opening direction is positive angle change. Otherwise,
            assumes opening direction is negative angle change.

    Returns:
        3-tuple:

            - float: Joint value at which closed <--> open
            - float: Extreme joint value for being opened
            - float: Extreme joint value for being closed
    """
    global m
    # Convert fractional threshold to actual joint position.
    f = m.JOINT_THRESHOLD_BY_TYPE[joint.joint_type]
    closed_end = joint.lower_limit if joint_direction == 1 else joint.upper_limit
    open_end = joint.upper_limit if joint_direction == 1 else joint.lower_limit
    threshold = (1 - f) * closed_end + f * open_end
    return threshold, open_end, closed_end


def _is_in_range(position, threshold, range_end):
    """
    Calculates whether a joint's position @position is in its opening / closing range

    Args:
        position (float): Joint value
        threshold (float): Joint value at which closed <--> open
        range_end (float): Extreme joint value for being opened / closed

    Returns:
        bool: Whether the joint position is past @threshold in the direction of @range_end
    """
    # Note that we are unable to do an actual range check here because the joint positions can actually go
    # slightly further than the denoted joint limits.
    return position > threshold if range_end > threshold else position < threshold


def _get_relevant_joints(obj):
    """
    Grabs the relevant joints for object @obj

    Args:
        obj (StatefulObject): Object to grab relevant joints for

    Returns:
        3-tuple:
            - bool: If True, check open/closed state for objects whose joints can switch positions
            - list of JointPrim: Relevant joints for determining whether @obj is open or closed
            - list of int: Joint directions for each corresponding relevant joint
    """
    global m

    default_both_sides = False
    default_relevant_joints = list(obj.joints.values())
    # 1 means the open direction corresponds to positive joint angle change and -1 means the opposite
    default_joint_directions = [1] * len(default_relevant_joints)

    if not hasattr(obj, "metadata") or obj.metadata is None:
        log.debug("No openable joint metadata found for object %s" % obj.name)
        return default_both_sides, default_relevant_joints, default_joint_directions

    # Get joint IDs and names from metadata annotation. If not, return default values.
    if m.METADATA_FIELD not in obj.metadata or len(obj.metadata[m.METADATA_FIELD]) == 0:
        log.debug(f"No openable joint metadata found for object {obj.name}")
        return default_both_sides, default_relevant_joints, default_joint_directions

    both_sides = obj.metadata[m.BOTH_SIDES_METADATA_FIELD] if m.BOTH_SIDES_METADATA_FIELD in obj.metadata else False
    joint_metadata = obj.metadata[m.METADATA_FIELD].items()

    # The joint metadata is in the format of [(joint_id, joint_name), ...] for legacy annotations and
    # [(joint_id, joint_name, joint_direction), ...] for direction-annotated objects.
    joint_names = [m[1] for m in joint_metadata]
    joint_directions = [m[2] if len(m) > 2 else 1 for m in joint_metadata]

    relevant_joints = []
    for key in joint_names:
        assert key in obj.joints, f"Unexpected joint name from Open metadata for object {obj.name}: {key}"
        relevant_joints.append(obj.joints[key])

    assert all(joint.joint_type in m.JOINT_THRESHOLD_BY_TYPE.keys() for joint in relevant_joints)

    return both_sides, relevant_joints, joint_directions


class Open(TensorizedValueState, BooleanStateMixin):
    """
    Tensorized Open state.

    VALUES shape: (S, O) — bool True = open, False = closed.
    """

    # (S, O, max_dof) bool — True for DOF columns that correspond to openable joints
    OPENABLE_MASK = None

    # (S, O, max_dof) float — threshold and direction per DOF for side=+1
    THRESHOLDS_S1 = None
    DIRECTIONS_S1 = None

    # (S, O, max_dof) float — threshold and direction per DOF for side=-1 (0 for non-both_sides)
    THRESHOLDS_S2 = None
    DIRECTIONS_S2 = None

    # (S, O) bool — whether each object uses both-sides open logic
    BOTH_SIDES = None

    # (S*O,) int64 — pre-built row indices into ArticulatedObjectViewAPI._JOINT_POSITIONS
    # rows [obj_idx*S .. (obj_idx+1)*S-1] = scenes 0..S-1 for object obj_idx
    OBJ_IDXES_IN_ARTICULATION_VIEW = None

    @classproperty
    def value_name(cls):
        return "open"

    @classproperty
    def value_type(cls):
        return th.bool

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this object has any openable joints
        return (
            (True, None)
            if obj.n_joints > 0
            else (False, f"No relevant joints for Open state found for object {obj.name}")
        )

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        def _find_articulated_joints(prim):
            for child in prim.GetChildren():
                child_type = child.GetTypeName().lower()
                if "joint" in child_type and "fixed" not in child_type:
                    return True
                for gchild in child.GetChildren():
                    gchild_type = gchild.GetTypeName().lower()
                    if "joint" in gchild_type and "fixed" not in gchild_type:
                        return True
            return False

        # Check whether this object has any openable joints
        return (
            (True, None)
            if _find_articulated_joints(prim=prim)
            else (False, f"No relevant joints for Open state found for asset prim {prim.GetName()}")
        )

    @classmethod
    def initialize_view(cls):
        super().initialize_view()  # builds OBJ_IDXS, IDX_OBJS, VALUES (S, O)

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        if O == 0:
            cls.OPENABLE_MASK = th.zeros((S, 0, 0), dtype=th.bool, device="cuda")
            cls.THRESHOLDS_S1 = th.zeros((S, 0, 0), device="cuda")
            cls.DIRECTIONS_S1 = th.zeros((S, 0, 0), device="cuda")
            cls.THRESHOLDS_S2 = th.zeros((S, 0, 0), device="cuda")
            cls.DIRECTIONS_S2 = th.zeros((S, 0, 0), device="cuda")
            cls.BOTH_SIDES = th.zeros((S, 0), dtype=th.bool, device="cuda")
            cls.OBJ_IDXES_IN_ARTICULATION_VIEW = th.zeros((0, 0), dtype=th.long, device="cuda")
            return

        max_dof = ArticulatedObjectViewAPI.get_max_dof()

        cls.OPENABLE_MASK = th.zeros((S, O, max_dof), dtype=th.bool, device="cuda")
        cls.THRESHOLDS_S1 = th.zeros((S, O, max_dof), device="cuda")
        cls.DIRECTIONS_S1 = th.zeros((S, O, max_dof), device="cuda")
        cls.THRESHOLDS_S2 = th.zeros((S, O, max_dof), device="cuda")
        cls.DIRECTIONS_S2 = th.zeros((S, O, max_dof), device="cuda")
        cls.BOTH_SIDES = th.zeros((S, O), dtype=th.bool, device="cuda")

        # Fill per (scene_idx, obj_idx) so objects with different models in different scenes
        # can have different joint counts and thresholds.
        for scene_idx, scene in enumerate(cls.IDX_OBJS):
            for obj_idx in range(O):
                obj = scene[obj_idx]
                if obj is None or obj.joints is None:
                    continue  # obj not initialized yet
                both_sides, relevant_joints, joint_directions = _get_relevant_joints(obj)
                cls.BOTH_SIDES[scene_idx, obj_idx] = both_sides
                for joint, direction in zip(relevant_joints, joint_directions):
                    for dof_col in joint.dof_indices:
                        cls.OPENABLE_MASK[scene_idx, obj_idx, dof_col] = True
                        for side, threshold_attr, direction_attr in [
                            (1, cls.THRESHOLDS_S1, cls.DIRECTIONS_S1),
                            (-1, cls.THRESHOLDS_S2, cls.DIRECTIONS_S2),
                        ]:
                            threshold, open_end, _ = _compute_joint_threshold(joint, direction * side)
                            threshold_attr[scene_idx, obj_idx, dof_col] = threshold
                            direction_attr[scene_idx, obj_idx, dof_col] = 1.0 if open_end > threshold else -1.0

        # Pre-build (S, O) row index into ArticulatedObjectViewAPI._JOINT_POSITIONS — built once, reused every step
        cls.OBJ_IDXES_IN_ARTICULATION_VIEW = th.zeros(S, O, dtype=th.long, device="cuda")
        for _, obj_idx in cls.OBJ_IDXS.items():
            for scene_idx, scene in enumerate(cls.IDX_OBJS):
                obj = scene[obj_idx]
                if obj is None:
                    continue
                row = ArticulatedObjectViewAPI.get_view_row(obj.articulation_root_path)
                cls.OBJ_IDXES_IN_ARTICULATION_VIEW[scene_idx, obj_idx] = row

    @classmethod
    def _update_values(cls, values):
        O = values.shape[1]

        # Early return
        if O == 0:
            return

        # (S, O, max_dof)
        pos = ArticulatedObjectViewAPI.get_articulation_positions(cls.OBJ_IDXES_IN_ARTICULATION_VIEW)

        open_s1 = ((pos - cls.THRESHOLDS_S1) * cls.DIRECTIONS_S1 > 0) & cls.OPENABLE_MASK  # (S, O, max_dof)
        any_s1 = open_s1.any(dim=2)  # (S, O)

        open_s2 = ((pos - cls.THRESHOLDS_S2) * cls.DIRECTIONS_S2 > 0) & cls.OPENABLE_MASK
        any_s2 = open_s2.any(dim=2)

        is_open = th.where(cls.BOTH_SIDES, any_s1 & any_s2, any_s1)  # (S, O)
        values.copy_(is_open)

    def _get_value(self):
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(self.VALUES_CPU[s, obj_idx])

    def _set_value(self, new_value, fully=False):
        """
        Set the openness state, either to a random joint position satisfying the new value, or fully open/closed.

        Args:
            new_value (bool): The new value for the openness state of the object.
            fully (bool): Whether the object should be fully opened/closed (e.g. all relevant joints to 0/1).

        Returns:
            bool: A boolean indicating the success of the setter. Failure may happen due to unannotated objects.
        """
        both_sides, relevant_joints, joint_directions = _get_relevant_joints(self.obj)
        if not relevant_joints:
            return False

        # The "sides" variable is used to check open/closed state for objects whose joints can switch
        # positions. These objects are annotated with the both_sides annotation and the idea is that switching
        # the directions of *all* of the joints results in a similarly valid checkable state. We want our object to be
        # open from *both* of the two sides, and I was too lazy to implement the logic for this without rejection
        # sampling, so that's what we do.
        # TODO: Implement a sampling method that's guaranteed to be correct, ditch the rejection method.
        sides = [1, -1] if both_sides else [1]

        for _ in range(m.OPEN_SAMPLING_ATTEMPTS):
            side = random.choice(sides)

            joints_to_set = list(zip(relevant_joints, joint_directions))

            # All joints are relevant if we are closing, but if we are opening let's sample a subset.
            if new_value and not fully:
                num_to_open = th.randint(1, len(relevant_joints) + 1, (1,)).item()
                random_indices = th.randperm(len(relevant_joints))[:num_to_open]
                joints_to_set = [joints_to_set[i] for i in random_indices]

            # Go through the relevant joints & set random positions.
            for joint, joint_direction in joints_to_set:
                threshold, open_end, closed_end = _compute_joint_threshold(joint, joint_direction * side)

                # Get the range
                if new_value:
                    joint_range = (threshold, open_end)
                else:
                    joint_range = (threshold, closed_end)

                if fully:
                    joint_pos = joint_range[1]
                else:
                    # Convert the range to the format numpy accepts.
                    low = min(joint_range)
                    high = max(joint_range)

                    # Sample a position.
                    joint_pos = (th.rand(1) * (high - low) + low).item()

                # Save sampled position.
                joint.set_pos(joint_pos)

            # Check success from joint positions (VALUES is stale here)
            # TODO(vector): When cache invalidation works, we can just call _update_values() here instead of recomputing everything.
            sides_open = []
            for check_side in sides:
                joint_thresholds = (
                    _compute_joint_threshold(joint, direction * check_side)
                    for joint, direction in zip(relevant_joints, joint_directions)
                )
                joint_positions = [joint.get_state()[0] for joint in relevant_joints]
                joint_openness = (
                    _is_in_range(pos, threshold, open_end)
                    for pos, (threshold, open_end, _) in zip(joint_positions, joint_thresholds)
                )
                sides_open.append(any(joint_openness))

            if all(sides_open) == new_value:
                # Write result into VALUES immediately so get_value() is correct before next global_update()
                super()._set_value(new_value)
                return True

        # We exhausted our attempts and could not find a working sample.
        return False

    # We don't need to load / save anything since the joints are saved elsewhere
