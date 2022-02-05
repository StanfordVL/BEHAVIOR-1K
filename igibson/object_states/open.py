import random



# from igibson.external.pybullet_tools import utils
from igibson.object_states.object_state_base import BooleanState, CachingEnabledObjectState

# Joint position threshold before a joint is considered open.
# Should be a number in the range [0, 1] which will be transformed
# to a position in the joint's min-max range.
_JOINT_THRESHOLD_BY_TYPE = {
    "RevoluteJoint": 0.05,#p.JOINT_REVOLUTE: 0.05,
    "PrismaticJoint": 0.05,#p.JOINT_PRISMATIC: 0.05,
}
_OPEN_SAMPLING_ATTEMPTS = 5

_METADATA_FIELD = "openable_joint_ids"
_BOTH_SIDES_METADATA_FIELD = "openable_both_sides"


def _compute_joint_threshold(joint_info, joint_direction):
    # Convert fractional threshold to actual joint position.
    f = _JOINT_THRESHOLD_BY_TYPE[joint_info.jointType]
    closed_end = joint_info.jointLowerLimit if joint_direction == 1 else joint_info.jointUpperLimit
    open_end = joint_info.jointUpperLimit if joint_direction == 1 else joint_info.jointLowerLimit
    threshold = (1 - f) * closed_end + f * open_end
    return threshold, open_end, closed_end


def _is_in_range(position, threshold, range_end):
    # Note that we are unable to do an actual range check here because the joint positions can actually go
    # slightly further than the denoted joint limits.
    if range_end > threshold:
        return position > threshold
    else:
        return position < threshold


def _get_relevant_joints(obj):
    if not hasattr(obj, "metadata"):
        return None, None, None

    both_sides = obj.metadata[_BOTH_SIDES_METADATA_FIELD] if _BOTH_SIDES_METADATA_FIELD in obj.metadata else False

    # Get joint IDs and names from metadata annotation. If object doesn't have the openable metadata,
    # we stop here and return Open=False.
    if _METADATA_FIELD not in obj.metadata:
        print("No openable joint metadata found for object %s" % obj.name)
        return None, None, None

    joint_metadata = obj.metadata[_METADATA_FIELD]

    # The joint metadata is in the format of [(joint_id, joint_name), ...] for legacy annotations and
    # [(joint_id, joint_name, joint_direction), ...] for direction-annotated objects.
    joint_names = [m[1] for m in joint_metadata]
    joint_directions = [m[2] if len(m) > 2 else 1 for m in joint_metadata]
    if not joint_names:
        print("No openable joint was listed in metadata for object %s" % obj.name)
        return None, None, None
    joint_names = set(obj.get_prefixed_joint_name(joint_name).encode(encoding="utf-8") for joint_name in joint_names)

    # Get joint infos and compute openness thresholds.
    body_ids = obj.get_body_ids()
    assert len(body_ids) == 1, "Open state only supports single-body objects."
    body_id = body_ids[0]
    all_joint_ids = utils.get_joints(body_id)
    all_joint_infos = [utils.get_joint_info(body_id, joint_id) for joint_id in all_joint_ids]
    relevant_joint_infos = [joint_info for joint_info in all_joint_infos if joint_info.jointName in joint_names]

    # Assert that all of the joints' names match our expectations.
    assert len(joint_names) == len(
        relevant_joint_infos
    ), "Unexpected joints found during Open state joint checking. Expected %r, found %r." % (
        joint_names,
        relevant_joint_infos,
    )
    assert all(joint_info.jointType in _JOINT_THRESHOLD_BY_TYPE.keys() for joint_info in relevant_joint_infos)

    return both_sides, relevant_joint_infos, joint_directions


class Open(CachingEnabledObjectState, BooleanState):
    def _compute_value(self):
        both_sides, relevant_joint_infos, joint_directions = _get_relevant_joints(self.obj)
        if not relevant_joint_infos:
            return False

        # The "sides" variable is used to check open/closed state for objects whose joints can switch
        # positions. These objects are annotated with the both_sides annotation and the idea is that switching
        # the directions of *all* of the joints results in a similarly valid checkable state. As a result, to check
        # each "side", we multiply *all* of the joint directions with the coefficient belonging to that side, which
        # may be 1 or -1.
        sides = [1, -1] if both_sides else [1]

        sides_openness = []
        for side in sides:
            # Compute a boolean openness state for each joint by comparing positions to thresholds.
            joint_ids = [joint_info.jointIndex for joint_info in relevant_joint_infos]
            joint_thresholds = (
                _compute_joint_threshold(joint_info, joint_direction * side)
                for joint_info, joint_direction in zip(relevant_joint_infos, joint_directions)
            )
            joint_positions = utils.get_joint_positions(self.obj.get_body_ids()[0], joint_ids)
            joint_openness = (
                _is_in_range(position, threshold, open_end)
                for position, (threshold, open_end, closed_end) in zip(joint_positions, joint_thresholds)
            )

            # Looking from this side, the object is open if any of its joints is open.
            sides_openness.append(any(joint_openness))

        # The object is open only if it's open from all of its sides.
        return all(sides_openness)

    def _set_value(self, new_value, fully=False):
        """
        Set the openness state, either to a random joint position satisfying the new value, or fully open/closed.

        @param new_value: bool value for the openness state of the object.
        @param fully: whether the object should be fully opened/closed (e.g. all relevant joints to 0/1).
        @return: bool indicating setter success. Failure may happen due to unannotated objects.
        """
        both_sides, relevant_joint_infos, joint_directions = _get_relevant_joints(self.obj)
        if not relevant_joint_infos:
            return False

        # The "sides" variable is used to check open/closed state for objects whose joints can switch
        # positions. These objects are annotated with the both_sides annotation and the idea is that switching
        # the directions of *all* of the joints results in a similarly valid checkable state. We want our object to be
        # open from *both* of the two sides, and I was too lazy to implement the logic for this without rejection
        # sampling, so that's what we do.
        # TODO: Implement a sampling method that's guaranteed to be correct, ditch the rejection method.
        sides = [1, -1] if both_sides else [1]

        for _ in range(_OPEN_SAMPLING_ATTEMPTS):
            side = random.choice(sides)

            # All joints are relevant if we are closing, but if we are opening let's sample a subset.
            if new_value and not fully:
                num_to_open = random.randint(1, len(relevant_joint_infos))
                relevant_joint_infos = random.sample(relevant_joint_infos, num_to_open)

            # Go through the relevant joints & set random positions.
            for joint_info, joint_direction in zip(relevant_joint_infos, joint_directions):
                threshold, open_end, closed_end = _compute_joint_threshold(joint_info, joint_direction * side)

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
                    joint_pos = random.uniform(low, high)

                # Save sampled position.
                utils.set_joint_position(self.obj.get_body_ids()[0], joint_info.jointIndex, joint_pos)

            # If we succeeded, return now.
            if self._compute_value() == new_value:
                return True

        # We exhausted our attempts and could not find a working sample.
        return False

    # We don't need to do anything here - since the joints are saved, this should work directly.
    def _dump(self):
        return None

    def load(self, data):
        return
