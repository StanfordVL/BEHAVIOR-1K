import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import Dict, create_module_macros, macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.utils import sampling_utils
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import debug_breakpoint

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 10
m.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 10
m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS = Dict(
    {
        "bimodal_stdev_fraction": 1e-6,
        "bimodal_mean_fraction": 1.0,
        "aabb_offset_fraction": 0.02,
        "max_sampling_attempts": 50,
    }
)

m.INSIDE_RAY_CASTING_SAMPLING_PARAMS = Dict(
    {
        "bimodal_stdev_fraction": 0.4,
        "bimodal_mean_fraction": 0.5,
        "aabb_offset_fraction": -0.02,
        "max_sampling_attempts": 100,
    }
)

m.UNDER_RAY_CASTING_SAMPLING_PARAMS = Dict(
    {
        "bimodal_stdev_fraction": 1e-6,
        "bimodal_mean_fraction": 0.5,
        "aabb_offset_fraction": 0.02,
        "max_sampling_attempts": 50,
    }
)


def sample_cuboid_for_predicate(predicate, on_obj, bbox_extent):
    if predicate == "onTop":
        params = m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS
    elif predicate == "inside":
        params = m.INSIDE_RAY_CASTING_SAMPLING_PARAMS
    elif predicate == "under":
        params = m.UNDER_RAY_CASTING_SAMPLING_PARAMS
    else:
        raise ValueError(
            f"predicate must be onTop, under or inside in order to use ray casting-based "
            f"kinematic sampling, but instead got: {predicate}"
        )

    if predicate == "under":
        start_points, end_points = sampling_utils.sample_raytest_start_end_symmetric_bimodal_distribution(
            obj=on_obj,
            num_samples=1,
            axis_probabilities=[0, 0, 1],
            **params,
        )
        return sampling_utils.sample_cuboid_on_object(
            obj=None,
            start_points=start_points,
            end_points=end_points,
            ignore_objs=[on_obj],
            cuboid_dimensions=bbox_extent,
            refuse_downwards=True,
            undo_cuboid_bottom_padding=True,
            max_angle_with_z_axis=0.17,
            hit_proportion=0.0,  # rays will NOT hit the object itself, but the surface below it.
        )
    else:
        return sampling_utils.sample_cuboid_on_object_symmetric_bimodal_distribution(
            on_obj,
            num_samples=1,
            axis_probabilities=[0, 0, 1],
            cuboid_dimensions=bbox_extent,
            refuse_downwards=True,
            undo_cuboid_bottom_padding=True,
            max_angle_with_z_axis=0.17,
            **params,
        )


def sample_kinematics(
    predicate,
    objA,
    objB,
    max_trials=None,
    z_offset=0.05,
    skip_falling=False,
):
    """
    Samples the given @predicate kinematic state for @objA with respect to @objB

    Args:
        predicate (str): Name of the predicate to sample, e.g.: "onTop"
        objA (StatefulObject): Object whose state should be sampled. e.g.: for sampling a microwave
            on a cabinet, @objA is the microwave
        objB (StatefulObject): Object who is the reference point for @objA's state. e.g.: for sampling
            a microwave on a cabinet, @objB is the cabinet
        max_trials (int): Number of attempts for sampling
        z_offset (float): Z-offset to apply to the sampled pose
        skip_falling (bool): Whether to let @objA fall after its position is sampled or not

    Returns:
        bool: True if successfully sampled, else False
    """
    if max_trials is None:
        max_trials = m.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS
    assert (
        z_offset > 0.5 * 9.81 * (og.sim.get_physics_dt() ** 2) + 0.02
    ), f"z_offset {z_offset} is too small for the current physics_dt {og.sim.get_physics_dt()}"

    # Wake objects accordingly and make sure both are kept still
    objA.wake()
    objB.wake()

    objA.keep_still()
    objB.keep_still()

    # Save the state of the simulator
    state = og.sim.dump_state()

    # Attempt sampling
    for i in range(max_trials):
        pos = None
        if hasattr(objA, "orientations") and objA.orientations is not None:
            orientation = objA.sample_orientation()
        else:
            orientation = th.tensor([0, 0, 0, 1.0])

        # Orientation needs to be set for stable_z_on_aabb to work correctly
        # Position needs to be set to be very far away because the object's
        # original position might be blocking rays (use_ray_casting_method=True)
        old_pos = th.tensor([100, 100, 10])
        objA.set_position_orientation(position=old_pos, orientation=orientation)
        objA.keep_still()
        # We also need to step physics to make sure the pose propagates downstream (e.g.: to Bounding Box computations)
        og.sim.step_physics()

        # This would slightly change because of the step_physics call.
        old_pos, orientation = objA.get_position_orientation()

        # Run import here to avoid circular imports
        from omnigibson.objects.dataset_object import DatasetObject

        if isinstance(objA, DatasetObject) and objA.prim_type == PrimType.RIGID:
            # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
            parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
                xy_aligned=True
            )
        else:
            aabb_lower, aabb_upper = objA.states[AABB].get_value()
            parallel_bbox_center = (aabb_lower + aabb_upper) / 2.0
            parallel_bbox_orn = th.tensor([0.0, 0.0, 0.0, 1.0])
            parallel_bbox_extents = aabb_upper - aabb_lower

        sampling_results = sample_cuboid_for_predicate(predicate, objB, parallel_bbox_extents)
        sampled_vector = sampling_results[0][0]
        sampled_quaternion = sampling_results[0][2]

        sampling_success = sampled_vector is not None

        if sampling_success:
            # Move the object from the original parallel bbox to the sampled bbox
            # The additional orientation to be applied should be the delta orientation
            # between the parallel bbox orientation and the sample orientation
            additional_quat = T.quat_multiply(sampled_quaternion, T.quat_inverse(parallel_bbox_orn))
            combined_quat = T.quat_multiply(additional_quat, orientation)
            orientation = combined_quat

            # The delta vector between the base CoM frame and the parallel bbox center needs to be rotated
            # by the same additional orientation
            diff = old_pos - parallel_bbox_center
            rotated_diff = T.quat_apply(additional_quat, diff)
            pos = sampled_vector + rotated_diff

        if pos is None:
            success = False
        else:
            pos[2] += z_offset
            objA.set_position_orientation(position=pos, orientation=orientation)
            objA.keep_still()

            og.sim.step_physics()
            objA.keep_still()
            success = len(objA.states[ContactBodies].get_value()) == 0

        if macros.utils.sampling_utils.DEBUG_SAMPLING:
            debug_breakpoint(f"sample_kinematics: {success}")

        if success:
            break
        else:
            og.sim.load_state(state)

    # If we didn't succeed, try last-ditch effort
    if not success and predicate in {"onTop", "inside"}:
        og.sim.step_physics()
        # Place objA at center of objB's AABB, offset in z direction such that their AABBs are "stacked", and let fall
        # until it settles
        aabb_lower_a, aabb_upper_a = objA.states[AABB].get_value()
        aabb_lower_b, aabb_upper_b = objB.states[AABB].get_value()
        bbox_to_obj = objA.get_position_orientation()[0] - (aabb_lower_a + aabb_upper_a) / 2.0
        desired_bbox_pos = (aabb_lower_b + aabb_upper_b) / 2.0
        desired_bbox_pos[2] = aabb_upper_b[2] + (aabb_upper_a[2] - aabb_lower_a[2]) / 2.0
        pos = desired_bbox_pos + bbox_to_obj
        success = True

    if success and not skip_falling:
        objA.set_position_orientation(position=pos, orientation=orientation)
        objA.keep_still()

        # Step until either (a) max steps is reached (total of 0.5 second in sim time) or (b) contact is made, then
        # step until (a) max steps is reached (restarted from 0) or (b) velocity is below some threshold
        n_steps_max = int(0.5 / og.sim.get_physics_dt())
        i = 0
        while len(objA.states[ContactBodies].get_value()) == 0 and i < n_steps_max:
            og.sim.step_physics()
            i += 1
        objA.keep_still()
        objB.keep_still()
        # Step a few times so velocity can become non-zero if the objects are moving
        for i in range(5):
            og.sim.step_physics()
        i = 0
        while th.norm(objA.get_linear_velocity()) > 1e-3 and i < n_steps_max:
            og.sim.step_physics()
            i += 1

        # Render at the end
        og.sim.render()

    return success


def sample_cloth_on_rigid(obj, other, max_trials=40, z_offset=0.05, randomize_xy=True):
    """
    Samples the cloth object @obj on the rigid object @other

    Args:
        obj (StatefulObject): Object whose state should be sampled. e.g.: for sampling a bed sheet on a rack,
            @obj is the bed sheet
        other (StatefulObject): Object who is the reference point for @obj's state. e.g.: for sampling a bed sheet
            on a rack, @other is the rack
        max_trials (int): Number of attempts for sampling
        z_offset (float): Z-offset to apply to the sampled pose
        randomize_xy (bool): Whether to randomize the XY position of the sampled pose. If False, the center of @other
            will always be used.

    Returns:
        bool: True if successfully sampled, else False
    """
    assert (
        z_offset > 0.5 * 9.81 * (og.sim.get_physics_dt() ** 2) + 0.02
    ), f"z_offset {z_offset} is too small for the current physics_dt {og.sim.get_physics_dt()}"

    if not (obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
        raise ValueError("sample_cloth_on_rigid requires obj1 is cloth and obj2 is rigid.")

    state = og.sim.dump_state(serialized=False)

    # Reset the cloth
    obj.root_link.reset()

    obj_aabb_low, obj_aabb_high = obj.states[AABB].get_value()
    other_aabb_low, other_aabb_high = other.states[AABB].get_value()

    # z value is always the same: the top-z of the other object + half the height of the object to be placed + offset
    z_value = other_aabb_high[2] + (obj_aabb_high[2] - obj_aabb_low[2]) / 2.0 + z_offset

    if randomize_xy:
        # Sample a random position in the x-y plane within the other object's AABB
        low = th.tensor([other_aabb_low[0], other_aabb_low[1], z_value])
        high = th.tensor([other_aabb_high[0], other_aabb_high[1], z_value])
    else:
        # Always sample the center of the other object's AABB
        low = th.tensor(
            [(other_aabb_low[0] + other_aabb_high[0]) / 2.0, (other_aabb_low[1] + other_aabb_high[1]) / 2.0, z_value]
        )
        high = low

    for _ in range(max_trials):
        # Sample a random position
        pos = th.rand(low.size()) * (high - low) + low
        # Sample a random orientation in the z-axis
        z_lo, z_hi = 0, math.pi * 2
        orn = T.euler2quat(th.tensor([0.0, 0.0, (th.rand(1) * (z_hi - z_lo) + z_lo).item()]))

        obj.set_position_orientation(position=pos, orientation=orn)
        obj.root_link.reset()
        obj.keep_still()

        og.sim.step_physics()
        success = len(obj.states[ContactBodies].get_value()) == 0

        if success:
            break
        else:
            og.sim.load_state(state)

    if success:
        # Let it fall for 0.2 second always to let the cloth settle
        for _ in range(int(0.2 / og.sim.get_physics_dt())):
            og.sim.step_physics()

        obj.keep_still()

        # Render at the end
        og.sim.render()

    return success
