import cv2
import numpy as np

from IPython import embed
from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson.macros import create_module_macros, Dict
from omnigibson import object_states
from omnigibson.object_states.aabb import AABB
from omnigibson.utils import sampling_utils
from omnigibson.utils.sim_utils import check_collision
import omnigibson.utils.transform_utils as T

from omni.physx import acquire_physx_scene_query_interface


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS = Dict({
    # "hit_to_plane_threshold": 0.1,  # TODO: Tune this parameter.
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 1.0,
    "max_sampling_attempts": 50,
    "aabb_offset": 0.01,
})

m.INSIDE_RAY_CASTING_SAMPLING_PARAMS = Dict({
    # "hit_to_plane_threshold": 0.1,  # TODO: Tune this parameter.
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 0.4,
    "bimodal_mean_fraction": 0.5,
    "max_sampling_attempts": 100,
    "aabb_offset": -0.01,
})


def get_center_extent(obj_states):
    assert AABB in obj_states
    aabb = obj_states[AABB].get_value()
    center, extent = (aabb[0] + aabb[1]) / 2.0, aabb[1] - aabb[0]
    return center, extent


def sample_kinematics(
    predicate,
    objA,
    objB,
    binary_state,
    use_ray_casting_method=False,
    max_trials=100,
    z_offset=0.05,
    skip_falling=False,
):
    # Can only sample kinematics for binary_states currently
    if not binary_state:
        raise NotImplementedError()

    # TODO: This seems hacky -- can we generalize this in any way?
    sample_on_floor = predicate == "onFloor"

    # Don't run kinematics under certain conditions
    if not use_ray_casting_method and not sample_on_floor and predicate not in objB.supporting_surfaces:
        return False

    # Wake objects accordingly
    objA.wake()
    if not sample_on_floor:
        objB.wake()

    # Save the state of the simulator
    state = og.sim.dump_state()

    # Attempt sampling
    for i in range(max_trials):
        pos = None
        if hasattr(objA, "orientations") and objA.orientations is not None:
            orientation = objA.sample_orientation()
        else:
            orientation = np.array([0, 0, 0, 1.0])

        # Orientation needs to be set for stable_z_on_aabb to work correctly
        # Position needs to be set to be very far away because the object's
        # original position might be blocking rays (use_ray_casting_method=True)
        old_pos = np.array([200, 200, 200])
        objA.set_position_orientation(old_pos, orientation)
        # We also need to step physics to make sure the pose propagates downstream (e.g.: to Bounding Box computations)
        og.sim.step_physics()

        if sample_on_floor:
            # Run import here to avoid circular imports
            from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
            assert isinstance(og.sim.scene, InteractiveTraversableScene), \
                "Active scene must be an InteractiveTraversableScene in order to sample kinematics on!"
            _, pos = og.sim.scene.seg_map.get_random_point_by_room_instance(objB.room_instance)

            if pos is not None:
                # Get the combined AABB.
                lower, _ = objA.states[object_states.AABB].get_value()
                # Move the position to a stable Z for the object.
                pos[2] += objA.get_position()[2] - lower[2]
        else:
            if use_ray_casting_method:
                if predicate == "onTop":
                    params = m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS
                elif predicate == "inside":
                    params = m.INSIDE_RAY_CASTING_SAMPLING_PARAMS
                else:
                    raise ValueError(f"predicate must be either onTop or inside in order to use ray casting-based "
                                     f"kinematic sampling, but instead got: {predicate}")

                # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
                parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
                    xy_aligned=True
                )

                sampling_results = sampling_utils.sample_cuboid_on_object(
                    objB,
                    num_samples=1,
                    cuboid_dimensions=parallel_bbox_extents,
                    axis_probabilities=[0, 0, 1],
                    refuse_downwards=True,
                    undo_padding=True,
                    **params,
                )

                sampled_vector = sampling_results[0][0]
                sampled_quaternion = sampling_results[0][2]

                sampling_success = sampled_vector is not None
                if sampling_success:
                    # Move the object from the original parallel bbox to the sampled bbox
                    parallel_bbox_rotation = R.from_quat(parallel_bbox_orn)
                    sample_rotation = R.from_quat(sampled_quaternion)
                    original_rotation = R.from_quat(orientation)

                    # The additional orientation to be applied should be the delta orientation
                    # between the parallel bbox orientation and the sample orientation
                    additional_rotation = sample_rotation * parallel_bbox_rotation.inv()
                    combined_rotation = additional_rotation * original_rotation
                    orientation = combined_rotation.as_quat()

                    # The delta vector between the base CoM frame and the parallel bbox center needs to be rotated
                    # by the same additional orientation
                    diff = old_pos - parallel_bbox_center
                    rotated_diff = additional_rotation.apply(diff)
                    pos = sampled_vector + rotated_diff
            else:
                # Object B must be a dataset object since it must have supporting surfaces metadata pre-annotated
                # Run import here to avoid circular imports
                from omnigibson.objects.dataset_object import DatasetObject
                assert isinstance(objB, DatasetObject), \
                    f"objB must be an instance of DatasetObject in order to use non-ray casting-based " \
                    f"kinematic sampling!"

                random_idx = np.random.randint(len(objB.supporting_surfaces[predicate].keys()))
                objB_link_name = list(objB.supporting_surfaces[predicate].keys())[random_idx]
                random_height_idx = np.random.randint(len(objB.supporting_surfaces[predicate][objB_link_name]))
                height, height_map = objB.supporting_surfaces[predicate][objB_link_name][random_height_idx]
                obj_half_size = np.max(objA.aabb_extent) / 2 * 100
                obj_half_size_scaled = np.array([obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
                obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(np.int)
                height_map_eroded = cv2.erode(height_map, np.ones(obj_half_size_scaled, np.uint8))

                valid_pos = np.array(height_map_eroded.nonzero())
                if valid_pos.shape[1] != 0:
                    random_pos_idx = np.random.randint(valid_pos.shape[1])
                    random_pos = valid_pos[:, random_pos_idx]
                    y_map, x_map = random_pos
                    y = y_map / 100.0 - 2
                    x = x_map / 100.0 - 2
                    z = height

                    pos = np.array([x, y, z])
                    pos *= objB.scale

                    # the supporting surface is defined w.r.t to the link frame, so we need to convert it into
                    # the world frame
                    link_pos, link_quat = objB.links[objB_link_name].get_position_orientation()
                    pos = T.quat2mat(link_quat).dot(pos) + np.array(link_pos)
                    # Get the combined AABB.
                    lower, _ = objA.states[object_states.AABB].get_value()
                    # Move the position to a stable Z for the object.
                    pos[2] += objA.get_position()[2] - lower[2]

        if pos is None:
            success = False
        else:
            pos[2] += z_offset
            objA.set_position_orientation(pos, orientation)
            # Step physics
            og.sim.step_physics()
            success = not objA.in_contact()

        if og.debug_sampling:
            print("sample_kinematics", success)
            embed()

        if success:
            break
        else:
            og.sim.load_state(state)

    if success and not skip_falling:
        objA.set_position_orientation(pos, orientation)

        # Let it fall for 0.2 second
        for _ in range(int(0.2 / og.sim.get_physics_dt())):
            og.sim.step_physics()
            if objA.in_contact():
                break

        # Render at the end
        og.sim.render()

    return success
