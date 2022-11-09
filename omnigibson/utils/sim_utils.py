import numpy as np
from collections import Iterable
import logging

import omnigibson as og
import omnigibson.utils.transform_utils as T


def prims_to_rigid_prim_set(inp_prims):
    # Avoid circular imports
    from omnigibson.prims.entity_prim import EntityPrim
    from omnigibson.prims.rigid_prim import RigidPrim

    out = set()
    for prim in inp_prims:
        if isinstance(prim, EntityPrim):
            out.update({link for link in prim.links.values()})
        elif isinstance(prim, RigidPrim):
            out.add(prim)
        else:
            raise ValueError(f"Inputted prims must be either EntityPrim or RigidPrim instances "
                             f"when getting collisions! Type: {type(prim)}")
    return out

def get_collisions(prims=None, prims_check=None, prims_exclude=None, step_physics=False):
    """
    Grab collisions that occurred during the most recent physics timestep associated with prims @prims

    Args:
        prims (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): Prim(s) to check for collision.
            If None, will check against all objects currently in the scene.
        prims_check (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): If specified, will
            only check for collisions with these specific prim(s)
        prims_exclude (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): If specified, will
            explicitly ignore any collisions with these specific prim(s)
        step_physics (bool): Whether to step the physics first before checking collisions. Default is False

    Returns:
        set of 2-tuple: Unique collision pairs occurring in the simulation at the current timestep between the
        specified prim(s), represented by their prim_paths
    """
    # Make sure sim is playing
    assert og.sim.is_playing(), "Cannot get collisions while sim is not playing!"

    # Optionally step physics and then update contacts
    if step_physics:
        og.sim.step_physics()

    # Standardize inputs
    prims = og.sim.scene.objects if prims is None else prims if isinstance(prims, Iterable) else [prims]
    prims_check = [] if prims_check is None else prims_check if isinstance(prims_check, Iterable) else [prims_check]
    prims_exclude = [] if prims_exclude is None else prims_exclude if isinstance(prims_exclude, Iterable) else [prims_exclude]

    # Convert into prim paths to check for collision
    def get_paths_from_rigid_prims(inp_prims):
        return {prim.prim_path for prim in inp_prims}

    def get_contacts(inp_prims):
        return {(c.body0, c.body1) for prim in inp_prims for c in prim.contact_list()}

    rprims = prims_to_rigid_prim_set(prims)
    rprims_check = prims_to_rigid_prim_set(prims_check)
    rprims_exclude = prims_to_rigid_prim_set(prims_exclude)

    paths = get_paths_from_rigid_prims(rprims)
    paths_check = get_paths_from_rigid_prims(rprims_check)
    paths_exclude = get_paths_from_rigid_prims(rprims_exclude)

    # Run sanity checks
    assert paths_check.isdisjoint(paths_exclude), \
        f"Paths to check and paths to ignore collisions for should be mutually exclusive! " \
        f"paths_check: {paths_check}, paths_exclude: {paths_exclude}"

    # Determine whether we're checking / filtering any collision from collision set A
    should_check_collisions = len(paths_check) > 0
    should_filter_collisions = len(paths_exclude) > 0

    # Get all collisions from the objects set
    collisions = get_contacts(rprims)

    # Only run the following (expensive) code if we are actively using filtering criteria
    if should_check_collisions or should_filter_collisions:

        # First filter out unnecessary collisions
        if should_filter_collisions:
            # First filter pass, remove the intersection of the main contacts and the contacts from the exclusion set minus
            # the intersection between the exclusion and normal set
            # This filters out any matching collisions in the exclusion set that are NOT an overlap
            # between @rprims and @rprims_exclude
            rprims_exclude_intersect = rprims_exclude.intersection(rprims)
            exclude_disjoint_collisions = get_contacts(rprims_exclude - rprims_exclude_intersect)
            collisions.difference_update(exclude_disjoint_collisions)

            # Second filter pass, we remove collisions that may include self-collisions
            # This is a bit more tricky because we need to actually look at the individual contact pairs to determine
            # whether it's a collision (which may include a self-collision) that should be filtered
            # We do this by grabbing the contacts of the intersection between the exclusion and normal rprims sets,
            # and then making sure the resulting contact pair sets are completely disjoint from the paths intersection
            exclude_intersect_collisions = get_contacts(rprims_exclude_intersect)
            collisions.difference_update({pair for pair in exclude_intersect_collisions if paths.issuperset(set(pair))})

        # Now, we additionally check for explicit collisions, filtering out any that do not meet this criteria
        # This is essentially the inverse of the filter collision process, where we do two passes again, but for each
        # case we look at the union rather than the subtraction of the two sets
        if should_check_collisions:
            # First check pass, keep the intersection of the main contacts and the contacts from the check set minus
            # the intersection between the check and normal set
            # This keeps any matching collisions in the check set that overlap between @rprims and @rprims_check
            rprims_check_intersect = rprims_check.intersection(rprims)
            check_disjoint_collisions = get_contacts(rprims_check - rprims_check_intersect)
            valid_other_collisions = collisions.intersection(check_disjoint_collisions)

            # Second check pass, we additionally keep collisions that may include self-collisions
            # This is a bit more tricky because we need to actually look at the individual contact pairs to determine
            # whether it's a collision (which may include a self-collision) that should be kept
            # We do this by grabbing the contacts of the intersection between the check and normal rprims sets,
            # and then making sure the resulting contact pair sets is strictly a subset of the original set
            # Lastly, we only keep the intersection of this resulting set with the original collision set, so that
            # any previously filtered collisions are respected
            check_intersect_collisions = get_contacts(rprims_check_intersect)
            valid_intersect_collisions = collisions.intersection({pair for pair in check_intersect_collisions if paths.issuperset(set(pair))})

            # Collisions is union of valid other and valid self collisions
            collisions = valid_other_collisions.union(valid_intersect_collisions)

    # Only going into this if it is for logging --> efficiency
    if logging.root.level <= logging.DEBUG:
        for item in collisions:
            logging.debug("linkA:{}, linkB:{}".format(item[0], item[1]))

    return collisions


def check_collision(prims=None, prims_check=None, prims_exclude=None, step_physics=False):
    """
    Checks if any valid collisions occurred during the most recent physics timestep associated with prims @prims

    Args:
        prims (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): Prim(s) to check for collision.
            If None, will check against all objects currently in the scene.
        prims_check (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): If specified, will
            only check for collisions with these specific prim(s)
        prims_exclude (None or EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): If specified, will
            explicitly ignore any collisions with these specific prim(s)
        step_physics (bool): Whether to step the physics first before checking collisions. Default is False

    Returns:
        bool: True if a valid collision has occurred, else False
    """
    return len(get_collisions(
        prims=prims,
        prims_check=prims_check,
        prims_exclude=prims_exclude,
        step_physics=step_physics)) > 0


def filter_collisions(collisions, filter_prims):
    """
    Filters collision pairs @collisions based on a set of prims @filter_prims.

    Args:
        collisions (set of 2-tuple): Collision pairs that should be filtered
        filter_prims (EntityPrim or RigidPrim or tuple of EntityPrim or RigidPrim): Prim(s) specifying which
            collisions to filter for

    Returns:
        set of 2-tuple: Filtered collision pairs
    """
    paths = prims_to_rigid_prim_set(filter_prims)

    filtered_collisions = set()
    for pair in collisions:
        if set(pair).isdisjoint(paths):
            filtered_collisions.add(pair)

    return filtered_collisions


def place_base_pose(obj, pos, quat=None, z_offset=None):
    """
    Place the object so that its base (z-min) rests at the location of @pos

    Args:
        obj (BaseObject): Object to place in the environment
        pos (3-array): Global (x,y,z) location to place the base of the robot
        quat (None or 4-array): Optional (x,y,z,w) quaternion orientation when placing the object.
            If None, the object's current orientation will be used
        z_offset (None or float): Optional additional z_offset to apply
    """
    # avoid circular dependency
    from omnigibson.object_states import AABB

    lower, _ = obj.states[AABB].get_value()
    cur_pos = obj.get_position()

    z_diff = cur_pos[2] - lower[2]

    pos[2] += z_diff
    if z_offset is not None:
        pos[2] += z_offset

    obj.set_position_orientation(pos, quat)


def test_valid_pose(obj, pos, quat=None, z_offset=None):
    """
    Test if the object can be placed with no collision.

    Args:
        obj (BaseObject): Object to place in the environment
        pos (3-array): Global (x,y,z) location to place the object
        quat (None or 4-array): Optional (x,y,z,w) quaternion orientation when placing the object.
            If None, the object's current orientation will be used
        z_offset (None or float): Optional additional z_offset to apply

    Returns:
        bool: Whether the placed object position is valid
    """
    # Make sure sim is playing
    assert og.sim.is_playing(), "Cannot test valid pose while sim is not playing!"

    # Store state before checking object position
    state = og.sim.scene.dump_state(serialized=False)

    # Set the pose of the object
    place_base_pose(obj, pos, quat, z_offset)

    # If we're placing a robot, make sure it's reset and not moving
    # Run import here to avoid circular imports
    from omnigibson.robots.robot_base import BaseRobot
    if isinstance(obj, BaseRobot):
        obj.reset()
        obj.keep_still()

    # Check whether we're in collision after taking a single physics step
    in_collision = check_collision(prims=obj, step_physics=True)

    # Restore state after checking the collision
    og.sim.load_state(state, serialized=False)

    # Valid if there are no collisions
    return not in_collision


def land_object(obj, pos, quat=None, z_offset=None):
    """
    Land the object at the specified position @pos, given a valid position and orientation.

    Args:
        obj (BaseObject): Object to place in the environment
        pos (3-array): Global (x,y,z) location to place the object
        quat (None or 4-array): Optional (x,y,z,w) quaternion orientation when placing the object.
            If None, a random orientation about the z-axis will be sampled
        z_offset (None or float): Optional additional z_offset to apply
    """
    # Make sure sim is playing
    assert og.sim.is_playing(), "Cannot land object while sim is not playing!"

    # Set the object's pose
    quat = T.euler2quat([0, 0, np.random.uniform(0, np.pi * 2)]) if quat is None else quat
    place_base_pose(obj, pos, quat, z_offset)

    # If we're placing a robot, make sure it's reset and not moving
    # Run import here to avoid circular imports
    from omnigibson.robots.robot_base import BaseRobot
    is_robot = isinstance(obj, BaseRobot)
    if is_robot:
        obj.reset()
        obj.keep_still()

    # Check to make sure we landed successfully
    # land for maximum 1 second, should fall down ~5 meters
    land_success = False
    max_simulator_step = int(1.0 / og.sim.get_rendering_dt())
    for _ in range(max_simulator_step):
        # Run a sim step and see if we have any contacts
        og.sim.step()
        land_success = check_collision(prims=obj)
        if land_success:
            # Once we're successful, we can break immediately
            print(f"Landed object {obj.name} successfully!")
            break

    # Print out warning in case we failed to land the object successfully
    if not land_success:
        logging.warning(f"Object {obj.name} failed to land.")

    # Make sure robot isn't moving at the end if we're a robot
    if is_robot:
        obj.reset()
        obj.keep_still()
