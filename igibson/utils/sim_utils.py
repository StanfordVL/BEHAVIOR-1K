import numpy as np
from collections import Iterable
import logging

from igibson import sim
from igibson.robots.robot_base import BaseRobot
import igibson.utils.transform_utils as T


def get_collisions(objects):
    """
    Grab collisions that occurred during the most recent physics timestep

    Args:
        objects (RigidPrim or EntityPrim or list of RigidPrim or EntityPrim): objects to check for collisions

    Returns:
        set of 2-tuple: Unique collision pairs occurring in the simulation at the current timestep, represented
            by their prim_paths
    """
    # Make sure sim is playing
    assert sim.is_playing(), "Cannot get collisions while sim is not playing!"

    objects = objects if isinstance(objects, Iterable) else [objects]
    # Grab collisions
    return {(c.body0, c.body1) for obj in objects for c in obj.contact_list()}


def check_collision(objsA=None, linksA=None, objsB=None, linksB=None, step_physics=False):
    """
    Check whether the given object @objsA or any of @links has collision after one simulator step. If both
    are specified, will take the union of the two.

    Note: This natively checks for collisions with @objsA and @linksA. If @objsB and @linksB are None, any valid
        collision will trigger a True

    Args:
        objsA (None or EntityPrim or list of EntityPrim): If specified, object(s) to check for collision
        linksA (None or RigidPrim or list of RigidPrim): If specified, link(s) to check for collision
        objsB (None or EntityPrim or list of EntityPrim): If specified, object(s) to check for collision with any
            of @objsA or @linksA
        linksB (None or RigidPrim or list of RigidPrim): If specified, link(s) to check for collision with any
            of @objsA or @linksA
        step_physics (bool): Whether to step the physics first before checking collisions. Default is False

    Returns:
        bool: Whether any of @objsA or @linksA are in collision or not, possibly with @objsB or @linksB if specified
    """
    # Make sure sim is playing
    assert sim.is_playing(), "Cannot check collisions while sim is not playing!"

    # Optionally step physics and then update contacts
    if step_physics:
        sim.step_physics()

    # Run sanity checks and standardize inputs
    assert objsA is not None or linksA is not None, \
        "Either objsA or linksA must be specified for collision checking!"

    objsA = [] if objsA is None else [objsA] if not isinstance(objsA, Iterable) else objsA
    linksA = [] if linksA is None else [linksA] if not isinstance(linksA, Iterable) else linksA
    objsB = [] if objsB is None else [objsB] if not isinstance(objsB, Iterable) else objsB
    linksB = [] if linksB is None else [linksB] if not isinstance(linksB, Iterable) else linksB

    # Grab all link prim paths owned by the collision set A
    paths_A = {link.prim_path for obj in objsA for link in obj.links.values()}
    paths_A = paths_A.union({link.prim_path for link in linksA})

    # Determine whether we're checking any collision from collision set A
    check_any_collision = objsB is None and linksB is None

    # Get all collisions from the objects set
    collisions = get_collisions(objects=objsA + linksA + objsB + linksB)

    in_collision = False
    if check_any_collision:
        # Immediately check collisions
        for col_pair in collisions:
            if len(set(col_pair) - paths_A) < 2:
                in_collision = True
                break
    else:
        # Grab all link prim paths owned by the collision set B
        paths_B = {link.prim_path for obj in objsB for link in obj.links.values()}
        paths_B = paths_B.union({link.prim_path for link in linksB})
        paths_shared = paths_A.intersection(paths_B)
        paths_disjoint = paths_A.union(paths_B) - paths_shared
        is_AB_shared = len(paths_shared) > 0

        # Check collisions specifically between groups A and B
        for col_pair in collisions:
            col_pair = set(col_pair)
            # Two cases -- either paths_A and paths_B overlap or they don't. Process collision checking logic
            # separately for each case
            if is_AB_shared:
                # Two cases for valid collision: there is a shared collision body in this pair or there isn't.
                # Process separately in each case
                col_pair_no_shared = col_pair - paths_shared
                if len(col_pair_no_shared) < 2:
                    # Make sure this set minus the disjoint set results in empty col_pair remaining -- this means
                    # a valid pair combo was found
                    if len(col_pair_no_shared - paths_disjoint) == 0:
                        in_collision = True
                        break
                else:
                    # Make sure A and B sets each have an entry in the col pair for a valid collision
                    if len(col_pair - paths_A) == 1 and len(col_pair - paths_B) == 1:
                        in_collision = True
                        break
            else:
                # Make sure A and B sets each have an entry in the col pair for a valid collision
                if len(col_pair - paths_A) == 1 and len(col_pair - paths_B) == 1:
                    in_collision = True
                    break

    # Only going into this if it is for logging --> efficiency
    if logging.root.level <= logging.DEBUG:
        for item in collisions:
            logging.debug("linkA:{}, linkB:{}".format(item[0], item[1]))

    return in_collision


def test_valid_pose(obj, pos, quat=None):
    """
    Test if the object can be placed with no collision.

    Args:
        obj (BaseObject): Object to place in the environment
        pos (3-array): Global (x,y,z) location to place the object
        quat (None or 4-array): Optional (x,y,z,w) quaternion orientation when placing the object.
            If None, the object's current orientation will be used

    Returns:
        bool: Whether the placed object position is valid
    """
    # Make sure sim is playing
    assert sim.is_playing(), "Cannot test valid pose while sim is not playing!"

    # Store state before checking object position
    state = sim.scene.dump_state(serialized=False)

    # Set the position of the object
    obj.set_position_orientation(position=pos, orientation=quat)

    # If we're placing a robot, make sure it's reset and not moving
    if isinstance(obj, BaseRobot):
        obj.reset()
        obj.keep_still()

    # Check whether we're in collision after taking a single physics step
    in_collision = check_collision(objsA=obj, step_physics=True)

    # Restore state after checking the collision
    sim.scene.load_state(state, serialized=False)

    # Valid if there are no collisions
    return not in_collision


def land_object(obj, pos, quat):
    """
    Land the object at the specified position @pos, given a valid position and orientation.

    Args:
        obj (BaseObject): Object to place in the environment
        pos (3-array): Global (x,y,z) location to place the object
        quat (None or 4-array): Optional (x,y,z,w) quaternion orientation when placing the object.
            If None, a random orientation about the z-axis will be sampled
    """
    # Make sure sim is playing
    assert sim.is_playing(), "Cannot land object while sim is not playing!"

    # Set the object's pose
    quat = T.euler2quat([0, 0, np.random.uniform(0, np.pi * 2)]) if quat is None else quat
    obj.set_position_orientation(position=pos, orientation=quat)

    # If we're placing a robot, make sure it's reset and not moving
    is_robot = isinstance(obj, BaseRobot)
    if is_robot:
        obj.reset()
        obj.keep_still()

    # Check to make sure we landed successfully
    # land for maximum 1 second, should fall down ~5 meters
    land_success = False
    max_simulator_step = int(1.0 / sim.get_rendering_dt())
    for _ in range(max_simulator_step):
        # Run a sim step and see if we have any contacts
        sim.step()
        land_success = check_collision(objsA=obj)
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
