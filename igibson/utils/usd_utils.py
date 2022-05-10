import math
import numpy as np
from collections import Iterable

import omni.usd
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
from omni.syntheticdata import helpers
from pxr import Gf, Vt, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema

from igibson.utils.constants import JointType
from igibson.utils.python_utils import assert_valid_key

GF_TO_VT_MAPPING = {
    Gf.Vec3d: Vt.Vec3dArray,
    Gf.Vec3f: Vt.Vec3fArray,
    Gf.Vec3h: Vt.Vec3hArray,
    Gf.Quatd: Vt.QuatdArray,
    Gf.Quatf: Vt.QuatfArray,
    Gf.Quath: Vt.QuathArray,
    int: Vt.IntArray,
    float: Vt.FloatArray,
    bool: Vt.BoolArray,
    str: Vt.StringArray,
    chr: Vt.CharArray,
}


def array_to_vtarray(arr, element_type):
    """
    Converts array @arr into a Vt-typed array, where each individual element of type @element_type.

    Args:
        arr (n-array): An array of values. Can be, e.g., a list, or numpy array
        element_type (type): Per-element type to convert the elements from @arr into.
            Valid options are keys of GF_TO_VT_MAPPING

    Returns:
        Vt.Array: Vt-typed array, of specified type corresponding to @element_type
    """
    # Make sure array type is valid
    assert_valid_key(key=element_type, valid_keys=GF_TO_VT_MAPPING, name="array element type")

    # Construct list of values
    arr_list = []

    # Check first to see if elements are vectors or not. If this is an iterable value that is not a string,
    # then this is a vector and we have to map it to the correct type via *
    is_vec_element = (isinstance(arr[0], Iterable)) and (not isinstance(arr[0], str))

    # Loop over array and set values
    for ele in arr:
        arr_list.append(element_type(*ele) if is_vec_element else ele)

    return GF_TO_VT_MAPPING[element_type](arr_list)


def get_prim_nested_children(prim):
    """
    Grabs all nested prims starting from root @prim via depth-first-search

    :param prim: Usd.Prim, root prim from which to search for nested children prims

    :return: Tuple[Usd.Prim], nested prims
    """
    prims = []
    for child in get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def get_camera_params(viewport):
    """Get active camera intrinsic and extrinsic parameters.

    Returns:
        dict: Keyword-mapped values of the active camera's parameters:

            pose (numpy.ndarray): camera position in world coordinates,
            fov (float): horizontal field of view in radians
            focal_length (float)
            horizontal_aperture (float)
            view_projection_matrix (numpy.ndarray(dtype=float64, shape=(4, 4)))
            resolution (dict): resolution as a dict with 'width' and 'height'.
            clipping_range (tuple(float, float)): Near and Far clipping values.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(viewport.get_active_camera())
    prim_tf = omni.usd.get_world_transform_matrix(prim)
    view_params = helpers.get_view_params(viewport)
    fov = 2 * math.atan(view_params["horizontal_aperture"] / (2 * view_params["focal_length"]))
    view_proj_mat = helpers.get_view_proj_mat(view_params)

    return {
        "pose": np.array(prim_tf),
        "fov": fov,
        "focal_length": view_params["focal_length"],
        "horizontal_aperture": view_params["horizontal_aperture"],
        "view_projection_matrix": view_proj_mat,
        "resolution": {"width": view_params["width"], "height": view_params["height"]},
        "clipping_range": view_params["clipping_range"],
    }


def get_semantic_objects_pose():
    """Get pose of all objects with a semantic label.
    """
    stage = omni.usd.get_context().get_stage()
    mappings = helpers.get_instance_mappings()
    pose = []
    for m in mappings:
        prim_path = m[1]
        prim = stage.GetPrimAtPath(prim_path)
        prim_tf = omni.usd.get_world_transform_matrix(prim)
        pose.append((str(prim_path), m[2], str(m[3]), np.array(prim_tf)))
    return pose


def create_joint(prim_path, joint_type, body0=None, body1=None, enabled=True, stage=None):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    :param prim_path: str, absolute path to where the joint will be created
    :param joint_type: str, type of joint to create. Valid options are:
        "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
                    (equivalently, one of JointType)
    :param body0: str, absolute path to the first body's prim. At least @body0 or @body1 must be specified.
    :param body1: str, absolute path to the second body's prim. At least @body0 or @body1 must be specified.
    :param enabled: bool, whether to enable this joint or not
    :param stage: Usd.Stage, if specified, should be specific stage to be used to load the joint.
        Otherwise, the current active stage will be used.

    :return Usd.Prim: Created joint prim
    """
    # Make sure we have valid joint_type
    assert JointType.is_valid(joint_type=joint_type), \
        f"Invalid joint specified for creation: {joint_type}"

    # Make sure at least body0 or body1 is specified
    assert body0 is not None or body1 is not None, \
        f"At least either body0 or body1 must be specified when creating a joint!"

    # Define an Xform prim at the current stage, or the simulator's stage if specified
    stage = get_current_stage() if stage is None else stage

    # Create the joint
    joint = UsdPhysics.__dict__[joint_type].Define(stage, prim_path)

    # Possibly add body0, body1 targets
    if body0 is not None:
        assert is_prim_path_valid(body0), f"Invalid body0 path specified: {body0}"
        joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    if body1 is not None:
        assert is_prim_path_valid(body1), f"Invalid body1 path specified: {body1}"
        joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])

    # Get the prim pointed to at this path
    joint_prim = get_prim_at_path(prim_path)

    # Apply joint API interface
    PhysxSchema.PhysxJointAPI.Apply(joint_prim)

    # Possibly (un-/)enable this joint
    joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)

    # Return this joint
    return joint_prim


class CollisionAPI:
    """
    Class containing class methods to facilitate collision handling, e.g. collision groups
    """
    ACTIVE_COLLISION_GROUPS = {}

    @classmethod
    def add_to_collision_group(cls, col_group, prim_path, create_if_not_exist=False):
        """
        Adds the prim and all nested prims specified by @prim_path to the global collision group @col_group. If @col_group
        does not exist, then it will either be created if @create_if_not_exist is True, otherwise will raise an Error.

        Args:
            col_group (str): Name of the collision group to assign the prim at @prim_path to
            prim_path (str): Prim (and all nested prims) to assign to this @col_group
            create_if_not_exist (bool): True if @col_group should be created if it does not already exist, otherwise an
                error will be raised
        """
        # Check if collision group exists or not
        if col_group not in cls.ACTIVE_COLLISION_GROUPS:
            # Raise error if we don't explicitly want to create a new group
            if not create_if_not_exist:
                raise ValueError(f"Collision group {col_group} not found in current registry, and create_if_not_exist"
                                 f"was set to False!")
            # Otherwise, create the new group
            col_group_name = f"/World/collisionGroup_{col_group}"
            group = UsdPhysics.CollisionGroup.Define(get_current_stage(), col_group_name)
            group.GetFilteredGroupsRel().AddTarget(col_group_name)  # Make sure that we can collide within our own group
            cls.ACTIVE_COLLISION_GROUPS[col_group] = group

        # Add this prim to the collision group
        cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this CollisionAPI
        """
        cls.ACTIVE_COLLISION_GROUPS = {}


class BoundingBoxAPI:
    """
    Class containing class methods to facilitate bounding box handling
    """
    CACHE = None

    @classmethod
    def compute_aabb(cls, prim_path):
        """
        Computes the AABB (world-frame oriented) for the prim specified at @prim_path

        Args:
            prim_path (str): Path to the prim to calculate AABB for

        Returns:
            tuple:
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box
        """
        # Create cache if it doesn't already exist
        if cls.CACHE is None:
            cls.CACHE = create_bbox_cache(use_extents_hint=False)

        # Grab aabb
        aabb = compute_aabb(bbox_cache=cls.CACHE, prim_path=prim_path)

        return aabb[:3], aabb[3:]

    @classmethod
    def compute_center_extent(cls, prim_path):
        """
        Computes the AABB (world-frame oriented) for the prim specified at @prim_path, and convert it into the center
        and extent values

        Args:
            prim_path (str): Path to the prim to calculate AABB for

        Returns:
            tuple:
                - 3-array: center position (x,y,z) of world-coordinate frame aligned bounding box
                - 3-array: end-to-end extent size (x,y,z) of world-coordinate frame aligned bounding box
        """
        low, high = cls.compute_aabb(prim_path=prim_path)

        return (low + high) / 2.0, high - low

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this BoundingBoxAPI
        """
        cls.CACHE = None


def clear():
    """
    Clear state tied to singleton classes
    """
    CollisionAPI.clear()
    BoundingBoxAPI.clear()
