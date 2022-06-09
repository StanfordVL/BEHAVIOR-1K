import math
import numpy as np
from collections import Iterable

import omni.usd
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache, compute_combined_aabb
from omni.syntheticdata import helpers
from omni.kit.primitive.mesh.evaluators.sphere import SphereEvaluator
from omni.kit.primitive.mesh.evaluators.disk import DiskEvaluator
from omni.kit.primitive.mesh.evaluators.plane import PlaneEvaluator
from omni.kit.primitive.mesh.evaluators.cylinder import CylinderEvaluator
from omni.kit.primitive.mesh.evaluators.torus import TorusEvaluator
from omni.kit.primitive.mesh.evaluators.cone import ConeEvaluator
from omni.kit.primitive.mesh.evaluators.cube import CubeEvaluator

from pxr import Gf, Vt, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
import carb
import numpy as np
import trimesh

from igibson.utils.constants import JointType
from igibson.utils.types import PRIMITIVE_MESH_TYPES
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

MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
    "Sphere": SphereEvaluator,
    "Disk": DiskEvaluator,
    "Plane": PlaneEvaluator,
    "Cylinder": CylinderEvaluator,
    "Torus": TorusEvaluator,
    "Cone": ConeEvaluator,
    "Cube": CubeEvaluator,
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
        # TODO: This slows things down and / or crashes the sim with large number of objects. Skipping this for now, look into this later
        pass
        # # Check if collision group exists or not
        # if col_group not in cls.ACTIVE_COLLISION_GROUPS:
        #     # Raise error if we don't explicitly want to create a new group
        #     if not create_if_not_exist:
        #         raise ValueError(f"Collision group {col_group} not found in current registry, and create_if_not_exist"
        #                          f"was set to False!")
        #     # Otherwise, create the new group
        #     col_group_name = f"/World/collisionGroup_{col_group}"
        #     group = UsdPhysics.CollisionGroup.Define(get_current_stage(), col_group_name)
        #     group.GetFilteredGroupsRel().AddTarget(col_group_name)  # Make sure that we can collide within our own group
        #     cls.ACTIVE_COLLISION_GROUPS[col_group] = group
        #
        # # Add this prim to the collision group
        # cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

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

    @classmethod
    def union(cls, prim_paths):
        """
        Computes the union of AABBs (world-frame oriented) for the prims specified at @prim_paths

        Args:
            prim_paths (str): Paths to the prims to calculate union AABB for

        Returns:
            tuple:
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box
        """
        # Create cache if it doesn't already exist
        if cls.CACHE is None:
            cls.CACHE = create_bbox_cache(use_extents_hint=False)

        # Grab aabb
        aabb = compute_combined_aabb(bbox_cache=cls.CACHE, prim_paths=prim_paths)

        return aabb[:3], aabb[3:]

    @classmethod
    def aabb_contains_point(cls, point, container):
        """
        Returns true if the point is contained in the container AABB

        Args:
            point (tuple): (x,y,z) position in world-coordinates
            container (tuple):
                - 3-array: start (x,y,z) corner of world-coordinate frame aligned bounding box
                - 3-array: end (x,y,z) corner of world-coordinate frame aligned bounding box

        Returns:
            bool
        """
        lower, upper = container
        return np.less_equal(lower, point).all() and np.less_equal(point, upper).all()

def clear():
    """
    Clear state tied to singleton classes
    """
    CollisionAPI.clear()
    BoundingBoxAPI.clear()


def create_mesh_prim_with_default_xform(primitive_type, prim_path, stage=None, u_patches=None, v_patches=None):
    """
    Computes the union of AABBs (world-frame oriented) for the prims specified at @prim_paths

    Args:
        primitive_type (str): Primitive mesh type, should be one of PRIMITIVE_MESH_TYPES to be valid
        prim_path (str): Destination prim path to store the mesh prim
        stage (Usd.Stage or None): If specified, should be specific stage to be used to load the mesh prim.
            Otherwise, the current active stage will be used.
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
    """

    assert primitive_type in PRIMITIVE_MESH_TYPES, "Invalid primitive mesh type: {primitive_type}"
    evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
    u_backup = carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
    v_backup = carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
    hs_backup = carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
    carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
    carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)

    # Default half_scale (i.e. half-extent, half_height, radius) is 1.
    # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
    #  (currently it only accepts integer-number and floors 0.5 into 0).
    carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)

    stage = get_current_stage() if stage is None else stage
    prim_path_from = Sdf.Path(omni.usd.get_stage_next_free_path(stage, primitive_type, True))
    if u_patches is not None and v_patches is not None:
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type=primitive_type,
            u_patches=u_patches,
            v_patches=v_patches,
        )
    else:
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type=primitive_type,
        )
    omni.kit.commands.execute("MovePrim", path_from=prim_path_from, path_to=prim_path)

    carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
    carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
    carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)


def mesh_prim_to_trimesh_mesh(mesh_prim):
    face_vertex_counts = np.array(mesh_prim.GetAttribute("faceVertexCounts").Get())
    vertices = np.array(mesh_prim.GetAttribute("points").Get())
    face_indices = np.array(mesh_prim.GetAttribute("faceVertexIndices").Get())

    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count - 2):
            faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
        i += count

    return trimesh.Trimesh(vertices=vertices, faces=faces)

