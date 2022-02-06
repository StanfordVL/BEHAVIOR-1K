from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
from pxr import Gf, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema

from igibson.utils.constants import JointType


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
            cls.ACTIVE_COLLISION_GROUPS[col_group] = UsdPhysics.CollisionGroup.Define(
                get_current_stage(), f"/World/collisionGroup_{col_group}"
            )

        # Add this prim to the collision group
        cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)


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
            cls.CACHE = create_bbox_cache(use_extents_hint=True)

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
