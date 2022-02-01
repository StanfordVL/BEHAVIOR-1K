from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
from pxr import Gf, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema


JOINT_TYPES = {
    "Joint",
    "FixedJoint",
    "PrismaticJoint",
    "RevoluteJoint",
    "SphericalJoint",
}


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


def create_joint(prim_path, joint_type, body0=None, body1=None, stage=None):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    :param prim_path: str, absolute path to where the joint will be created
    :param joint_type: str, type of joint to create. Valid options are:
        "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
    :param body0: str, absolute path to the first body's prim. At least @body0 or @body1 must be specified.
    :param body1: str, absolute path to the second body's prim. At least @body0 or @body1 must be specified.
    :param stage: Usd.Stage, if specified, should be specific stage to be used to load the joint.
        Otherwise, the current active stage will be used.

    :return Usd.Prim: Created joint prim
    """
    # Make sure we have valid joint_type
    assert joint_type in JOINT_TYPES, \
        f"Invalid joint specified for creation: {joint_type}"

    # Make sure at least body0 or body1 is specified
    assert body0 is not None and body1 is not None, \
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

    # Apply this joint
    PhysxSchema.PhysxJointAPI.Apply()

    # Get the prim pointed to at this path
    joint_prim = get_prim_at_path(prim_path)

    # Return this joint
    return joint_prim
