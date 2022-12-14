"""
Set of rendering utility functions when working with Omni
"""
import numpy as np
from pxr import Gf
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
import omnigibson as og
from omnigibson.prims import EntityPrim, RigidPrim, VisualGeomPrim
from omnigibson.utils.physx_utils import bind_material
import omnigibson.utils.transform_utils as T


def make_glass(prim):
    """
    Links the OmniGlass material with EntityPrim, RigidPrim, or VisualGeomPrim @obj, and procedurally generates
    the necessary OmniGlass material prim if necessary.

    Args:
        prim (EntityPrim or RigidPrim or VisualGeomPrim): Desired prim to convert into glass
    """
    # Generate the set of visual meshes we'll convert into glass
    if isinstance(prim, EntityPrim):
        # Grab all visual meshes from all links
        visual_meshes = [vm for link in prim.links.values() for vm in link.visual_meshes.values()]
    elif isinstance(prim, RigidPrim):
        # Grab all visual meshes from the link
        visual_meshes = [vm for vm in prim.visual_meshes.values()]
    elif isinstance(prim, VisualGeomPrim):
        # Just use this visual mesh
        visual_meshes = [prim]
    else:
        raise ValueError(f"Inputted prim must an instance of EntityPrim, RigidPrim, or VisualGeomPrim "
                         f"in order to be converted into glass!")

    # Grab the glass material prim; if it doesn't exist, we create it on the fly
    glass_prim_path = "/Looks/OmniGlass"
    if not get_prim_at_path(glass_prim_path):
        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created,
        )

    # Iterate over all meshes and bind the glass material to the mesh
    for vm in visual_meshes:
        bind_material(vm.prim_path, material_path=glass_prim_path)


def create_pbr_material(prim_path):
    """
    Creates an omni pbr material prim at the specified @prim_path

    Args:
        prim_path (str): Prim path where the PBR material should be generated

    Returns:
        Usd.Prim: Generated PBR material prim
    """
    # Use DeepWater omni present for rendering water
    mtl_created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created,
    )
    material_path = mtl_created[0]

    # Move prim to desired location
    omni.kit.commands.execute("MovePrim", path_from=material_path, path_to=prim_path)

    # Return generated material
    return get_prim_at_path(material_path)


def create_skylight(intensity=500, color=(1.0, 1.0, 1.0)):
    """
    Creates a skylight object with the requested @color

    Args:
        intensity (float): Intensity of the generated skylight
        color (3-array): Desired (R,G,B) color to assign to the skylight

    Returns:
        LightObject: Generated skylight object
    """
    # Avoid circular imports
    from omnigibson.objects.light_object import LightObject
    light = LightObject(prim_path="/World/skylight", name="skylight", light_type="Dome", intensity=intensity)
    og.sim.import_object(light)
    light.set_orientation(T.euler2quat([0, 0, -np.pi / 4]))
    light_prim = light.light_link.prim
    light_prim.GetAttribute("color").Set(Gf.Vec3f(*color))
    return light
