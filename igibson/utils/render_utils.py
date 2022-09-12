"""Set of rendering utility functions when working with Omni"""

import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from igibson.prims import EntityPrim, RigidPrim, VisualGeomPrim
from igibson.utils.physx_utils import bind_material


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
