"""
Set of rendering utility functions when working with Omni
"""

import omnigibson.lazy as lazy
from omnigibson.utils.physx_utils import bind_material
from omnigibson.utils.ui_utils import create_module_logger


# Create module logger
log = create_module_logger(module_name=__name__)


def make_glass(prim):
    """
    Links the OmniGlass material with EntityPrim, RigidPrim, or VisualGeomPrim @obj, and procedurally generates
    the necessary OmniGlass material prim if necessary.

    Args:
        prim (EntityPrim or RigidPrim or VisualGeomPrim): Desired prim to convert into glass
    """
    # Do this here to avoid circular imports
    from omnigibson.prims import EntityPrim, RigidPrim, VisualGeomPrim

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
        raise ValueError(
            "Inputted prim must an instance of EntityPrim, RigidPrim, or VisualGeomPrim "
            "in order to be converted into glass!"
        )

    # Grab the glass material prim; if it doesn't exist, we create it on the fly
    glass_prim_path = "/Looks/OmniGlass"
    if not lazy.isaacsim.core.utils.prims.get_prim_at_path(glass_prim_path):
        mtl_created = []
        lazy.omni.kit.commands.execute(
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
    lazy.omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created,
    )
    material_path = mtl_created[0]

    # Move prim to desired location
    lazy.omni.kit.commands.execute("MovePrim", path_from=material_path, path_to=prim_path)

    # Return generated material
    return lazy.isaacsim.core.utils.prims.get_prim_at_path(material_path)


def force_pbr_material_for_link(entity_prim, link_name):
    if "meta__" in link_name:
        # Don't override the meta link material
        return

    entity_prim_path = entity_prim.GetPrimPath().__str__()
    looks_prim = entity_prim.GetChild("Looks")
    link_prim = entity_prim.GetChild(link_name)
    assert link_prim, f"Could not find link {link_name} for {entity_prim_path}"

    link_visuals_prim = link_prim.GetChild("visuals")
    if not looks_prim:
        log.debug(f"Could not find Looks prim for {entity_prim_path}")
        return

    if not link_visuals_prim:
        log.warning(f"Could not find visuals prim for link {link_name} of {entity_prim_path}")
        return

    binding_api = (
        lazy.pxr.UsdShade.MaterialBindingAPI(link_visuals_prim)
        if link_visuals_prim.HasAPI(lazy.pxr.UsdShade.MaterialBindingAPI)
        else lazy.pxr.UsdShade.MaterialBindingAPI.Apply(link_visuals_prim)
    )

    material_path = binding_api.GetDirectBinding().GetMaterialPath().pathString
    if "OmniGlass" in material_path:
        # Don't override the glass material
        return

    # Find the material prim that has the link's name in it
    link_pbr_material_pattern = f"__{link_name}_pbr"
    for mtl_prim in looks_prim.GetChildren():
        mtl_prim_name = mtl_prim.GetName()
        if link_pbr_material_pattern in mtl_prim_name:
            # Bind that material and stop
            lazy.omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=link_visuals_prim.GetPrimPath().__str__(),
                material_path=mtl_prim.GetPrimPath().__str__(),
                strength=None,
            )
            return
    else:
        log.warning(f"Could not find PBR material for link {link_name} of {entity_prim_path}")
