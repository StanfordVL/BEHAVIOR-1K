"""
Ray-traced render backend (``gm.RENDER_BACKEND = "ovrtx"``) through Newton's ``ViewerRTX``. See newton_viewer_backend.py.

``ViewerRTX`` bakes its camera's FOV into the renderer when it initializes, so under OVRTX cameras only share a
viewer when both their resolution and their FOV match. The scene's UsdLux lights are copied into the RTX stage
once, when the viewer is built.
"""

import omnigibson as og
from omnigibson.render_backends.newton_viewer_backend import NewtonViewerRenderBackend
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)

# UsdLux light prim types copied into the RTX viewer's stage (see _copy_scene_lights_into_rtx_viewer)
_USD_LIGHT_TYPES = ("SphereLight", "RectLight", "DiskLight", "DistantLight", "DomeLight", "CylinderLight")


def _copy_scene_lights_into_rtx_viewer(viewer):
    """
    Copy every visible UsdLux light on the OmniGibson stage into ``viewer``'s own USD stage (under
    ``/root/_OGSceneLights/``), baking each light's world transform onto the copy.

    The Newton model carries only geometry, so without the scene's own lights an enclosed interior renders near-black
    under RTX (its default dome + sun rig is blocked by walls and ceilings). Lights are copied once, before the first
    frame: OVRTX serializes its stage at init and crashes on later light edits.

    Args:
        viewer (ViewerRTX): Viewer whose stage receives the lights

    Returns:
        int: Number of lights copied
    """
    from pxr import Gf, Usd, UsdGeom

    copied = 0
    for prim in og.sim.stage.Traverse():
        if prim.GetTypeName() not in _USD_LIGHT_TYPES:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        dst = viewer.stage.DefinePrim(f"/root/_OGSceneLights/light_{copied}", prim.GetTypeName())
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith("inputs:"):
                continue
            value = attr.Get()
            if value is None:
                continue
            dst.CreateAttribute(name, attr.GetTypeName()).Set(value)

        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        dst_xf = UsdGeom.Xformable(dst)
        dst_xf.ClearXformOpOrder()
        dst_xf.AddTransformOp().Set(Gf.Matrix4d(world))
        copied += 1

    log.info(f"Copied {copied} scene light(s) into the RTX viewer stage")
    return copied


class OVRTXRenderBackend(NewtonViewerRenderBackend):
    """
    Ray-traced rendering through Newton's headless ``ViewerRTX`` (ovrtx), lit by the scene's own UsdLux lights.
    """

    fov_fixed_at_build = True

    def _create_viewer(self, width, height):
        import newton

        # Default environment, not "studio": mutating the studio rig's light prims after construction crashed OVRTX
        viewer = newton.viewer.ViewerRTX(width=width, height=height, headless=True)
        _copy_scene_lights_into_rtx_viewer(viewer)
        return viewer

    def _rebind_viewer(self, viewer, model):
        # ViewerRTX serializes its stage once at init, so set_model() on a live one renders black
        viewer.close()
        return None
