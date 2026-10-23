"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""

import carb
import numpy as np
import omni
from omni.kit.primitive.mesh.command import CreateMeshPrimWithDefaultXformCommand as CMPWDXC
from omni.kit.primitive.mesh.command import _get_all_evaluators
from omni.replicator.core import random_colours
from PIL import Image, ImageDraw


class CreateMeshPrimWithDefaultXformCommand(CMPWDXC):
    def __init__(self, prim_type: str, **kwargs):
        """
        Creates primitive.

        Args:
            prim_type (str): It supports Plane/Sphere/Cone/Cylinder/Disk/Torus/Cube.

        kwargs:
            object_origin (Gf.Vec3f): Position of mesh center in stage units.

            u_patches (int): The number of patches to tessellate U direction.

            v_patches (int): The number of patches to tessellate V direction.

            w_patches (int): The number of patches to tessellate W direction.
                             It only works for Cone/Cylinder/Cube.

            half_scale (float): Half size of mesh in centimeters. Default is None, which means it's controlled by settings.

            u_verts_scale (int): Tessellation Level of U. It's a multiplier of u_patches.

            v_verts_scale (int): Tessellation Level of V. It's a multiplier of v_patches.

            w_verts_scale (int): Tessellation Level of W. It's a multiplier of w_patches.
                                 It only works for Cone/Cylinder/Cube.
                                 For Cone/Cylinder, it's to tessellate the caps.
                                 For Cube, it's to tessellate along z-axis.

            above_ground (bool): It will offset the center of mesh above the ground plane if it's True,
                False otherwise. It's False by default. This param only works when param object_origin is not given.
                Otherwise, it will be ignored.

            stage (Usd.Stage): If specified, stage to create prim on
        """

        self._prim_type = prim_type[0:1].upper() + prim_type[1:].lower()
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage = kwargs.get("stage", self._usd_context.get_stage())
        self._settings = carb.settings.get_settings()
        self._default_path = kwargs.get("prim_path", None)
        self._select_new_prim = kwargs.get("select_new_prim", True)
        self._prepend_default_prim = kwargs.get("prepend_default_prim", True)
        self._above_round = kwargs.get("above_ground", False)

        self._attributes = {**kwargs}
        # Supported mesh types should have an associated evaluator class
        self._evaluator_class = _get_all_evaluators()[prim_type]
        assert isinstance(self._evaluator_class, type)


def colorize_bboxes(bboxes_2d_data, bboxes_2d_rgb, num_channels=3):
    """Colorizes 2D bounding box data for visualization.

    We are overriding the replicator native version of this function to fix a bug.
    In their version of this function, the ordering of the rectangle corners is incorrect and we fix it here.

    Args:
        bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
        bboxes_2d_rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
        num_channels (int): Specify number of channels i.e. 3 or 4.
    """
    semantic_id_list = []
    bbox_2d_list = []
    rgb_img = Image.fromarray(bboxes_2d_rgb)
    rgb_img_draw = ImageDraw.Draw(rgb_img)
    for bbox_2d in bboxes_2d_data:
        semantic_id_list.append(bbox_2d[0])
        bbox_2d_list.append(bbox_2d)
    semantic_id_list_np = np.unique(np.array(semantic_id_list))
    color_list = random_colours(len(semantic_id_list_np.tolist()), False, num_channels)
    for bbox_2d in bbox_2d_list:
        index = np.where(semantic_id_list_np == bbox_2d[0])[0][0]
        bbox_color = color_list[index]
        outline = (bbox_color[0], bbox_color[1], bbox_color[2])
        if num_channels == 4:
            outline = (
                bbox_color[0],
                bbox_color[1],
                bbox_color[2],
                bbox_color[3],
            )
        rgb_img_draw.rectangle([(bbox_2d[1], bbox_2d[2]), (bbox_2d[3], bbox_2d[4])], outline=outline, width=2)
    bboxes_2d_rgb = np.array(rgb_img)
    return bboxes_2d_rgb
