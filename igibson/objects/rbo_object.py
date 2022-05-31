import os

import igibson
from igibson.objects.object_base import BaseObject
from omni.isaac.core.objects import FixedCuboid


# TODO: This does not work!
class RBOObject(BaseObject):
    """
    RBO object from assets/models/rbo
    Reference: https://tu-rbo.github.io/articulated-objects/
    """

    def __init__(
            self,
            prim_path,
            usd_path,
            name=None,
            category="object",
            class_id=None,
            scale=None,
            rendering_params=None,
            visible=True,
    ):
        filename = os.path.join(igibson.assets_path, "models", "rbo", name, "configuration", "{}.urdf".format(name))
        super(RBOObject, self).__init__(filename, scale)
