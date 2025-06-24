import glob
import os

import pymxs

ROOT_DIR = (
    r"C:\Users\cgokmen\research\iGibson\igibson\data\scene-conversion\Wainscott_1_int"
)
rt = pymxs.runtime


def get_layer(name):
    if name == "":
        name = "0"

    rt.LayerManager.NewLayerFromName(name)
    layer = rt.LayerManager.GetLayerFromName(name)
    assert layer is not None
    return layer


objs = list(glob.glob(os.path.join(ROOT_DIR, "*", "*.obj")))
for fullpath in objs:
    file_name = os.path.split(os.path.split(fullpath)[0])[1]
    obj_name, rooms = file_name.split("---")
    rt.importFile(fullpath, pymxs.runtime.Name("noPrompt"))
    obj = rt.selection[0]
    obj.name = obj_name

    layer_name = rooms.replace("-", ",")
    get_layer(layer_name).addNode(obj)

# Set the reflection color of the vray materials to be 0
# for obj in rt.objects: obj.material.reflection = rt.Color(0, 0, 0)
