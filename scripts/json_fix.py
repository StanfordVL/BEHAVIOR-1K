import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.constants import semantic_class_name_to_id, semantic_class_id_to_name
import omnigibson.utils.transform_utils as T
import numpy as np
from omnigibson.utils.asset_utils import decrypt_file, encrypt_file

config = {
    "scene": {
        "type": "Scene",
    },
}

env = og.Environment(configs=config)

encrypted_usd_path = (
    "/scr/OmniGibson/omnigibson/data/og_dataset/objects/breakfast_table/skczfi/usd/skczfi.encrypted.usd"
)
usd_path = "/scr/OmniGibson/omnigibson/data/og_dataset/objects/breakfast_table/skczfi/usd/skczfi.usd"

decrypt_file(encrypted_usd_path, usd_path)

lazy.omni.isaac.core.utils.stage.open_stage(usd_path)
stage = lazy.omni.isaac.core.utils.stage.get_current_stage()
prim = stage.GetDefaultPrim()

metadata = prim.GetCustomData()

breakpoint()


# recursively go through this entire metadata dictionary, and replace all lazy.pxr.Vt.FloatArray (of length 3) with lazy.pxr.Gf.Vec3f
def replace_float_array_with_vec3f(d):
    for k, v in d.items():
        if isinstance(v, dict):
            replace_float_array_with_vec3f(v)
        # elif isinstance(v, lazy.pxr.Vt.FloatArray) and len(v) == 3:
        elif isinstance(v, lazy.pxr.Gf.Vec3f):
            d[k] = [v[0], v[1], v[2]]
        else:
            continue
    return d


metadata = replace_float_array_with_vec3f(metadata)

prim.SetCustomData(metadata)

stage.Save()

encrypt_file(usd_path, encrypted_usd_path)

del stage
