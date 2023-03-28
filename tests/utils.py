import omnigibson as og

from omnigibson.macros import gm
from omnigibson.utils.constants import PrimType
import omnigibson.utils.transform_utils as T
import numpy as np


TEMP_RELATED_ABILITIES = {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}}

def og_test(func):
    def wrapper():
        assert_test_scene()
        try:
            func()
        finally:
            og.sim.scene.reset()
    return wrapper

num_objs = 0

def get_obj_cfg(name, category, model, prim_type=PrimType.RIGID, scale=None, abilities=None):
    global num_objs
    num_objs += 1
    return {
        "type": "DatasetObject",
        "fit_avg_dim_volume": scale is None,
        "name": name,
        "category": category,
        "model": model,
        "prim_type": prim_type,
        "position": [150, 150, num_objs * 5],
        "scale": scale,
        "abilities": abilities,
    }

def assert_test_scene():
    if og.sim.scene is None:
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                get_obj_cfg("breakfast_table", "breakfast_table", "skczfi"),
                get_obj_cfg("bottom_cabinet", "bottom_cabinet", "immwzb"),
                get_obj_cfg("dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH),
                get_obj_cfg("carpet", "carpet", "ctclvd", prim_type=PrimType.CLOTH),
                get_obj_cfg("bowl", "bowl", "ajzltc"),
                get_obj_cfg("bagel", "bagel", "zlxkry", abilities=TEMP_RELATED_ABILITIES),
                get_obj_cfg("cookable_dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH, abilities=TEMP_RELATED_ABILITIES),
                get_obj_cfg("microwave", "microwave", "hjjxmi"),
                get_obj_cfg("stove", "stove", "yhjzwg"),
                get_obj_cfg("fridge", "fridge", "dszchb"),
                get_obj_cfg("plywood", "plywood", "fkmkqa", abilities={"flammable": {}}),
            ],
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": [],
                }
            ]
        }

        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True

        # Create the environment
        env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
        env.robots[0].set_position_orientation([150, 150, 0], [0, 0, 0, 1])
        og.sim.step()
        og.sim.scene.update_initial_state()

def get_random_pose(pos_low=10.0, pos_hi=20.0):
    pos = np.random.uniform(pos_low, pos_hi, 3)
    orn = T.euler2quat(np.random.uniform(-np.pi, np.pi, 3))
    return pos, orn
