import omnigibson as og

from omnigibson.macros import gm
from omnigibson.object_states import *
from omnigibson.utils.constants import PrimType, ParticleModifyCondition, ParticleModifyMethod
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

def get_obj_cfg(name, category, model, prim_type=PrimType.RIGID, scale=None, bounding_box=None, abilities=None, visual_only=False):
    global num_objs
    num_objs += 1
    return {
        "type": "DatasetObject",
        "fit_avg_dim_volume": scale is None and bounding_box is None,
        "name": name,
        "category": category,
        "model": model,
        "prim_type": prim_type,
        "position": [150, 150, 150 + num_objs * 5],
        "scale": scale,
        "bounding_box": bounding_box,
        "abilities": abilities,
        "visual_only": visual_only,
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
                get_obj_cfg("dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH, abilities={"cloth": {}}),
                get_obj_cfg("carpet", "carpet", "ctclvd", prim_type=PrimType.CLOTH, abilities={"cloth": {}}),
                get_obj_cfg("bowl", "bowl", "ajzltc"),
                get_obj_cfg("bagel", "bagel", "zlxkry", abilities=TEMP_RELATED_ABILITIES),
                get_obj_cfg("cookable_dishtowel", "dishtowel", "dtfspn", prim_type=PrimType.CLOTH, abilities={**TEMP_RELATED_ABILITIES, **{"cloth": {}}}),
                get_obj_cfg("microwave", "microwave", "hjjxmi"),
                get_obj_cfg("stove", "stove", "yhjzwg"),
                get_obj_cfg("fridge", "fridge", "dszchb"),
                get_obj_cfg("plywood", "plywood", "fkmkqa", abilities={"flammable": {}}),
                get_obj_cfg("shelf_back_panel", "shelf_back_panel", "gjsnrt", abilities={"attachable": {}}),
                get_obj_cfg("shelf_shelf", "shelf_shelf", "ymtnqa", abilities={"attachable": {}}),
                get_obj_cfg("shelf_baseboard", "shelf_baseboard", "hlhneo", abilities={"attachable": {}}),
                get_obj_cfg("bracelet", "bracelet", "thqqmo"),
                get_obj_cfg("oyster", "oyster", "enzocs"),
                get_obj_cfg("sink", "sink", "egwapq", scale=np.ones(3)),
                get_obj_cfg("stockpot", "stockpot", "dcleem", abilities={"fillable": {}}),
                get_obj_cfg("applier_dishtowel", "dishtowel", "dtfspn", abilities={"particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}}),
                get_obj_cfg("remover_dishtowel", "dishtowel", "dtfspn", abilities={"particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}}),
                get_obj_cfg("spray_bottle", "spray_bottle", "asztxi", visual_only=True, abilities={"toggleable": {}, "particleApplier": {"method": ParticleModifyMethod.PROJECTION, "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]}}}),
                get_obj_cfg("vacuum", "vacuum", "bdmsbr", visual_only=True, abilities={"toggleable": {}, "particleRemover": {"method": ParticleModifyMethod.PROJECTION, "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]}}}),
                get_obj_cfg("blender", "blender", "cwkvib", bounding_box=[0.316, 0.318, 0.649], abilities={"fillable": {}, "blender": {}, "toggleable": {}}),
                get_obj_cfg("oven", "oven", "cgtaer", bounding_box=[0.943, 0.837, 1.297]),
            ],
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": [],
                    "position": [150, 150, 100],
                    "orientation": [0, 0, 0, 1],
                }
            ]
        }

        # Make sure sim is stopped
        og.sim.stop()

        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False

        # Create the environment
        env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)


def get_random_pose(pos_low=10.0, pos_hi=20.0):
    pos = np.random.uniform(pos_low, pos_hi, 3)
    orn = T.euler2quat(np.random.uniform(-np.pi, np.pi, 3))
    return pos, orn


def place_objA_on_objB_bbox(objA, objB, x_offset=0.0, y_offset=0.0, z_offset=0.01):
    objA.keep_still()
    objB.keep_still()
    # Reset pose if cloth object
    if objA.prim_type == PrimType.CLOTH:
        objA.root_link.reset()

    objA_aabb_center, objA_aabb_extent = objA.aabb_center, objA.aabb_extent
    objB_aabb_center, objB_aabb_extent = objB.aabb_center, objB.aabb_extent
    objA_aabb_offset = objA.get_position() - objA_aabb_center

    target_objA_aabb_pos = objB_aabb_center + np.array([0, 0, (objB_aabb_extent[2] + objA_aabb_extent[2]) / 2.0]) + \
                           np.array([x_offset, y_offset, z_offset])
    objA.set_position(target_objA_aabb_pos + objA_aabb_offset)


def place_obj_on_floor_plane(obj, x_offset=0.0, y_offset=0.0, z_offset=0.01):
    obj.keep_still()
    # Reset pose if cloth object
    if obj.prim_type == PrimType.CLOTH:
        obj.root_link.reset()

    obj_aabb_center, obj_aabb_extent = obj.aabb_center, obj.aabb_extent
    obj_aabb_offset = obj.get_position() - obj_aabb_center

    target_obj_aabb_pos = np.array([0, 0, obj_aabb_extent[2] / 2.0]) + np.array([x_offset, y_offset, z_offset])
    obj.set_position(target_obj_aabb_pos + obj_aabb_offset)
