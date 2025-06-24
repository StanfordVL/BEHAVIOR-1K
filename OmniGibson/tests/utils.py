import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.systems import FluidSystem, GranularSystem, MacroPhysicalParticleSystem, MacroVisualParticleSystem
from omnigibson.utils.constants import ParticleModifyCondition, ParticleModifyMethod, PrimType

TEMP_RELATED_ABILITIES = {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}}

SYSTEM_EXAMPLES = {
    "water": FluidSystem,
    "white_rice": GranularSystem,
    "diced__apple": MacroPhysicalParticleSystem,
    "stain": MacroVisualParticleSystem,
}

env = None


def og_test(func):
    def wrapper():
        assert_test_env()
        try:
            func(env)
        finally:
            og.sim.step()  # Make sure any objects that need to be initialized are initialized
            env.scene.reset()

    return wrapper


num_objs = 0


def retrieve_obj_cfg(obj):
    return {
        "name": obj.name,
        "category": obj.category,
        "model": obj.model,
        "prim_type": obj.prim_type,
        "position": obj.get_position_orientation()[0],
        "scale": obj.scale,
        "abilities": obj.abilities,
        "visual_only": obj.visual_only,
    }


def get_obj_cfg(
    name, category, model, prim_type=PrimType.RIGID, scale=None, bounding_box=None, abilities=None, visual_only=False
):
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


def assert_test_env():
    global env
    if env is None:
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
                get_obj_cfg(
                    "cookable_dishtowel",
                    "dishtowel",
                    "dtfspn",
                    prim_type=PrimType.CLOTH,
                    abilities={**TEMP_RELATED_ABILITIES, **{"cloth": {}}},
                ),
                get_obj_cfg("microwave", "microwave", "hjjxmi"),
                get_obj_cfg("stove", "stove", "yhjzwg"),
                get_obj_cfg("fridge", "fridge", "xyejdx"),
                get_obj_cfg("plywood", "plywood", "fkmkqa", abilities={"flammable": {}}),
                get_obj_cfg("bookcase_back", "bookcase_back", "gjsnrt", abilities={"attachable": {}}),
                get_obj_cfg("bookcase_shelf", "bookcase_shelf", "ymtnqa", abilities={"attachable": {}}),
                get_obj_cfg("bookcase_baseboard", "bookcase_baseboard", "hlhneo", abilities={"attachable": {}}),
                get_obj_cfg("bracelet", "bracelet", "thqqmo"),
                get_obj_cfg("oyster", "oyster", "enzocs"),
                get_obj_cfg("furniture_sink", "furniture_sink", "bnpjjy", scale=th.ones(3)),
                get_obj_cfg("stockpot", "stockpot", "dcleem", abilities={"fillable": {}, "heatable": {}}),
                get_obj_cfg(
                    "applier_dishtowel",
                    "dishtowel",
                    "dtfspn",
                    abilities={
                        "particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}
                    },
                ),
                get_obj_cfg(
                    "remover_dishtowel",
                    "dishtowel",
                    "dtfspn",
                    abilities={
                        "particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}
                    },
                ),
                get_obj_cfg(
                    "acetone_atomizer",
                    "acetone_atomizer",
                    "krtwsl",
                    visual_only=True,
                    abilities={
                        "toggleable": {},
                        "particleApplier": {
                            "method": ParticleModifyMethod.PROJECTION,
                            "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]},
                        },
                    },
                ),
                get_obj_cfg(
                    "vacuum",
                    "vacuum",
                    "bdmsbr",
                    visual_only=True,
                    abilities={
                        "toggleable": {},
                        "particleRemover": {
                            "method": ParticleModifyMethod.PROJECTION,
                            "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]},
                        },
                    },
                ),
                get_obj_cfg(
                    "blender",
                    "blender",
                    "cwkvib",
                    bounding_box=[0.316, 0.318, 0.649],
                    abilities={"fillable": {}, "toggleable": {}, "heatable": {}},
                ),
                get_obj_cfg("oven", "oven", "cgtaer", bounding_box=[0.943, 0.837, 1.297]),
                get_obj_cfg("baking_sheet", "baking_sheet", "yhurut"),
                get_obj_cfg("bagel_dough", "bagel_dough", "iuembm", bounding_box=[0.20, 0.20, 0.02]),
                get_obj_cfg("raw_egg", "raw_egg", "ydgivr"),
                get_obj_cfg("scoop_of_ice_cream", "scoop_of_ice_cream", "dodndj", bounding_box=[0.076, 0.077, 0.065]),
                get_obj_cfg("food_processor", "food_processor", "gamkbo"),
                get_obj_cfg("electric_mixer", "electric_mixer", "qornxa"),
                get_obj_cfg("another_raw_egg", "raw_egg", "ydgivr"),
                get_obj_cfg("chicken", "chicken", "nppsmz", scale=th.ones(3) * 0.7),
                get_obj_cfg("tablespoon", "tablespoon", "huudhe"),
                get_obj_cfg("swiss_cheese", "swiss_cheese", "hwxeto"),
                get_obj_cfg("apple", "apple", "agveuv"),
                get_obj_cfg("table_knife", "table_knife", "jxdfyy"),
                get_obj_cfg("half_apple", "half_apple", "sguztn"),
                get_obj_cfg("washer", "washer", "dobgmu"),
                get_obj_cfg("carpet_sweeper", "carpet_sweeper", "xboreo"),
                get_obj_cfg("clothes_dryer", "clothes_dryer", "smcyys"),
            ],
            "robots": [
                {
                    "type": "Fetch",
                    "obs_modalities": "rgb",
                    "position": [150, 150, 100],
                    "orientation": [0, 0, 0, 1],
                }
            ],
        }

        if og.sim is None:
            # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
            gm.ENABLE_OBJECT_STATES = True
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_FLATCACHE = False
            gm.ENABLE_TRANSITION_RULES = True
        else:
            # Make sure sim is stopped
            og.sim.stop()

        # Create the environment
        env = og.Environment(configs=cfg)

        # Additional processing for the tests to pass more deterministically
        og.sim.stop()
        bounding_box_object_names = ["bagel_dough", "raw_egg"]
        for name in bounding_box_object_names:
            obj = env.scene.object_registry("name", name)
            for collision_mesh in obj.root_link.collision_meshes.values():
                collision_mesh.set_collision_approximation("boundingCube")
        og.sim.play()

    assert env is not None, "Environment not created"


def get_random_pose(pos_low=10.0, pos_hi=20.0):
    pos = th.rand(3) * (pos_hi - pos_low) + pos_low
    ori_lo, ori_hi = -math.pi, math.pi
    orn = T.euler2quat(th.rand(3) * (ori_hi - ori_lo) + ori_lo)
    return pos, orn


def place_objA_on_objB_bbox(objA, objB, x_offset=0.0, y_offset=0.0, z_offset=0.001):
    objA.keep_still()
    objB.keep_still()
    # Reset pose if cloth object
    if objA.prim_type == PrimType.CLOTH:
        objA.root_link.reset()

    objA_aabb_center, objA_aabb_extent = objA.aabb_center, objA.aabb_extent
    objB_aabb_center, objB_aabb_extent = objB.aabb_center, objB.aabb_extent
    objA_aabb_offset = objA.get_position_orientation()[0] - objA_aabb_center

    target_objA_aabb_pos = (
        objB_aabb_center
        + th.tensor([0, 0, (objB_aabb_extent[2] + objA_aabb_extent[2]) / 2.0])
        + th.tensor([x_offset, y_offset, z_offset])
    )
    objA.set_position_orientation(position=target_objA_aabb_pos + objA_aabb_offset)


def place_obj_on_floor_plane(obj, x_offset=0.0, y_offset=0.0, z_offset=0.01):
    obj.keep_still()
    # Reset pose if cloth object
    if obj.prim_type == PrimType.CLOTH:
        obj.root_link.reset()

    obj_aabb_center, obj_aabb_extent = obj.aabb_center, obj.aabb_extent
    obj_aabb_offset = obj.get_position_orientation()[0] - obj_aabb_center

    target_obj_aabb_pos = th.tensor([0, 0, obj_aabb_extent[2] / 2.0]) + th.tensor([x_offset, y_offset, z_offset])
    obj.set_position_orientation(position=target_obj_aabb_pos + obj_aabb_offset)


def remove_all_systems(scene):
    for system in scene.active_systems.values():
        system.remove_all_particles()
    og.sim.step()
