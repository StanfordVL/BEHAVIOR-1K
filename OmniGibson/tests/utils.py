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

# Maps object name -> kwargs for get_obj_cfg. Only objects actually used in tests.
_OBJ_KWARGS = {
    "breakfast_table": dict(name="breakfast_table", category="breakfast_table", model="skczfi"),
    "bottom_cabinet": dict(name="bottom_cabinet", category="bottom_cabinet", model="immwzb"),
    "dishtowel": dict(
        name="dishtowel", category="dishtowel", model="dtfspn", prim_type=PrimType.CLOTH, abilities={"cloth": {}}
    ),
    "carpet": dict(name="carpet", category="carpet", model="ctclvd", prim_type=PrimType.CLOTH, abilities={"cloth": {}}),
    "bowl": dict(name="bowl", category="bowl", model="ajzltc"),
    "bagel": dict(name="bagel", category="bagel", model="zlxkry", abilities=TEMP_RELATED_ABILITIES),
    "cookable_dishtowel": dict(
        name="cookable_dishtowel",
        category="dishtowel",
        model="dtfspn",
        prim_type=PrimType.CLOTH,
        abilities={**TEMP_RELATED_ABILITIES, **{"cloth": {}}},
    ),
    "microwave": dict(name="microwave", category="microwave", model="hjjxmi"),
    "stove": dict(name="stove", category="stove", model="yhjzwg"),
    "fridge": dict(name="fridge", category="fridge", model="xyejdx"),
    "plywood": dict(name="plywood", category="plywood", model="fkmkqa", abilities={"flammable": {}}),
    "bookcase_back": dict(name="bookcase_back", category="bookcase_back", model="gjsnrt", abilities={"attachable": {}}),
    "bookcase_shelf": dict(
        name="bookcase_shelf", category="bookcase_shelf", model="ymtnqa", abilities={"attachable": {}}
    ),
    "bookcase_baseboard": dict(
        name="bookcase_baseboard", category="bookcase_baseboard", model="hlhneo", abilities={"attachable": {}}
    ),
    "bracelet": dict(name="bracelet", category="bracelet", model="thqqmo"),
    "furniture_sink": dict(name="furniture_sink", category="furniture_sink", model="bnpjjy", scale=th.ones(3)),
    "stockpot": dict(name="stockpot", category="stockpot", model="dcleem", abilities={"fillable": {}, "heatable": {}}),
    "applier_dishtowel": dict(
        name="applier_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    ),
    "remover_dishtowel": dict(
        name="remover_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    ),
    "acetone_atomizer": dict(
        name="acetone_atomizer",
        category="acetone_atomizer",
        model="krtwsl",
        visual_only=True,
        abilities={
            "toggleable": {},
            "particleApplier": {
                "method": ParticleModifyMethod.PROJECTION,
                "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]},
            },
        },
    ),
    "vacuum": dict(
        name="vacuum",
        category="vacuum",
        model="bdmsbr",
        visual_only=True,
        abilities={
            "toggleable": {},
            "particleRemover": {
                "method": ParticleModifyMethod.PROJECTION,
                "conditions": {"water": [(ParticleModifyCondition.TOGGLEDON, True)]},
            },
        },
    ),
    "blender": dict(
        name="blender",
        category="blender",
        model="cwkvib",
        bounding_box=[0.316, 0.318, 0.649],
        abilities={"fillable": {}, "toggleable": {}, "heatable": {}},
    ),
    "oven": dict(name="oven", category="oven", model="cgtaer", bounding_box=[0.943, 0.837, 1.297]),
    "baking_sheet": dict(name="baking_sheet", category="baking_sheet", model="yhurut"),
    "bagel_dough": dict(name="bagel_dough", category="bagel_dough", model="iuembm", bounding_box=[0.20, 0.20, 0.02]),
    "raw_egg": dict(name="raw_egg", category="raw_egg", model="ydgivr"),
    "another_raw_egg": dict(name="another_raw_egg", category="raw_egg", model="ydgivr"),
    "scoop_of_ice_cream": dict(
        name="scoop_of_ice_cream", category="scoop_of_ice_cream", model="dodndj", bounding_box=[0.076, 0.077, 0.065]
    ),
    "food_processor": dict(name="food_processor", category="food_processor", model="gamkbo"),
    "electric_mixer": dict(name="electric_mixer", category="electric_mixer", model="qornxa"),
    "swiss_cheese": dict(name="swiss_cheese", category="swiss_cheese", model="hwxeto"),
    "apple": dict(name="apple", category="apple", model="agveuv"),
    "table_knife": dict(name="table_knife", category="table_knife", model="jxdfyy"),
    "half_apple": dict(name="half_apple", category="half_apple", model="sguztn"),
    "tablespoon": dict(name="tablespoon", category="tablespoon", model="huudhe"),
    "chicken": dict(name="chicken", category="chicken", model="nppsmz", scale=th.ones(3) * 0.7),
    "washer": dict(name="washer", category="washer", model="dobgmu"),
    "clothes_dryer": dict(name="clothes_dryer", category="clothes_dryer", model="smcyys"),
    "oyster": dict(name="oyster", category="oyster", model="enzocs"),
}

num_objs = 0


def og_test(*obj_names, needs_robot=False):
    """
    Decorator factory that creates a fresh minimal environment for each test with only
    the specified objects. On the first use it configures required macros and launches
    the simulator; on subsequent uses it clears the existing simulator via og.clear()
    before creating a new environment to ensure test isolation.

    Usage:
        @og_test("breakfast_table", "bowl")
        def test_something(env):
            ...

        @og_test("stove", needs_robot=True)
        def test_with_robot(env):
            ...
    """
    unknown = [name for name in obj_names if name not in _OBJ_KWARGS]
    if unknown:
        raise ValueError(f"Unknown object name(s): {unknown}")

    def decorator(func):
        global num_objs
        num_objs = 0

        if og.sim is None:
            # First test: set required macros before launching
            gm.ENABLE_OBJECT_STATES = True
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_FLATCACHE = False
            gm.ENABLE_TRANSITION_RULES = True
        else:
            # Subsequent tests: fully reset stage and relaunch sim
            og.clear()

        cfg = {
            "scene": {"type": "Scene"},
            "objects": [get_obj_cfg(**_OBJ_KWARGS[name]) for name in obj_names],
        }
        if needs_robot:
            cfg["robots"] = [
                {
                    "type": "Fetch",
                    "obs_modalities": "rgb",
                    "position": [150, 150, 100],
                    "orientation": [0, 0, 0, 1],
                }
            ]

        env = og.Environment(configs=cfg)

        # Additional processing for the tests to pass more deterministically
        og.sim.stop()
        bounding_box_object_names = ["bagel_dough", "raw_egg"]
        for name in bounding_box_object_names:
            obj = env.scene.object_registry("name", name)
            obj.root_link.set_collision_approximation("boundingCube")
        og.sim.play()

        func(env)

    return decorator


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
