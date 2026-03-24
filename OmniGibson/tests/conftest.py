import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition, ParticleModifyMethod, PrimType

TEMP_RELATED_ABILITIES = {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}}

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

_num_objs = 0


def _get_obj_cfg(
    name, category, model, prim_type=PrimType.RIGID, scale=None, bounding_box=None, abilities=None, visual_only=False
):
    global _num_objs
    _num_objs += 1
    return {
        "type": "DatasetObject",
        "fit_avg_dim_volume": scale is None and bounding_box is None,
        "name": name,
        "category": category,
        "model": model,
        "prim_type": prim_type,
        "position": [150, 150, 150 + _num_objs * 5],
        "scale": scale,
        "bounding_box": bounding_box,
        "abilities": abilities,
        "visual_only": visual_only,
    }


@pytest.fixture
def env(request):
    global _num_objs
    _num_objs = 0

    marker = request.node.get_closest_marker("og_objects")
    obj_names = marker.args if marker else ()
    needs_robot = marker.kwargs.get("needs_robot", False) if marker else False

    unknown = [name for name in obj_names if name not in _OBJ_KWARGS]
    if unknown:
        raise ValueError(f"Unknown object name(s): {unknown}")

    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = True

    cfg = {
        "scene": {"type": "Scene"},
        "objects": [_get_obj_cfg(**_OBJ_KWARGS[name]) for name in obj_names],
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

    og.sim.stop()
    for name in ["bagel_dough", "raw_egg"]:
        if name in obj_names:
            obj = env.scene.object_registry("name", name)
            obj.root_link.set_collision_approximation("boundingCube")
    og.sim.play()

    yield env

    og.clear()


def _make_object_fixture(obj_name):
    @pytest.fixture
    def _fixture(env):
        return env.scene.object_registry("name", obj_name)

    _fixture.__name__ = obj_name
    return _fixture


for _name in _OBJ_KWARGS:
    globals()[_name] = _make_object_fixture(_name)


def pytest_addoption(parser):
    parser.addoption("--test-args", action="store", default="", help="Extra args passed to the example under test")


def pytest_unconfigure(config):
    og.shutdown()
