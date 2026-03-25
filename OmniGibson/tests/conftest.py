import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.robots import Robot
from omnigibson.utils.constants import ParticleModifyCondition, ParticleModifyMethod, PrimType


@pytest.fixture
def stopped_env():
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = True

    yield og.Environment(configs={"scene": {"type": "Scene"}})

    og.clear()


@pytest.fixture
def env(request, stopped_env):
    for name in request.fixturenames:
        if name not in ("env", "stopped_env", "request"):
            request.getfixturevalue(name)

    og.sim.play()
    yield stopped_env


# --- Robot fixture ---


@pytest.fixture
def robot(stopped_env):
    obj = Robot(
        name="fetch",
        model="fetch",
        obs_modalities="rgb",
        position=[150, 150, 100],
        orientation=[0, 0, 0, 1],
    )
    stopped_env.scene.add_object(obj)
    return obj


# --- Object fixtures ---


@pytest.fixture
def breakfast_table(stopped_env):
    obj = DatasetObject(name="breakfast_table", category="breakfast_table", model="skczfi")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bottom_cabinet(stopped_env):
    obj = DatasetObject(name="bottom_cabinet", category="bottom_cabinet", model="immwzb")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def dishtowel(stopped_env):
    obj = DatasetObject(
        name="dishtowel", category="dishtowel", model="dtfspn", prim_type=PrimType.CLOTH, abilities={"cloth": {}}
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def carpet(stopped_env):
    obj = DatasetObject(
        name="carpet", category="carpet", model="ctclvd", prim_type=PrimType.CLOTH, abilities={"cloth": {}}
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bowl(stopped_env):
    obj = DatasetObject(name="bowl", category="bowl", model="ajzltc")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bagel(stopped_env):
    obj = DatasetObject(
        name="bagel",
        category="bagel",
        model="zlxkry",
        abilities={"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def cookable_dishtowel(stopped_env):
    obj = DatasetObject(
        name="cookable_dishtowel",
        category="dishtowel",
        model="dtfspn",
        prim_type=PrimType.CLOTH,
        abilities={"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}, "cloth": {}},
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def microwave(stopped_env):
    obj = DatasetObject(name="microwave", category="microwave", model="hjjxmi")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def stove(stopped_env):
    obj = DatasetObject(name="stove", category="stove", model="yhjzwg")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def fridge(stopped_env):
    obj = DatasetObject(name="fridge", category="fridge", model="xyejdx")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def plywood(stopped_env):
    obj = DatasetObject(name="plywood", category="plywood", model="fkmkqa", abilities={"flammable": {}})
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_back(stopped_env):
    obj = DatasetObject(name="bookcase_back", category="bookcase_back", model="gjsnrt", abilities={"attachable": {}})
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_shelf(stopped_env):
    obj = DatasetObject(name="bookcase_shelf", category="bookcase_shelf", model="ymtnqa", abilities={"attachable": {}})
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_baseboard(stopped_env):
    obj = DatasetObject(
        name="bookcase_baseboard", category="bookcase_baseboard", model="hlhneo", abilities={"attachable": {}}
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bracelet(stopped_env):
    obj = DatasetObject(name="bracelet", category="bracelet", model="thqqmo")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def furniture_sink(stopped_env):
    obj = DatasetObject(name="furniture_sink", category="furniture_sink", model="bnpjjy", scale=th.ones(3))
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def stockpot(stopped_env):
    obj = DatasetObject(
        name="stockpot", category="stockpot", model="dcleem", abilities={"fillable": {}, "heatable": {}}
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def applier_dishtowel(stopped_env):
    obj = DatasetObject(
        name="applier_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def remover_dishtowel(stopped_env):
    obj = DatasetObject(
        name="remover_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def acetone_atomizer(stopped_env):
    obj = DatasetObject(
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
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def vacuum(stopped_env):
    obj = DatasetObject(
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
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def blender(stopped_env):
    obj = DatasetObject(
        name="blender",
        category="blender",
        model="cwkvib",
        bounding_box=[0.316, 0.318, 0.649],
        abilities={"fillable": {}, "toggleable": {}, "heatable": {}},
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def oven(stopped_env):
    obj = DatasetObject(name="oven", category="oven", model="cgtaer", bounding_box=[0.943, 0.837, 1.297])
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def baking_sheet(stopped_env):
    obj = DatasetObject(name="baking_sheet", category="baking_sheet", model="yhurut")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def bagel_dough(stopped_env):
    obj = DatasetObject(name="bagel_dough", category="bagel_dough", model="iuembm", bounding_box=[0.20, 0.20, 0.02])
    stopped_env.scene.add_object(obj)
    obj.root_link.set_collision_approximation("boundingCube")
    return obj


@pytest.fixture
def raw_egg(stopped_env):
    obj = DatasetObject(name="raw_egg", category="raw_egg", model="ydgivr")
    stopped_env.scene.add_object(obj)
    obj.root_link.set_collision_approximation("boundingCube")
    return obj


@pytest.fixture
def another_raw_egg(stopped_env):
    obj = DatasetObject(name="another_raw_egg", category="raw_egg", model="ydgivr")
    stopped_env.scene.add_object(obj)
    obj.root_link.set_collision_approximation("boundingCube")
    return obj


@pytest.fixture
def scoop_of_ice_cream(stopped_env):
    obj = DatasetObject(
        name="scoop_of_ice_cream", category="scoop_of_ice_cream", model="dodndj", bounding_box=[0.076, 0.077, 0.065]
    )
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def food_processor(stopped_env):
    obj = DatasetObject(name="food_processor", category="food_processor", model="gamkbo")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def electric_mixer(stopped_env):
    obj = DatasetObject(name="electric_mixer", category="electric_mixer", model="qornxa")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def swiss_cheese(stopped_env):
    obj = DatasetObject(name="swiss_cheese", category="swiss_cheese", model="hwxeto")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def apple(stopped_env):
    obj = DatasetObject(name="apple", category="apple", model="agveuv")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def table_knife(stopped_env):
    obj = DatasetObject(name="table_knife", category="table_knife", model="jxdfyy")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def half_apple(stopped_env):
    obj = DatasetObject(name="half_apple", category="half_apple", model="sguztn")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def tablespoon(stopped_env):
    obj = DatasetObject(name="tablespoon", category="tablespoon", model="huudhe")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def chicken(stopped_env):
    obj = DatasetObject(name="chicken", category="chicken", model="nppsmz", scale=th.ones(3) * 0.7)
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def washer(stopped_env):
    obj = DatasetObject(name="washer", category="washer", model="dobgmu")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def clothes_dryer(stopped_env):
    obj = DatasetObject(name="clothes_dryer", category="clothes_dryer", model="smcyys")
    stopped_env.scene.add_object(obj)
    return obj


@pytest.fixture
def oyster(stopped_env):
    obj = DatasetObject(name="oyster", category="oyster", model="enzocs")
    stopped_env.scene.add_object(obj)
    return obj


def pytest_addoption(parser):
    parser.addoption("--test-args", action="store", default="", help="Extra args passed to the example under test")


def pytest_unconfigure(config):
    og.shutdown()
