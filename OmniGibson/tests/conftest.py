import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.robots import Robot
from omnigibson.utils.constants import ParticleModifyCondition, ParticleModifyMethod, PrimType


@pytest.fixture
def env():
    if og.sim is None:
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = True

    yield og.Environment(configs={"scene": {"type": "Scene"}})

    og.clear()


# --- Robot fixture ---


@pytest.fixture
def robot(env):
    obj = Robot(
        name="fetch",
        model="fetch",
        obs_modalities="rgb",
        position=[150, 150, 100],
        orientation=[0, 0, 0, 1],
    )
    env.scene.add_object(obj)
    return obj


# --- Object fixtures ---


@pytest.fixture
def breakfast_table(env):
    obj = DatasetObject(name="breakfast_table", category="breakfast_table", model="skczfi")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bottom_cabinet(env):
    obj = DatasetObject(name="bottom_cabinet", category="bottom_cabinet", model="immwzb")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def dishtowel(env):
    obj = DatasetObject(
        name="dishtowel", category="dishtowel", model="dtfspn", prim_type=PrimType.CLOTH, abilities={"cloth": {}}
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def carpet(env):
    obj = DatasetObject(
        name="carpet", category="carpet", model="ctclvd", prim_type=PrimType.CLOTH, abilities={"cloth": {}}
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bowl(env):
    obj = DatasetObject(name="bowl", category="bowl", model="ajzltc")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bagel(env):
    obj = DatasetObject(
        name="bagel",
        category="bagel",
        model="zlxkry",
        abilities={"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def cookable_dishtowel(env):
    obj = DatasetObject(
        name="cookable_dishtowel",
        category="dishtowel",
        model="dtfspn",
        prim_type=PrimType.CLOTH,
        abilities={"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}, "cloth": {}},
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def microwave(env):
    obj = DatasetObject(name="microwave", category="microwave", model="hjjxmi")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def stove(env):
    obj = DatasetObject(name="stove", category="stove", model="yhjzwg")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def fridge(env):
    obj = DatasetObject(name="fridge", category="fridge", model="xyejdx")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def plywood(env):
    obj = DatasetObject(name="plywood", category="plywood", model="fkmkqa", abilities={"flammable": {}})
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_back(env):
    obj = DatasetObject(name="bookcase_back", category="bookcase_back", model="gjsnrt", abilities={"attachable": {}})
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_shelf(env):
    obj = DatasetObject(name="bookcase_shelf", category="bookcase_shelf", model="ymtnqa", abilities={"attachable": {}})
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bookcase_baseboard(env):
    obj = DatasetObject(
        name="bookcase_baseboard", category="bookcase_baseboard", model="hlhneo", abilities={"attachable": {}}
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bracelet(env):
    obj = DatasetObject(name="bracelet", category="bracelet", model="thqqmo")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def furniture_sink(env):
    obj = DatasetObject(name="furniture_sink", category="furniture_sink", model="bnpjjy", scale=th.ones(3))
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def stockpot(env):
    obj = DatasetObject(
        name="stockpot", category="stockpot", model="dcleem", abilities={"fillable": {}, "heatable": {}}
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def applier_dishtowel(env):
    obj = DatasetObject(
        name="applier_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleApplier": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def remover_dishtowel(env):
    obj = DatasetObject(
        name="remover_dishtowel",
        category="dishtowel",
        model="dtfspn",
        abilities={"particleRemover": {"method": ParticleModifyMethod.ADJACENCY, "conditions": {"water": []}}},
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def acetone_atomizer(env):
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
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def vacuum(env):
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
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def blender(env):
    obj = DatasetObject(
        name="blender",
        category="blender",
        model="cwkvib",
        bounding_box=[0.316, 0.318, 0.649],
        abilities={"fillable": {}, "toggleable": {}, "heatable": {}},
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def oven(env):
    obj = DatasetObject(name="oven", category="oven", model="cgtaer", bounding_box=[0.943, 0.837, 1.297])
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def baking_sheet(env):
    obj = DatasetObject(name="baking_sheet", category="baking_sheet", model="yhurut")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def bagel_dough(env):
    obj = DatasetObject(name="bagel_dough", category="bagel_dough", model="iuembm", bounding_box=[0.20, 0.20, 0.02])
    env.scene.add_object(obj)
    og.sim.stop()
    obj.root_link.set_collision_approximation("boundingCube")
    og.sim.play()
    return obj


@pytest.fixture
def raw_egg(env):
    obj = DatasetObject(name="raw_egg", category="raw_egg", model="ydgivr")
    env.scene.add_object(obj)
    og.sim.stop()
    obj.root_link.set_collision_approximation("boundingCube")
    og.sim.play()
    return obj


@pytest.fixture
def another_raw_egg(env):
    obj = DatasetObject(name="another_raw_egg", category="raw_egg", model="ydgivr")
    env.scene.add_object(obj)
    og.sim.stop()
    obj.root_link.set_collision_approximation("boundingCube")
    og.sim.play()
    return obj


@pytest.fixture
def scoop_of_ice_cream(env):
    obj = DatasetObject(
        name="scoop_of_ice_cream", category="scoop_of_ice_cream", model="dodndj", bounding_box=[0.076, 0.077, 0.065]
    )
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def food_processor(env):
    obj = DatasetObject(name="food_processor", category="food_processor", model="gamkbo")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def electric_mixer(env):
    obj = DatasetObject(name="electric_mixer", category="electric_mixer", model="qornxa")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def swiss_cheese(env):
    obj = DatasetObject(name="swiss_cheese", category="swiss_cheese", model="hwxeto")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def apple(env):
    obj = DatasetObject(name="apple", category="apple", model="agveuv")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def table_knife(env):
    obj = DatasetObject(name="table_knife", category="table_knife", model="jxdfyy")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def half_apple(env):
    obj = DatasetObject(name="half_apple", category="half_apple", model="sguztn")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def tablespoon(env):
    obj = DatasetObject(name="tablespoon", category="tablespoon", model="huudhe")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def chicken(env):
    obj = DatasetObject(name="chicken", category="chicken", model="nppsmz", scale=th.ones(3) * 0.7)
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def washer(env):
    obj = DatasetObject(name="washer", category="washer", model="dobgmu")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def clothes_dryer(env):
    obj = DatasetObject(name="clothes_dryer", category="clothes_dryer", model="smcyys")
    env.scene.add_object(obj)
    return obj


@pytest.fixture
def oyster(env):
    obj = DatasetObject(name="oyster", category="oyster", model="enzocs")
    env.scene.add_object(obj)
    return obj


def pytest_addoption(parser):
    parser.addoption("--test-args", action="store", default="", help="Extra args passed to the example under test")


def pytest_unconfigure(config):
    og.shutdown()
