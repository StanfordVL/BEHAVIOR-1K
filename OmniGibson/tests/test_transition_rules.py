import math

import torch as th
from utils import og_test, place_obj_on_floor_plane, remove_all_systems, retrieve_obj_cfg

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import macros as m
from omnigibson.object_states import (
    Contains,
    Cooked,
    Covered,
    Heated,
    HeatSourceOrSink,
    Inside,
    OnTop,
    Open,
    Saturated,
    Temperature,
    ToggledOn,
    Touching,
)
from omnigibson.objects import DatasetObject
from omnigibson.transition_rules import REGISTERED_RULES


@og_test
def test_dryer_rule(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    clothes_dryer = env.scene.object_registry("name", "clothes_dryer")
    remover_dishtowel = env.scene.object_registry("name", "remover_dishtowel")
    bowl = env.scene.object_registry("name", "bowl")
    water = env.scene.get_system("water")

    place_obj_on_floor_plane(clothes_dryer)
    og.sim.step()

    # Place the two objects inside the dryer
    remover_dishtowel.set_position_orientation(
        position=[0.06, 0, 0.2], orientation=[0.0311883, -0.23199339, -0.06849886, 0.96980107]
    )
    bowl.set_position_orientation(position=[0.0, 0.0, 0.2], orientation=[0, 0, 0, 1])
    og.sim.step()

    assert remover_dishtowel.states[Saturated].set_value(water, True)
    assert bowl.states[Covered].set_value(water, True)
    og.sim.step()

    assert remover_dishtowel.states[Saturated].get_value(water)
    assert clothes_dryer.states[Contains].get_value(water)

    # The rule will not execute if Open is True
    clothes_dryer.states[Open].set_value(True)
    clothes_dryer.states[ToggledOn].set_value(True)
    og.sim.step()

    assert remover_dishtowel.states[Saturated].get_value(water)
    assert clothes_dryer.states[Contains].get_value(water)

    # The rule will not execute if ToggledOn is False
    clothes_dryer.states[Open].set_value(False)
    clothes_dryer.states[ToggledOn].set_value(False)
    og.sim.step()

    assert remover_dishtowel.states[Saturated].get_value(water)
    assert clothes_dryer.states[Contains].get_value(water)

    # The rule will execute if Open is False and ToggledOn is True
    clothes_dryer.states[Open].set_value(False)
    clothes_dryer.states[ToggledOn].set_value(True)
    og.sim.step()

    # Need to take one more step for the state setters to take effect
    og.sim.step()

    assert not remover_dishtowel.states[Saturated].get_value(water)
    # TODO: check this once we fix the dryer fillable volume
    # assert not clothes_dryer.states[Contains].get_value(water)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_washer_rule(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    washer = env.scene.object_registry("name", "washer")
    remover_dishtowel = env.scene.object_registry("name", "remover_dishtowel")
    bowl = env.scene.object_registry("name", "bowl")
    water = env.scene.get_system("water")
    dust = env.scene.get_system("dust")  # always remove
    salt = env.scene.get_system("salt")  # always remove (not explicitly specified)
    rust = env.scene.get_system("rust")  # never remove
    spray_paint = env.scene.get_system("spray_paint")  # requires acetone
    acetone = env.scene.get_system("acetone")  # solvent for spray paint
    cooking_oil = env.scene.get_system("cooking_oil")  # requires vinegar, lemon_juice, vinegar, etc.

    place_obj_on_floor_plane(washer)
    og.sim.step()

    # Place the two objects inside the washer
    # (Hacky) use baking_sheet as a stepping stone to elevate the objects so that they are inside the container volume.
    baking_sheet.set_position_orientation(
        position=[0.0, 0.0, 0.04], orientation=T.euler2quat(th.tensor([math.pi, 0, 0], dtype=th.float32))
    )
    remover_dishtowel.set_position_orientation(position=[0.0, 0.0, 0.05], orientation=[0, 0, 0, 1])
    bowl.set_position_orientation(position=[0.10, 0.0, 0.08], orientation=[0, 0, 0, 1])
    og.sim.step()

    assert bowl.states[Covered].set_value(dust, True)
    assert bowl.states[Covered].set_value(salt, True)
    assert bowl.states[Covered].set_value(rust, True)
    assert bowl.states[Covered].set_value(spray_paint, True)
    assert bowl.states[Covered].set_value(acetone, True)
    assert bowl.states[Covered].set_value(cooking_oil, True)
    assert not remover_dishtowel.states[Saturated].get_value(water)
    assert not bowl.states[Covered].get_value(water)

    # The rule will not execute if Open is True
    washer.states[Open].set_value(True)
    og.sim.step()

    assert bowl.states[Covered].get_value(dust)
    assert bowl.states[Covered].get_value(salt)
    assert bowl.states[Covered].get_value(rust)
    assert bowl.states[Covered].get_value(spray_paint)
    assert bowl.states[Covered].get_value(acetone)
    assert bowl.states[Covered].get_value(cooking_oil)
    assert not remover_dishtowel.states[Saturated].get_value(water)
    assert not bowl.states[Covered].get_value(water)

    washer.states[Open].set_value(False)
    washer.states[ToggledOn].set_value(True)

    # The rule will execute when Open is False and ToggledOn is True
    og.sim.step()

    # Need to take one more step for the state setters to take effect
    og.sim.step()

    assert not bowl.states[Covered].get_value(dust)
    assert not bowl.states[Covered].get_value(salt)
    assert bowl.states[Covered].get_value(rust)
    assert not bowl.states[Covered].get_value(spray_paint)
    assert not bowl.states[Covered].get_value(acetone)
    assert bowl.states[Covered].get_value(cooking_oil)
    assert remover_dishtowel.states[Saturated].get_value(water)
    assert bowl.states[Covered].get_value(water)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_slicing_rule(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    apple = env.scene.object_registry("name", "apple")
    table_knife = env.scene.object_registry("name", "table_knife")

    deleted_objs = [apple]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    assert apple.states[Cooked].set_value(True)

    initial_half_apples = env.scene.object_registry("category", "half_apple", set()).copy()

    place_obj_on_floor_plane(apple)
    og.sim.step()

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.15], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()
    assert not table_knife.states[Touching].get_value(apple)
    final_half_apples = env.scene.object_registry("category", "half_apple", set()).copy()
    assert len(final_half_apples) == len(initial_half_apples)
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is not None

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.10], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()
    final_half_apples = env.scene.object_registry("category", "half_apple", set()).copy()
    assert len(final_half_apples) > len(initial_half_apples)
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # One more step for the half apples to be initialized
    og.sim.step()

    # All new half_apple should be cooked
    new_half_apples = final_half_apples - initial_half_apples
    for half_apple in new_half_apples:
        assert half_apple.states[Cooked].get_value()

    # Clean up
    for apple in new_half_apples:
        env.scene.remove_object(apple)
    og.sim.step()

    objs = [DatasetObject(**obj_cfg) for obj_cfg in deleted_objs_cfg]
    og.sim.batch_add_objects(objs, scenes=[env.scene] * len(objs))
    og.sim.step()


@og_test
def test_dicing_rule_cooked(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    half_apple = env.scene.object_registry("name", "half_apple")
    table_knife = env.scene.object_registry("name", "table_knife")
    cooked_diced_apple = env.scene.get_system("cooked__diced__apple")

    deleted_objs = [half_apple]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    half_apple.set_orientation(T.euler2quat(th.tensor([0, -math.pi / 2, 0], dtype=th.float32)))
    place_obj_on_floor_plane(half_apple)
    og.sim.step()

    assert half_apple.states[Cooked].set_value(True)

    assert cooked_diced_apple.n_particles == 0

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.15], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    assert not table_knife.states[Touching].get_value(half_apple)
    assert cooked_diced_apple.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is not None

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.065], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    assert cooked_diced_apple.n_particles > 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Move the knife away so that it doesn't immediately dice the half_apple again once it's imported back
    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 1.15], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_dicing_rule_uncooked(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    half_apple = env.scene.object_registry("name", "half_apple")
    table_knife = env.scene.object_registry("name", "table_knife")
    diced_apple = env.scene.get_system("diced__apple")

    deleted_objs = [half_apple]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    half_apple.set_orientation(T.euler2quat(th.tensor([0, -math.pi / 2, 0], dtype=th.float32)))
    place_obj_on_floor_plane(half_apple)
    og.sim.step()

    assert diced_apple.n_particles == 0

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.15], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    assert not table_knife.states[Touching].get_value(half_apple)
    assert diced_apple.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is not None

    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 0.065], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    assert diced_apple.n_particles > 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Move the knife away so that it doesn't immediately dice the half_apple again once it's imported back
    table_knife.set_position_orientation(
        position=[-0.05, 0.0, 1.15], orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32))
    )
    og.sim.step()

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_melting_rule(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    swiss_cheese = env.scene.object_registry("name", "swiss_cheese")
    melted_swiss_cheese = env.scene.get_system("melted__swiss_cheese")

    deleted_objs = [swiss_cheese]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    swiss_cheese.set_position_orientation(position=stockpot.aabb_center, orientation=[0, 0, 0, 1])
    og.sim.step()
    assert swiss_cheese.states[Inside].get_value(stockpot)

    assert melted_swiss_cheese.n_particles == 0

    # To save time, directly set the temperature of the swiss cheese to be below the melting point
    assert swiss_cheese.states[Temperature].set_value(m.transition_rules.MELTING_TEMPERATURE - 1)
    og.sim.step()

    assert melted_swiss_cheese.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is not None

    # To save time, directly set the temperature of the swiss cheese to be above the melting point
    assert swiss_cheese.states[Temperature].set_value(m.transition_rules.MELTING_TEMPERATURE + 1)
    og.sim.step()

    # Recipe should execute successfully: new melted swiss cheese should be created, and the ingredients should be deleted
    assert melted_swiss_cheese.n_particles > 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_cooking_physical_particle_rule_failure_recipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    brown_rice = env.scene.get_system("brown_rice")
    water = env.scene.get_system("water")
    cooked_water = env.scene.get_system("cooked__water")
    cooked_brown_rice = env.scene.get_system("cooked__brown_rice")

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    brown_rice.generate_particles(positions=[(stockpot.aabb_center + th.tensor([0.03, 0.0, 0.0])).tolist()])
    # This fails the recipe because water (recipe system) is not in the stockpot
    water.generate_particles(positions=[[-0.25, 0.17, 1.95]])

    assert stockpot.states[Contains].get_value(brown_rice)
    assert not stockpot.states[Contains].get_value(water)

    assert cooked_brown_rice.n_particles == 0

    # To save time, directly set the stockpot to be heated
    assert stockpot.states[Heated].set_value(True)
    og.sim.step()

    # Recipe should fail: no cooked arborio rice should be created
    assert water.n_particles > 0
    assert cooked_water.n_particles == 0
    assert brown_rice.n_particles > 0
    assert cooked_brown_rice.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_physical_particle_rule_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    brown_rice = env.scene.get_system("brown_rice")
    water = env.scene.get_system("water")
    cooked_water = env.scene.get_system("cooked__water")
    cooked_brown_rice = env.scene.get_system("cooked__brown_rice")

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    brown_rice.generate_particles(positions=[(stockpot.aabb_center + th.tensor([0.03, 0.0, 0.0])).tolist()])
    water.generate_particles(positions=[(stockpot.aabb_center + th.tensor([-0.03, 0.0, 0.0])).tolist()])

    assert stockpot.states[Contains].get_value(brown_rice)
    assert stockpot.states[Contains].get_value(water)

    assert cooked_brown_rice.n_particles == 0
    assert cooked_water.n_particles == 0

    # To save time, directly set the stockpot to be heated
    assert stockpot.states[Heated].set_value(True)
    og.sim.step()

    assert water.n_particles == 0
    assert cooked_water.n_particles > 0
    assert brown_rice.n_particles > 0
    assert cooked_brown_rice.n_particles == 0

    # Recipe should execute successfully: new cooked arborio rice should be created, and the ingredients should be deleted
    og.sim.step()

    assert water.n_particles == 0
    assert cooked_water.n_particles == 0
    assert brown_rice.n_particles == 0
    assert cooked_brown_rice.n_particles > 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_mixing_rule_failure_recipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    bowl = env.scene.object_registry("name", "bowl")
    tablespoon = env.scene.object_registry("name", "tablespoon")
    water = env.scene.get_system("water")
    granulated_sugar = env.scene.get_system("granulated_sugar")
    lemon_juice = env.scene.get_system("lemon_juice")
    lemonade = env.scene.get_system("lemonade")
    sludge = env.scene.get_system("sludge")

    place_obj_on_floor_plane(bowl)
    og.sim.step()

    water.generate_particles(positions=[[-0.02, 0.0, 0.02]])
    granulated_sugar.generate_particles(positions=[[0.0, 0.0, 0.02]])
    # This fails the recipe because lemon juice (recipe system) is not in the bowl
    lemon_juice.generate_particles(positions=[[0.02, 0.0, 1.02]])

    assert bowl.states[Contains].get_value(water)
    assert bowl.states[Contains].get_value(granulated_sugar)
    assert not bowl.states[Contains].get_value(lemon_juice)

    assert lemonade.n_particles == 0
    assert sludge.n_particles == 0

    # Move the tablespoon to touch the bowl
    tablespoon.set_position_orientation(
        position=[0.10, 0.0, 0.01], orientation=T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0]))
    )
    tablespoon.keep_still()
    tablespoon.set_linear_velocity(th.tensor([-1.0, 0.0, 0.0]))
    for _ in range(3):
        og.sim.step()

    assert tablespoon.states[Touching].get_value(bowl)

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert lemonade.n_particles == 0
    assert sludge.n_particles > 0
    assert water.n_particles == 0
    assert granulated_sugar.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_mixing_rule_failure_nonrecipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    bowl = env.scene.object_registry("name", "bowl")
    tablespoon = env.scene.object_registry("name", "tablespoon")
    water = env.scene.get_system("water")
    granulated_sugar = env.scene.get_system("granulated_sugar")
    lemon_juice = env.scene.get_system("lemon_juice")
    lemonade = env.scene.get_system("lemonade")
    salt = env.scene.get_system("salt")
    sludge = env.scene.get_system("sludge")

    place_obj_on_floor_plane(bowl)
    og.sim.step()

    water.generate_particles(positions=[[-0.02, 0, 0.02]])
    granulated_sugar.generate_particles(positions=[[0.0, 0.0, 0.02]])
    lemon_juice.generate_particles(positions=[[0.02, 0.0, 0.02]])
    # This fails the recipe because salt (nonrecipe system) is in the bowl
    salt.generate_particles(positions=[[0.0, 0.02, 0.02]])

    assert bowl.states[Contains].get_value(water)
    assert bowl.states[Contains].get_value(granulated_sugar)
    assert bowl.states[Contains].get_value(lemon_juice)
    assert bowl.states[Contains].get_value(salt)

    assert lemonade.n_particles == 0
    assert sludge.n_particles == 0

    # Move the tablespoon to touch the bowl
    tablespoon.set_position_orientation(
        position=[0.10, 0.0, 0.01], orientation=T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0]))
    )
    tablespoon.keep_still()
    tablespoon.set_linear_velocity(th.tensor([-1.0, 0.0, 0.0]))
    for _ in range(3):
        og.sim.step()

    assert tablespoon.states[Touching].get_value(bowl)

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert lemonade.n_particles == 0
    assert sludge.n_particles > 0
    assert water.n_particles == 0
    assert granulated_sugar.n_particles == 0
    assert lemon_juice.n_particles == 0
    assert salt.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_mixing_rule_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    bowl = env.scene.object_registry("name", "bowl")
    tablespoon = env.scene.object_registry("name", "tablespoon")
    water = env.scene.get_system("water")
    granulated_sugar = env.scene.get_system("granulated_sugar")
    lemon_juice = env.scene.get_system("lemon_juice")
    lemonade = env.scene.get_system("lemonade")

    place_obj_on_floor_plane(bowl)
    og.sim.step()

    water.generate_particles(positions=[[-0.02, 0.0, 0.02]])
    granulated_sugar.generate_particles(positions=[[0.0, 0.0, 0.02]])
    lemon_juice.generate_particles(positions=[[0.02, 0.0, 0.02]])

    assert bowl.states[Contains].get_value(water)
    assert bowl.states[Contains].get_value(granulated_sugar)
    assert bowl.states[Contains].get_value(lemon_juice)

    assert lemonade.n_particles == 0

    # Move the tablespoon to touch the bowl
    tablespoon.set_position_orientation(
        position=[0.10, 0.0, 0.01], orientation=T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0]))
    )
    tablespoon.keep_still()
    tablespoon.set_linear_velocity(th.tensor([-1.0, 0.0, 0.0]))
    for _ in range(3):
        og.sim.step()

    assert tablespoon.states[Touching].get_value(bowl)

    # Recipe should execute successfully: new lemonade should be created, and the ingredients should be deleted
    assert lemonade.n_particles > 0
    assert water.n_particles == 0
    assert granulated_sugar.n_particles == 0
    assert lemon_juice.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_system_rule_failure_recipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    chicken = env.scene.object_registry("name", "chicken")
    chicken_broth = env.scene.get_system("chicken_broth")
    diced_carrot = env.scene.get_system("diced__carrot")
    diced_celery = env.scene.get_system("diced__celery")
    salt = env.scene.get_system("salt")
    rosemary = env.scene.get_system("rosemary")
    chicken_soup = env.scene.get_system("cooked__chicken_soup")

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    chicken.set_position_orientation(position=stockpot.aabb_center, orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    # This fails the recipe because chicken broth (recipe system) is not in the stockpot
    chicken_broth.generate_particles(positions=[[-0.1, -0.22, 1.93]])
    diced_carrot.generate_particles(positions=[[-0.1, -0.24, 0.93]])
    diced_celery.generate_particles(positions=[[-0.1, -0.26, 0.93]])
    salt.generate_particles(positions=[[-0.1, -0.28, 0.93]])
    rosemary.generate_particles(positions=[[-0.1, -0.3, 0.93]])
    og.sim.step()

    assert chicken.states[Inside].get_value(stockpot)
    assert not chicken.states[Cooked].get_value()
    assert not stockpot.states[Contains].get_value(chicken_broth)
    assert stockpot.states[Contains].get_value(diced_carrot)
    assert stockpot.states[Contains].get_value(diced_celery)
    assert stockpot.states[Contains].get_value(salt)
    assert stockpot.states[Contains].get_value(rosemary)

    assert chicken_soup.n_particles == 0

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no chicken soup should be created
    assert chicken_soup.n_particles == 0
    assert chicken_broth.n_particles > 0
    assert diced_carrot.n_particles > 0
    assert diced_celery.n_particles > 0
    assert salt.n_particles > 0
    assert rosemary.n_particles > 0
    assert env.scene.object_registry("name", "chicken") is not None

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_system_rule_failure_nonrecipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    chicken = env.scene.object_registry("name", "chicken")
    water = env.scene.get_system("water")
    chicken_broth = env.scene.get_system("chicken_broth")
    diced_carrot = env.scene.get_system("diced__carrot")
    diced_celery = env.scene.get_system("diced__celery")
    salt = env.scene.get_system("salt")
    rosemary = env.scene.get_system("rosemary")
    chicken_soup = env.scene.get_system("cooked__chicken_soup")

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    chicken.set_position_orientation(position=stockpot.aabb_center, orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    # This fails the recipe because water (nonrecipe system) is inside the stockpot
    water.generate_particles(positions=[[-0.1, -0.18, 0.93]])
    chicken_broth.generate_particles(positions=[[-0.1, -0.22, 0.93]])
    diced_carrot.generate_particles(positions=[[-0.1, -0.24, 0.93]])
    diced_celery.generate_particles(positions=[[-0.1, -0.26, 0.93]])
    salt.generate_particles(positions=[[-0.1, -0.28, 0.93]])
    rosemary.generate_particles(positions=[[-0.1, -0.3, 0.93]])
    og.sim.step()

    assert chicken.states[Inside].get_value(stockpot)
    assert not chicken.states[Cooked].get_value()
    assert stockpot.states[Contains].get_value(water)
    assert stockpot.states[Contains].get_value(chicken_broth)
    assert stockpot.states[Contains].get_value(diced_carrot)
    assert stockpot.states[Contains].get_value(diced_celery)
    assert stockpot.states[Contains].get_value(salt)
    assert stockpot.states[Contains].get_value(rosemary)

    assert chicken_soup.n_particles == 0

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no chicken soup should be created
    assert chicken_soup.n_particles == 0
    assert chicken_broth.n_particles > 0
    assert diced_carrot.n_particles > 0
    assert diced_celery.n_particles > 0
    assert salt.n_particles > 0
    assert rosemary.n_particles > 0
    assert water.n_particles > 0
    assert env.scene.object_registry("name", "chicken") is not None

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_system_rule_failure_nonrecipe_objects(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    chicken = env.scene.object_registry("name", "chicken")
    bowl = env.scene.object_registry("name", "bowl")
    chicken_broth = env.scene.get_system("chicken_broth")
    diced_carrot = env.scene.get_system("diced__carrot")
    diced_celery = env.scene.get_system("diced__celery")
    salt = env.scene.get_system("salt")
    rosemary = env.scene.get_system("rosemary")
    chicken_soup = env.scene.get_system("cooked__chicken_soup")

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    chicken.set_position_orientation(position=stockpot.aabb_center, orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    # This fails the recipe because the bowl (nonrecipe object) is inside the stockpot
    bowl.set_position_orientation(position=[-0.1, -0.15, 1], orientation=[0, 0, 0, 1])
    chicken_broth.generate_particles(positions=[[-0.1, -0.22, 0.93]])
    diced_carrot.generate_particles(positions=[[-0.1, -0.24, 0.93]])
    diced_celery.generate_particles(positions=[[-0.1, -0.26, 0.93]])
    salt.generate_particles(positions=[[-0.1, -0.28, 0.93]])
    rosemary.generate_particles(positions=[[-0.1, -0.3, 0.93]])
    og.sim.step()

    assert chicken.states[Inside].get_value(stockpot)
    assert bowl.states[Inside].get_value(stockpot)
    assert not chicken.states[Cooked].get_value()
    assert stockpot.states[Contains].get_value(chicken_broth)
    assert stockpot.states[Contains].get_value(diced_carrot)
    assert stockpot.states[Contains].get_value(diced_celery)
    assert stockpot.states[Contains].get_value(salt)
    assert stockpot.states[Contains].get_value(rosemary)

    assert chicken_soup.n_particles == 0

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no chicken soup should be created
    assert chicken_soup.n_particles == 0
    assert chicken_broth.n_particles > 0
    assert diced_carrot.n_particles > 0
    assert diced_celery.n_particles > 0
    assert salt.n_particles > 0
    assert rosemary.n_particles > 0
    assert env.scene.object_registry("name", "chicken") is not None
    assert env.scene.object_registry("name", "bowl") is not None

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_system_rule_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    stove = env.scene.object_registry("name", "stove")
    stockpot = env.scene.object_registry("name", "stockpot")
    chicken = env.scene.object_registry("name", "chicken")
    chicken_broth = env.scene.get_system("chicken_broth")
    diced_carrot = env.scene.get_system("diced__carrot")
    diced_celery = env.scene.get_system("diced__celery")
    salt = env.scene.get_system("salt")
    rosemary = env.scene.get_system("rosemary")
    chicken_soup = env.scene.get_system("cooked__chicken_soup")

    deleted_objs = [chicken]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(stove)
    og.sim.step()

    stockpot.set_position_orientation(position=[-0.1, -0.2, 0.89], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[OnTop].get_value(stove)

    chicken.set_position_orientation(stockpot.aabb_center, [0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    chicken_broth.generate_particles(positions=[[-0.1, -0.22, 0.93]])
    diced_carrot.generate_particles(positions=[[-0.1, -0.24, 0.93]])
    diced_celery.generate_particles(positions=[[-0.1, -0.26, 0.93]])
    salt.generate_particles(positions=[[-0.1, -0.28, 0.93]])
    rosemary.generate_particles(positions=[[-0.1, -0.3, 0.93]])
    og.sim.step()

    assert chicken.states[Inside].get_value(stockpot)
    assert not chicken.states[Cooked].get_value()
    assert stockpot.states[Contains].get_value(chicken_broth)
    assert stockpot.states[Contains].get_value(diced_carrot)
    assert stockpot.states[Contains].get_value(diced_celery)
    assert stockpot.states[Contains].get_value(salt)
    assert stockpot.states[Contains].get_value(rosemary)

    assert chicken_soup.n_particles == 0

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should execute successfully: new chicken soup should be created, and the ingredients should be deleted
    assert chicken_soup.n_particles > 0
    assert chicken_broth.n_particles == 0
    assert diced_carrot.n_particles == 0
    assert diced_celery.n_particles == 0
    assert salt.n_particles == 0
    assert rosemary.n_particles == 0

    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_cooking_object_rule_failure_wrong_container(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    stockpot = env.scene.object_registry("name", "stockpot")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    # This fails the recipe because it requires the baking sheet to be inside the oven, not the stockpot
    stockpot.set_position_orientation([0, 0, 0.487], [0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.464], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.506], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[Inside].get_value(stockpot)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_failure_recipe_objects(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.05, 0.455], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    # This fails the recipe because it requires the bagel dough to be on top of the baking sheet
    bagel_dough.set_position_orientation(position=[1, 0, 0.5], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[1.02, 0, 0.55], orientation=[0, 0, 0, 1])
    og.sim.step()
    assert not bagel_dough.states[OnTop].get_value(baking_sheet)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_failure_unary_states(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[0.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    # This fails the recipe because it requires the bagel dough and the raw egg to be not cooked
    assert bagel_dough.states[Cooked].set_value(True)
    assert raw_egg.states[Cooked].set_value(True)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_failure_binary_system_states(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[0.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    # This fails the recipe because it requires the bagel dough to be covered with sesame seed
    assert bagel_dough.states[Covered].set_value(sesame_seed, False)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_failure_binary_object_states(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[2.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    # This fails the recipe because it requires the raw egg to be on top of the bagel dough
    assert not raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_failure_wrong_heat_source(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    stove = env.scene.object_registry("name", "stove")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    # This fails the recipe because it requires the oven to be the heat source, not the stove
    place_obj_on_floor_plane(stove)
    og.sim.step()

    # Check that the stove heat source link exists
    stove.states[HeatSourceOrSink].link.get_position_orientation()[0]

    # Put the baking sheet on the stove
    baking_sheet.set_position_orientation(position=[-0.1, -0.15, 0.80], orientation=[0, 0, 0, 1])
    og.sim.step()

    bagel_dough.set_position_orientation(position=[-0.1, -0.15, 0.84], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[-0.1, -0.15, 0.89], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(True)
    assert raw_egg.states[Cooked].set_value(True)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Make sure the stove affects the baking sheet
    assert stove.states[HeatSourceOrSink].affects_obj(baking_sheet)

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_cooking_object_rule_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = env.scene.object_registry("name", "oven")
    baking_sheet = env.scene.object_registry("name", "baking_sheet")
    bagel_dough = env.scene.object_registry("name", "bagel_dough")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    sesame_seed = env.scene.get_system("sesame_seed")

    deleted_objs = [bagel_dough, raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[0.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(3):
        og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    # Recipe should execute successfully: new bagels should be created, and the ingredients should be deleted
    assert len(final_bagels) > len(initial_bagels)
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Need to step again for the new bagels to be initialized, placed in the container, and cooked.
    og.sim.step()

    # All new bagels should be cooked
    new_bagels = final_bagels - initial_bagels
    for bagel in new_bagels:
        assert bagel.states[Cooked].get_value()
        # This assertion occasionally fails, because when four bagels are sampled on top of the baking sheet one by one,
        # there is no guarantee that all four of them will be on top of the baking sheet at the end.
        assert (
            bagel.states[OnTop].get_value(baking_sheet)
            or bagel.states[Inside].get_value(oven)
            or bagel.states[Touching].get_value(baking_sheet)
            or bagel.states[Touching].get_value(oven)
        )

    # Clean up
    remove_all_systems(env.scene)

    for bagel in new_bagels:
        env.scene.remove_object(bagel)
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_system_failure_wrong_container(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    food_processor = env.scene.object_registry("name", "food_processor")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    # This fails the recipe because it requires the blender to be the container, not the food processor
    place_obj_on_floor_plane(food_processor)
    og.sim.step()

    milk.generate_particles(positions=th.tensor([[0.02, 0.06, 0.22]]))
    chocolate_sauce.generate_particles(positions=th.tensor([[-0.05, -0.04, 0.22]]))
    ice_cream.set_position_orientation(position=[0.03, -0.02, 0.23], orientation=[0, 0, 0, 1])

    og.sim.step()

    assert food_processor.states[Contains].get_value(milk)
    assert food_processor.states[Contains].get_value(chocolate_sauce)
    assert ice_cream.states[Inside].get_value(food_processor)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    food_processor.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert milkshake.n_particles == 0
    assert sludge.n_particles > 0
    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_system_failure_recipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = env.scene.object_registry("name", "blender")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    # This fails the recipe because it requires the milk to be in the blender
    milk.generate_particles(positions=th.tensor([[0.02, 0, 1.57]], dtype=th.float32))
    chocolate_sauce.generate_particles(positions=th.tensor([[0, -0.02, 0.57]], dtype=th.float32))
    ice_cream.set_position_orientation(position=[0, 0, 0.51], orientation=[0, 0, 0, 1])
    og.sim.step()

    assert not blender.states[Contains].get_value(milk)
    assert blender.states[Contains].get_value(chocolate_sauce)
    assert ice_cream.states[Inside].get_value(blender)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert milkshake.n_particles == 0
    assert sludge.n_particles > 0
    assert chocolate_sauce.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_system_failure_recipe_objects(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = env.scene.object_registry("name", "blender")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=th.tensor([[0.02, 0, 0.57]]))
    chocolate_sauce.generate_particles(positions=th.tensor([[0, -0.02, 0.57]]))
    # This fails the recipe because it requires the ice cream to be inside the blender
    ice_cream.set_position_orientation(position=[0, 0, 1.51], orientation=[0, 0, 0, 1])

    og.sim.step()

    assert blender.states[Contains].get_value(milk)
    assert blender.states[Contains].get_value(chocolate_sauce)
    assert not ice_cream.states[Inside].get_value(blender)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert milkshake.n_particles == 0
    assert sludge.n_particles > 0
    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)


@og_test
def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_systems(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = env.scene.object_registry("name", "blender")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")
    water = env.scene.get_system("water")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=th.tensor([[0.02, 0, 0.57]]))
    chocolate_sauce.generate_particles(positions=th.tensor([[0, -0.02, 0.57]]))
    # This fails the recipe because water (nonrecipe system) is in the blender
    water.generate_particles(positions=th.tensor([[0, 0, 0.57]]))
    ice_cream.set_position_orientation(position=[0, 0, 0.51], orientation=[0, 0, 0, 1])

    og.sim.step()

    assert blender.states[Contains].get_value(milk)
    assert blender.states[Contains].get_value(chocolate_sauce)
    assert blender.states[Contains].get_value(water)
    assert ice_cream.states[Inside].get_value(blender)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert milkshake.n_particles == 0
    assert sludge.n_particles > 0
    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0
    assert water.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)
    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_objects(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = env.scene.object_registry("name", "blender")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    bowl = env.scene.object_registry("name", "bowl")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")

    deleted_objs = [ice_cream, bowl]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=th.tensor([[0.02, 0, 0.57]]))
    chocolate_sauce.generate_particles(positions=th.tensor([[0, -0.02, 0.57]]))
    ice_cream.set_position_orientation(position=[0, 0, 0.51], orientation=[0, 0, 0, 1])
    # This fails the recipe because the bowl (nonrecipe object) is in the blender
    bowl.set_position_orientation(position=[0, 0, 0.58], orientation=[0, 0, 0, 1])

    og.sim.step()

    assert blender.states[Contains].get_value(milk)
    assert blender.states[Contains].get_value(chocolate_sauce)
    assert ice_cream.states[Inside].get_value(blender)
    assert bowl.states[Inside].get_value(blender)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no milkshake should be created, and sludge should be created.
    assert milkshake.n_particles == 0
    assert sludge.n_particles > 0
    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0

    # Clean up
    remove_all_systems(env.scene)
    # Spawning the bowl in the blender's location causes PhysX to crash,
    # resulting in "Invalid PhysX transform detected" errors. To avoid this, we temporarily move the blender aside.
    blender.set_position_orientation(
        position=blender.get_position_orientation()[0] + th.tensor([1.0, 0.0, 0.0], dtype=th.float32)
    )
    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_system_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = env.scene.object_registry("name", "blender")
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
    milk = env.scene.get_system("whole_milk")
    chocolate_sauce = env.scene.get_system("chocolate_sauce")
    milkshake = env.scene.get_system("milkshake")
    sludge = env.scene.get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=th.tensor([[0.02, 0, 0.57]]))
    chocolate_sauce.generate_particles(positions=th.tensor([[0, -0.02, 0.57]]))
    ice_cream.set_position_orientation(position=[0, 0, 0.51], orientation=[0, 0, 0, 1])

    og.sim.step()

    assert blender.states[Contains].get_value(milk)
    assert blender.states[Contains].get_value(chocolate_sauce)
    assert ice_cream.states[Inside].get_value(blender)

    assert milkshake.n_particles == 0
    assert sludge.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should execute successfully: new milkshake should be created, and the ingredients should be deleted
    assert milkshake.n_particles > 0
    assert sludge.n_particles == 0
    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_object_failure_unary_states(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    electric_mixer = env.scene.object_registry("name", "electric_mixer")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    another_raw_egg = env.scene.object_registry("name", "another_raw_egg")
    flour = env.scene.get_system("flour")
    granulated_sugar = env.scene.get_system("granulated_sugar")
    vanilla = env.scene.get_system("vanilla")
    melted_butter = env.scene.get_system("melted__butter")
    baking_powder = env.scene.get_system("baking_powder")
    salt = env.scene.get_system("salt")
    sludge = env.scene.get_system("sludge")

    initial_doughs = env.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    deleted_objs = [raw_egg, another_raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(electric_mixer)
    og.sim.step()

    another_raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.50], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.47], orientation=[0, 0, 0, 1])
    flour.generate_particles(positions=th.tensor([[-0.01, -0.15, 0.43]]))
    granulated_sugar.generate_particles(positions=th.tensor([[0.01, -0.15, 0.43]]))
    vanilla.generate_particles(positions=th.tensor([[0.03, -0.15, 0.43]]))
    melted_butter.generate_particles(positions=th.tensor([[-0.01, -0.13, 0.43]]))
    baking_powder.generate_particles(positions=th.tensor([[0.01, -0.13, 0.43]]))
    salt.generate_particles(positions=th.tensor([[0.03, -0.13, 0.43]]))
    # This fails the recipe because the egg should not be cooked
    raw_egg.states[Cooked].set_value(True)
    og.sim.step()

    assert electric_mixer.states[Contains].get_value(flour)
    assert electric_mixer.states[Contains].get_value(granulated_sugar)
    assert electric_mixer.states[Contains].get_value(vanilla)
    assert electric_mixer.states[Contains].get_value(melted_butter)
    assert electric_mixer.states[Contains].get_value(baking_powder)
    assert electric_mixer.states[Contains].get_value(salt)
    assert raw_egg.states[Inside].get_value(electric_mixer)
    assert raw_egg.states[Cooked].get_value()
    assert another_raw_egg.states[Inside].get_value(electric_mixer)
    assert not another_raw_egg.states[Cooked].get_value()

    assert sludge.n_particles == 0

    electric_mixer.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should fail: no dough should be created, and sludge should be created.
    final_doughs = env.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    # Recipe should execute successfully: new dough should be created, and the ingredients should be deleted
    assert len(final_doughs) == len(initial_doughs)
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None
    assert flour.n_particles == 0
    assert granulated_sugar.n_particles == 0
    assert vanilla.n_particles == 0
    assert melted_butter.n_particles == 0
    assert baking_powder.n_particles == 0
    assert salt.n_particles == 0
    assert sludge.n_particles > 0

    # Clean up
    remove_all_systems(env.scene)

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()


@og_test
def test_single_toggleable_machine_rule_output_object_success(env):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    electric_mixer = env.scene.object_registry("name", "electric_mixer")
    raw_egg = env.scene.object_registry("name", "raw_egg")
    another_raw_egg = env.scene.object_registry("name", "another_raw_egg")
    flour = env.scene.get_system("flour")
    granulated_sugar = env.scene.get_system("granulated_sugar")
    vanilla = env.scene.get_system("vanilla")
    melted_butter = env.scene.get_system("melted__butter")
    baking_powder = env.scene.get_system("baking_powder")
    salt = env.scene.get_system("salt")
    sludge = env.scene.get_system("sludge")

    initial_doughs = env.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    deleted_objs = [raw_egg, another_raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(electric_mixer)
    og.sim.step()

    another_raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.50], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.47], orientation=[0, 0, 0, 1])
    flour.generate_particles(positions=th.tensor([[-0.01, -0.15, 0.43]]))
    granulated_sugar.generate_particles(positions=th.tensor([[0.01, -0.15, 0.43]]))
    vanilla.generate_particles(positions=th.tensor([[0.03, -0.15, 0.43]]))
    melted_butter.generate_particles(positions=th.tensor([[-0.01, -0.13, 0.43]]))
    baking_powder.generate_particles(positions=th.tensor([[0.01, -0.13, 0.43]]))
    salt.generate_particles(positions=th.tensor([[0.03, -0.13, 0.43]]))

    og.sim.step()

    assert electric_mixer.states[Contains].get_value(flour)
    assert electric_mixer.states[Contains].get_value(granulated_sugar)
    assert electric_mixer.states[Contains].get_value(vanilla)
    assert electric_mixer.states[Contains].get_value(melted_butter)
    assert electric_mixer.states[Contains].get_value(baking_powder)
    assert electric_mixer.states[Contains].get_value(salt)
    assert raw_egg.states[Inside].get_value(electric_mixer)
    assert not raw_egg.states[Cooked].get_value()
    assert another_raw_egg.states[Inside].get_value(electric_mixer)
    assert not another_raw_egg.states[Cooked].get_value()

    assert sludge.n_particles == 0

    electric_mixer.states[ToggledOn].set_value(True)
    og.sim.step()

    # Recipe should execute successfully: new dough should be created, and the ingredients should be deleted
    final_doughs = env.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    # Recipe should execute successfully: new dough should be created, and the ingredients should be deleted
    assert len(final_doughs) > len(initial_doughs)
    for obj in deleted_objs:
        assert env.scene.object_registry("name", obj.name) is None
    assert flour.n_particles == 0
    assert granulated_sugar.n_particles == 0
    assert vanilla.n_particles == 0
    assert melted_butter.n_particles == 0
    assert baking_powder.n_particles == 0
    assert salt.n_particles == 0

    # Need to step again for the new dough to be initialized, placed in the container, and cooked.
    og.sim.step()

    # All new doughs should not be cooked
    new_doughs = final_doughs - initial_doughs
    for dough in new_doughs:
        assert not dough.states[Cooked].get_value()
        assert dough.states[OnTop].get_value(electric_mixer)

    # Clean up
    for dough in new_doughs:
        env.scene.remove_object(dough)
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        env.scene.add_object(obj)
    og.sim.step()
