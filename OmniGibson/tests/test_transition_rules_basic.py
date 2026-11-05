import math


import torch as th
from utils import place_obj_on_floor_plane, remove_all_systems, retrieve_obj_cfg

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import macros as m
from omnigibson.object_states import (
    Contains,
    Cooked,
    Covered,
    Heated,
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


def test_dryer_rule(env, clothes_dryer, remover_dishtowel, bowl):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_washer_rule(env, baking_sheet, washer, remover_dishtowel, bowl):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_slicing_rule(env, apple, table_knife):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

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


def test_dicing_rule_cooked(env, half_apple, table_knife):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_dicing_rule_uncooked(env, half_apple, table_knife):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_melting_rule(env, stove, stockpot, swiss_cheese):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_physical_particle_rule_failure_recipe_systems(env, stove, stockpot):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_physical_particle_rule_success(env, stove, stockpot):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_mixing_rule_failure_recipe_systems(env, bowl, tablespoon):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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
    og.sim.step()
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


def test_mixing_rule_failure_nonrecipe_systems(env, bowl, tablespoon):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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
    og.sim.step()
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


def test_mixing_rule_success(env, bowl, tablespoon):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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
    og.sim.step()
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
