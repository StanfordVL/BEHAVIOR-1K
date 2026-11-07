import pytest

import torch as th
from utils import place_obj_on_floor_plane, remove_all_systems, retrieve_obj_cfg

import omnigibson as og
from omnigibson.object_states import (
    Contains,
    Cooked,
    Covered,
    HeatSourceOrSink,
    Inside,
    OnTop,
    ToggledOn,
    Touching,
)
from omnigibson.objects import DatasetObject
from omnigibson.transition_rules import REGISTERED_RULES


def test_cooking_system_rule_failure_recipe_systems(env, stove, stockpot, chicken):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_system_rule_failure_nonrecipe_systems(env, stove, stockpot, chicken):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_system_rule_failure_nonrecipe_objects(env, stove, stockpot, chicken, bowl):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_system_rule_success(env, stove, stockpot, chicken):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_object_rule_failure_wrong_container(env, oven, stockpot, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    # This fails the recipe because it requires the baking sheet to be inside the oven, not the stockpot
    stockpot.set_position_orientation([0, 0, 0.487], [0, 0, 0, 1])
    for _ in range(10):
        og.sim.step()
    assert stockpot.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.464], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.506], [0, 0, 0, 1])
    for _ in range(10):
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


def test_cooking_object_rule_failure_recipe_objects(env, oven, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_object_rule_failure_unary_states(env, oven, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(5):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[0.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(10):
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


def test_cooking_object_rule_failure_binary_system_states(env, oven, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    sesame_seed = env.scene.get_system("sesame_seed")

    initial_bagels = env.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation(position=[0.0, 0.07, 0.42], orientation=[0, 0, 0, 1])
    for _ in range(5):
        og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation(position=[0.0, 0.07, 0.45], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[0.0, 0.07, 0.48], orientation=[0, 0, 0, 1])
    for _ in range(10):
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


def test_cooking_object_rule_failure_binary_object_states(env, oven, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_cooking_object_rule_failure_wrong_heat_source(env, stove, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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
    for _ in range(10):
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


@pytest.mark.skip(reason="To be fixed as part of #2017.")
def test_cooking_object_rule_success(env, oven, baking_sheet, bagel_dough, raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


@pytest.mark.skip(reason="To be fixed as part of #2017.")
def test_single_toggleable_machine_rule_output_system_failure_wrong_container(env, food_processor, scoop_of_ice_cream):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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
