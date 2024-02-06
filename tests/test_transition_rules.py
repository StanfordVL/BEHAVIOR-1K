from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.objects import DatasetObject
from omnigibson.transition_rules import REGISTERED_RULES
import omnigibson as og
from omnigibson.macros import macros as m

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane, retrieve_obj_cfg

import pytest
import numpy as np

@og_test
def test_cooking_object_rule_failure_wrong_container():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    stockpot = og.sim.scene.object_registry("name", "stockpot")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    # This fails the recipe because it requires the baking sheet to be inside the oven, not the stockpot
    stockpot.set_position_orientation([0, 0, 0.47], [0, 0, 0, 1])
    og.sim.step()
    assert stockpot.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.45], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.50], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(stockpot)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_recipe_objects():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation([0, 0, 0.455], [0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    # This fails the recipe because it requires the bagel dough to be on top of the baking sheet
    bagel_dough.set_position_orientation([1, 0, 0.495], [0, 0, 0, 1])
    raw_egg.set_position_orientation([1.02, 0, 0.54], [0, 0, 0, 1])
    og.sim.step()
    assert not bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_unary_states():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation([0, 0, 0.455], [0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.495], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.54], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    # This fails the recipe because it requires the bagel dough and the raw egg to be not cooked
    assert bagel_dough.states[Cooked].set_value(True)
    assert raw_egg.states[Cooked].set_value(True)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_binary_system_states():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation([0, 0, 0.455], [0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.495], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.54], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    # This fails the recipe because it requires the bagel dough and the raw egg to be covered with sesame seed
    assert bagel_dough.states[Covered].set_value(sesame_seed, False)
    assert raw_egg.states[Covered].set_value(sesame_seed, False)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_binary_object_states():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation([0, 0, 0.455], [0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.495], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.12, 0.15, 0.47], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    # This fails the recipe because it requires the raw egg to be on top of the bagel dough
    assert not raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_wrong_heat_source():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    stove = og.sim.scene.object_registry("name", "stove")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    # This fails the recipe because it requires the oven to be the heat source, not the stove
    place_obj_on_floor_plane(stove)
    og.sim.step()

    heat_source_position = stove.states[HeatSourceOrSink].link.get_position()
    baking_sheet.set_position_orientation([-0.20, 0, 0.80], [0, 0, 0, 1])
    og.sim.step()

    bagel_dough.set_position_orientation([-0.20, 0, 0.84], [0, 0, 0, 1])
    raw_egg.set_position_orientation([-0.18, 0, 0.845], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(True)
    assert raw_egg.states[Cooked].set_value(True)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert stove.states[ToggledOn].set_value(True)
    og.sim.step()

    # Make sure the stove affects the baking sheet
    assert stove.states[HeatSourceOrSink].affects_obj(baking_sheet)

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_success():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    deleted_objs = [bagel_dough, raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    initial_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    place_obj_on_floor_plane(oven)
    og.sim.step()

    baking_sheet.set_position_orientation([0, 0, 0.455], [0, 0, 0, 1])
    og.sim.step()
    assert baking_sheet.states[Inside].get_value(oven)

    bagel_dough.set_position_orientation([0, 0, 0.495], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0.02, 0, 0.54], [0, 0, 0, 1])
    og.sim.step()
    assert bagel_dough.states[OnTop].get_value(baking_sheet)
    assert raw_egg.states[OnTop].get_value(bagel_dough)

    assert bagel_dough.states[Cooked].set_value(False)
    assert raw_egg.states[Cooked].set_value(False)
    og.sim.step()

    assert bagel_dough.states[Covered].set_value(sesame_seed, True)
    assert raw_egg.states[Covered].set_value(sesame_seed, True)
    og.sim.step()

    assert oven.states[ToggledOn].set_value(True)
    og.sim.step()

    final_bagels = og.sim.scene.object_registry("category", "bagel", set()).copy()

    # Recipe should execute successfully: new bagels should be created, and the ingredients should be deleted
    assert len(final_bagels) > len(initial_bagels)
    for obj in deleted_objs:
        assert og.sim.scene.object_registry("name", obj.name) is None

    # Need to step again for the new bagels to be initialized, placed in the container, and cooked.
    og.sim.step()

    # All new bagels should be cooked
    new_bagels = final_bagels - initial_bagels
    for bagel in new_bagels:
        assert bagel.states[Cooked].get_value()
        # This assertion occasionally fails, because when four bagels are sampled on top of the baking sheet one by one,
        # there is no guarantee that all four of them will be on top of the baking sheet at the end.
        # assert bagel.states[OnTop].get_value(baking_sheet)
        assert bagel.states[Inside].get_value(oven)

    # Clean up
    sesame_seed.remove_all_particles()
    og.sim.step()

    og.sim.remove_objects(new_bagels)
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_failure_wrong_container():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    food_processor = og.sim.scene.object_registry("name", "food_processor")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    # This fails the recipe because it requires the blender to be the container, not the food processor
    place_obj_on_floor_plane(food_processor)
    og.sim.step()

    milk.generate_particles(positions=np.array([[0.02, 0, 0.25]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.25]]))
    ice_cream.set_position([0, 0, 0.2])

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
        assert og.sim.scene.object_registry("name", obj.name) is None

    # Clean up
    sludge.remove_all_particles()
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_failure_recipe_systems():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    # This fails the recipe because it requires the milk to be in the blender
    milk.generate_particles(positions=np.array([[0.02, 0, 1.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
    ice_cream.set_position([0, 0, 0.54])

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
        assert og.sim.scene.object_registry("name", obj.name) is None

    # Clean up
    sludge.remove_all_particles()
    milk.remove_all_particles()
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_failure_recipe_objects():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
    # This fails the recipe because it requires the ice cream to be inside the blender
    ice_cream.set_position([0, 0, 1.54])

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
    sludge.remove_all_particles()
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_systems():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")
    water = get_system("water")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
    # This fails the recipe because water (nonrecipe system) is in the blender
    water.generate_particles(positions=np.array([[0, 0, 0.5]]))
    ice_cream.set_position([0, 0, 0.54])

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
    sludge.remove_all_particles()
    og.sim.step()
    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_systems():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    bowl = og.sim.scene.object_registry("name", "bowl")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")

    deleted_objs = [ice_cream, bowl]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
    ice_cream.set_position([0, 0, 0.54])
    # This fails the recipe because the bowl (nonrecipe object) is in the blender
    bowl.set_position([0, 0, 0.6])

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
    sludge.remove_all_particles()
    og.sim.step()
    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_system_success():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    blender = og.sim.scene.object_registry("name", "blender")
    ice_cream = og.sim.scene.object_registry("name", "scoop_of_ice_cream")
    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    sludge = get_system("sludge")

    deleted_objs = [ice_cream]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
    ice_cream.set_position([0, 0, 0.54])

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
        assert og.sim.scene.object_registry("name", obj.name) is None

    # Clean up
    milkshake.remove_all_particles()
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_object_failure_unary_states():
    from IPython import embed; embed()
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    electric_mixer = og.sim.scene.object_registry("name", "electric_mixer")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    another_raw_egg = og.sim.scene.object_registry("name", "another_raw_egg")
    flour = get_system("flour")
    granulated_sugar = get_system("granulated_sugar")
    vanilla = get_system("vanilla")
    melted_butter = get_system("melted__butter")
    baking_powder = get_system("baking_powder")
    salt = get_system("salt")
    sludge = get_system("sludge")

    initial_doughs = og.sim.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    deleted_objs = [raw_egg, another_raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(electric_mixer)
    og.sim.step()

    another_raw_egg.set_position_orientation([0, 0.1, 0.2], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0, 0.1, 0.17], [0, 0, 0, 1])
    flour.generate_particles(positions=np.array([[-0.02, 0.06, 0.15]]))
    granulated_sugar.generate_particles(positions=np.array([[0.0, 0.06, 0.15]]))
    vanilla.generate_particles(positions=np.array([[0.02, 0.06, 0.15]]))
    melted_butter.generate_particles(positions=np.array([[-0.02, 0.08, 0.15]]))
    baking_powder.generate_particles(positions=np.array([[0.0, 0.08, 0.15]]))
    salt.generate_particles(positions=np.array([[0.02, 0.08, 0.15]]))
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
    final_doughs = og.sim.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    # Recipe should execute successfully: new dough should be created, and the ingredients should be deleted
    assert len(final_doughs) == len(initial_doughs)
    for obj in deleted_objs:
        assert og.sim.scene.object_registry("name", obj.name) is None
    assert flour.n_particles == 0
    assert granulated_sugar.n_particles == 0
    assert vanilla.n_particles == 0
    assert melted_butter.n_particles == 0
    assert baking_powder.n_particles == 0
    assert salt.n_particles == 0
    assert sludge.n_particles > 0

    # Clean up
    sludge.remove_all_particles()
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

@og_test
def test_single_toggleable_machine_rule_output_object_success():
    from IPython import embed; embed()
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    electric_mixer = og.sim.scene.object_registry("name", "electric_mixer")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    another_raw_egg = og.sim.scene.object_registry("name", "another_raw_egg")
    flour = get_system("flour")
    granulated_sugar = get_system("granulated_sugar")
    vanilla = get_system("vanilla")
    melted_butter = get_system("melted__butter")
    baking_powder = get_system("baking_powder")
    salt = get_system("salt")
    sludge = get_system("sludge")

    initial_doughs = og.sim.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    deleted_objs = [raw_egg, another_raw_egg]
    deleted_objs_cfg = [retrieve_obj_cfg(obj) for obj in deleted_objs]

    place_obj_on_floor_plane(electric_mixer)
    og.sim.step()

    another_raw_egg.set_position_orientation([0, 0.1, 0.2], [0, 0, 0, 1])
    raw_egg.set_position_orientation([0, 0.1, 0.17], [0, 0, 0, 1])
    flour.generate_particles(positions=np.array([[-0.02, 0.06, 0.15]]))
    granulated_sugar.generate_particles(positions=np.array([[0.0, 0.06, 0.15]]))
    vanilla.generate_particles(positions=np.array([[0.02, 0.06, 0.15]]))
    melted_butter.generate_particles(positions=np.array([[-0.02, 0.08, 0.15]]))
    baking_powder.generate_particles(positions=np.array([[0.0, 0.08, 0.15]]))
    salt.generate_particles(positions=np.array([[0.02, 0.08, 0.15]]))

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
    final_doughs = og.sim.scene.object_registry("category", "sugar_cookie_dough", set()).copy()

    # Recipe should execute successfully: new dough should be created, and the ingredients should be deleted
    assert len(final_doughs) > len(initial_doughs)
    for obj in deleted_objs:
        assert og.sim.scene.object_registry("name", obj.name) is None
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
    og.sim.remove_objects(new_doughs)
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()


test_single_toggleable_machine_rule_output_object_failure_unary_states()
