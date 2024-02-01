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
def test_cooking_object_rule_failure_unary_states():
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    baking_sheet = og.sim.scene.object_registry("name", "baking_sheet")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()
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

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()
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

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()
    assert len(final_bagels) == len(initial_bagels)

    # Clean up
    sesame_seed.remove_all_particles()

@og_test
def test_cooking_object_rule_failure_wrong_container():
    # from IPython import embed; print("debug"); embed()
    assert len(REGISTERED_RULES) > 0, "No rules registered!"

    oven = og.sim.scene.object_registry("name", "oven")
    stockpot = og.sim.scene.object_registry("name", "stockpot")
    bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
    raw_egg = og.sim.scene.object_registry("name", "raw_egg")
    sesame_seed = get_system("sesame_seed")

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()
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

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()
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

    initial_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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

    final_bagels = og.sim.scene.object_registry("category", "bagel").copy()

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
        assert bagel.states[OnTop].get_value(baking_sheet)

    # Clean up
    sesame_seed.remove_all_particles()
    og.sim.step()

    og.sim.remove_objects(new_bagels)
    og.sim.step()

    for obj_cfg in deleted_objs_cfg:
        obj = DatasetObject(**obj_cfg)
        og.sim.import_object(obj)
    og.sim.step()

# @og_test
# def test_slicing_rule():
#     assert len(REGISTERED_RULES) > 0, "No rules registered!"

# @og_test
# def test_blender_rule():
#     assert len(REGISTERED_RULES) > 0, "No rules registered!"
#     blender = og.sim.scene.object_registry("name", "blender")
#
#     blender.set_orientation([0, 0, 0, 1])
#     place_obj_on_floor_plane(blender)
#     og.sim.step()
#
#     milk = get_system("whole_milk")
#     chocolate_sauce = get_system("chocolate_sauce")
#     milkshake = get_system("milkshake")
#     milk.generate_particles(positions=np.array([[0.02, 0, 0.5]]))
#     chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.5]]))
#
#     ice_cream = DatasetObject(
#         name="ice_cream",
#         category="scoop_of_ice_cream",
#         model="dodndj",
#         bounding_box=[0.076, 0.077, 0.065],
#     )
#     og.sim.import_object(ice_cream)
#     ice_cream.set_position([0, 0, 0.54])
#
#     for i in range(5):
#         og.sim.step()
#
#     assert milkshake.n_particles == 0
#
#     blender.states[ToggledOn].set_value(True)
#     og.sim.step()
#
#     assert milk.n_particles == 0
#     assert chocolate_sauce.n_particles == 0
#     assert milkshake.n_particles > 0
#
#     # Remove objects and systems from recipe output
#     milkshake.remove_all_particles()
#     if og.sim.scene.object_registry("name", "ice_cream") is not None:
#         og.sim.remove_object(obj=ice_cream)
#
#
# @og_test
# def test_cooking_rule():
#     assert len(REGISTERED_RULES) > 0, "No rules registered!"
#     oven = og.sim.scene.object_registry("name", "oven")
#     oven.keep_still()
#     oven.set_orientation([0, 0, -0.707, 0.707])
#     place_obj_on_floor_plane(oven)
#     og.sim.step()
#
#     sheet = DatasetObject(
#         name="sheet",
#         category="baking_sheet",
#         model="yhurut",
#         bounding_box=[0.520, 0.312, 0.0395],
#     )
#
#     og.sim.import_object(sheet)
#     sheet.set_position_orientation([0.072, 0.004, 0.455], [0, 0, 0, 1])
#
#     dough = DatasetObject(
#         name="dough",
#         category="sugar_cookie_dough",
#         model="qewbbb",
#         bounding_box=[0.200, 0.192, 0.0957],
#     )
#     og.sim.import_object(dough)
#     dough.set_position_orientation([0.072, 0.004, 0.555], [0, 0, 0, 1])
#
#     for i in range(10):
#         og.sim.step()
#
#     assert len(og.sim.scene.object_registry("category", "sugar_cookie", default_val=[])) == 0
#
#     oven.states[ToggledOn].set_value(True)
#     og.sim.step()
#     og.sim.step()
#
#     dough_exists = og.sim.scene.object_registry("name", "dough") is not None
#     assert not dough_exists or not dough.states[OnTop].get_value(sheet)
#     assert len(og.sim.scene.object_registry("category", "sugar_cookie")) > 0
#
#     # Remove objects
#     if dough_exists:
#         og.sim.remove_object(dough)
#     og.sim.remove_object(sheet)

# test_cooking_object_rule_failure_wrong_container()
# test_cooking_object_rule_failure_unary_states()