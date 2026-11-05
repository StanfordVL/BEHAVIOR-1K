import torch as th
from utils import place_obj_on_floor_plane, remove_all_systems, retrieve_obj_cfg

import omnigibson as og
from omnigibson.object_states import (
    Contains,
    Cooked,
    Inside,
    OnTop,
    ToggledOn,
)
from omnigibson.objects import DatasetObject
from omnigibson.transition_rules import REGISTERED_RULES


def test_single_toggleable_machine_rule_output_system_failure_recipe_systems(env, blender, scoop_of_ice_cream):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_single_toggleable_machine_rule_output_system_failure_recipe_objects(env, blender, scoop_of_ice_cream):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_systems(env, blender, scoop_of_ice_cream):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_single_toggleable_machine_rule_output_system_failure_nonrecipe_objects(env, blender, scoop_of_ice_cream, bowl):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
    ice_cream = env.scene.object_registry("name", "scoop_of_ice_cream")
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


def test_single_toggleable_machine_rule_output_system_success(env, blender, scoop_of_ice_cream):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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


def test_single_toggleable_machine_rule_output_object_failure_unary_states(
    env, electric_mixer, raw_egg, another_raw_egg
):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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

    another_raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.47], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.45], orientation=[0, 0, 0, 1])
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


def test_single_toggleable_machine_rule_output_object_success(env, electric_mixer, raw_egg, another_raw_egg):
    assert len(REGISTERED_RULES) > 0, "No rules registered!"
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

    another_raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.47], orientation=[0, 0, 0, 1])
    raw_egg.set_position_orientation(position=[-0.01, -0.14, 0.45], orientation=[0, 0, 0, 1])
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
