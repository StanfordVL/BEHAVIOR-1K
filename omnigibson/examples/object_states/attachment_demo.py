import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject, LightObject
from omnigibson.utils.ui_utils import choose_from_options


def setup_scene_for_abilities(abilities1, abilities2):
    # Make sure simulation is stopped
    og.sim.stop()

    # Recreate the environment (this will automatically override the old environment instance)
    # We load the default config, which is simply an EmptyScene with no objects loaded in by default
    env = og.Environment(configs=f"{og.example_config_path}/default_cfg.yaml") #, physics_timestep=1/120., action_timestep=1/60.)

    objs = [None, None]
    abilities_arr = [abilities1, abilities2]
    position_arr = [np.array([0, 0, 0.04]), np.array([2, 0, 0.8])]

    # Add light
    light = LightObject(
        prim_path="/World/light",
        name="light",
        light_type="Sphere",
        radius=0.01,
        intensity=5000,
    )
    og.sim.import_object(light)
    light.set_position(np.array([0, 0, 1.0]))

    for idx, (obj_category, obj_model) in enumerate((("apple", "00_0"), ("fridge", "12252"))):
        name = obj_category
        objs[idx] = DatasetObject(
            prim_path=f"/World/{name}",
            category=obj_category,
            model=obj_model,
            name=f"{name}",
            abilities=abilities_arr[idx],
        )
        og.sim.import_object(objs[idx])
        objs[idx].set_position_orientation(position=position_arr[idx])

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.972333, -2.0899  ,  1.0654  ]),
        orientation=np.array([ 0.60682517, -0.24656188, -0.28443909,  0.70004632]),
    )

    # Take a few steps
    for _ in range(5):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    return env, objs[0], objs[1]


def demo_sticky_attachment():
    ######################################################################################
    # StickyAttachment
    #   can attach if touching and at least one object has sticky state.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"sticky": {}}, abilities2={})
    assert object_states.StickyAttachment in obj1.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        og.sim.step()
    assert obj1.states[object_states.StickyAttachment].get_value(obj2)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[object_states.StickyAttachment].set_value(obj2, False)
    assert not obj1.states[object_states.StickyAttachment].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_magnetic_attachment():
    ######################################################################################
    # MagneticAttachment
    #   can attach if touching and both objects have magnetic state.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"magnetic": {}}, abilities2={"magnetic": {}})
    assert object_states.MagneticAttachment in obj1.states
    assert object_states.MagneticAttachment in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[object_states.MagneticAttachment].get_value(obj2)
    assert obj2.states[object_states.MagneticAttachment].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[object_states.MagneticAttachment].set_value(obj2, False)
    assert not obj1.states[object_states.MagneticAttachment].get_value(obj2)
    assert not obj2.states[object_states.MagneticAttachment].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_magnetic_attachment():
    ######################################################################################
    # MagneticAttachment - FAIL because only 1 object has magnetic state
    #   can attach if touching and both objects have magnetic state.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"magnetic": {}}, abilities2={})
    assert object_states.MagneticAttachment in obj1.states
    assert object_states.MagneticAttachment not in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[object_states.MagneticAttachment].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_male_female_attachment():
    ######################################################################################
    # MaleAttachment / FemaleAttachment
    #   can attach if touching, self is male and the other is female.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"maleAttachable": {}}, abilities2={"femaleAttachable": {}})
    assert object_states.MaleAttachment in obj1.states
    assert object_states.FemaleAttachment in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[object_states.MaleAttachment].get_value(obj2)
    assert obj2.states[object_states.FemaleAttachment].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[object_states.MaleAttachment].set_value(obj2, False)
    assert not obj1.states[object_states.MaleAttachment].get_value(obj2)
    assert not obj2.states[object_states.FemaleAttachment].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_male_female_attachment():
    ######################################################################################
    # MaleAttachment - FAIL because the other object is not female
    #   can attach if touching, self is male and the other is female.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"maleAttachable": {}}, abilities2={"maleAttachable": {}})
    assert object_states.MaleAttachment in obj1.states
    assert object_states.FemaleAttachment not in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[object_states.MaleAttachment].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_hung_male_female_attachment():
    ######################################################################################
    # HungMaleAttachment / HungFemaleAttachment
    #   can attach if touching, self is male, the other is female,
    #   and the male hanging object is "below" the female mounting object (center of bbox).
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungFemaleAttachable": {}})
    assert object_states.HungMaleAttachment in obj1.states
    assert object_states.HungFemaleAttachment in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[object_states.HungMaleAttachment].get_value(obj2)
    assert obj2.states[object_states.HungFemaleAttachment].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[object_states.HungMaleAttachment].set_value(obj2, False)
    assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)
    assert not obj2.states[object_states.HungFemaleAttachment].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_hung_male_female_incompatibility_attachment():
    ######################################################################################
    # HungMaleAttachment - FAIL because the other object is not female hung
    #   can attach if touching, self is male, the other is female,
    #   and the male hanging object is "below" the female mounting object (center of bbox).
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungMaleAttachable": {}})
    assert object_states.HungMaleAttachment in obj1.states
    assert object_states.HungFemaleAttachment not in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_hung_male_female_mislocation_attachment():
    ######################################################################################
    # HungMaleAttachment / FemaleAttachment - FAIL because the male object is above the female object
    #   can attach if touching, self is male, the other is female,
    #   and the male hanging object is "below" the female mounting object (center of bbox).
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(abilities1={"hungMaleAttachable": {}}, abilities2={"hungFemaleAttachable": {}})
    assert object_states.HungMaleAttachment in obj1.states
    assert object_states.HungFemaleAttachment in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([5.0, 0, 5.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[object_states.HungMaleAttachment].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(200):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


demo_names_to_demos = {
    "demo_sticky_attachment": demo_sticky_attachment,
    "demo_magnetic_attachment": demo_magnetic_attachment,
    "demo_failed_magnetic_attachment" : demo_failed_magnetic_attachment,
    "demo_male_female_attachment": demo_male_female_attachment,
    "demo_failed_male_female_attachment": demo_failed_male_female_attachment,
    "demo_hung_male_female_attachment": demo_hung_male_female_attachment,
    "demo_failed_hung_male_female_incompatibility_attachment": demo_failed_hung_male_female_incompatibility_attachment,
    "demo_failed_hung_male_female_mislocation_attachment": demo_failed_hung_male_female_mislocation_attachment,
}


def main(random_selection=False, headless=False, short_exec=False):
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Loop indefinitely and choose different examples to run
    for i in range(len(demo_names_to_demos)):
        demo_name = choose_from_options(options=list(demo_names_to_demos.keys()), name="attachment demo")
        # Run the demo
        demo_names_to_demos[demo_name]()


if __name__ == "__main__":
    main()
