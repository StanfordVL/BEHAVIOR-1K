import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states.attachment import AttachmentType
from omnigibson.object_states import Attached

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def setup_scene_for_abilities(abilities1, abilities2):
    # Make sure simulation is stopped
    og.sim.stop()

    cfg = yaml.load(open(f"{og.example_config_path}/default_cfg.yaml", "r"), Loader=yaml.FullLoader)

    # Add objects that we want to create
    light_cfg = dict(
        type="LightObject",
        name="light",
        light_type="Sphere",
        radius=0.01,
        intensity=5000,
        position=[0, 0, 1.0],
    )

    apple_cfg = dict(
        type="DatasetObject",
        name="apple",
        category="apple",
        model="00_0",
        abilities=abilities1,
        position=[0, 0, 0.04],
    )

    fridge_cfg = dict(
        type="DatasetObject",
        name="fridge",
        category="fridge",
        model="12252",
        abilities=abilities2,
        position=[2, 0, 0.8],
    )

    cfg["objects"] = [light_cfg, apple_cfg, fridge_cfg]

    # Recreate the environment (this will automatically override the old environment instance)
    # We load the default config, which is simply an Scene with no objects loaded in by default
    env = og.Environment(configs=cfg)

    # Grab apple and fridge
    apple = env.scene.object_registry("name", "apple")
    fridge = env.scene.object_registry("name", "fridge")

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.972333, -2.0899  ,  1.0654  ]),
        orientation=np.array([ 0.60682517, -0.24656188, -0.28443909,  0.70004632]),
    )

    # Take a few steps
    for _ in range(5):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    return env, apple, fridge


def demo_sticky_attachment():
    ######################################################################################
    # Sticky attachment
    #   can attach if touching and at least one object has the sticky attachment type.
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.STICKY}},
        abilities2={})
    assert Attached in obj1.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        og.sim.step()

    assert obj1.states[Attached].get_value(obj2)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[Attached].set_value(obj2, False)
    assert not obj1.states[Attached].get_value(obj2)

    # Obj1 moves away from obj2.
    # obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_symmetric_attachment():
    ######################################################################################
    # Symmetric attachment
    #   can attach if touching and both objects have the symmetric attachment type and the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.SYMMETRIC, "attachment_category": "magnet"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.SYMMETRIC, "attachment_category": "magnet"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[Attached].get_value(obj2)
    assert obj2.states[Attached].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[Attached].set_value(obj2, False)
    assert not obj1.states[Attached].get_value(obj2)
    assert not obj2.states[Attached].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_symmetric_attachment_missing_symmetric():
    ######################################################################################
    # Symmetric attachment - FAIL because only 1 object has the symmetric attachment type
    #   can attach if touching and both objects have the symmetric attachment type and the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.SYMMETRIC, "attachment_category": "magnet"}},
        abilities2={})
    assert Attached in obj1.states
    assert Attached not in obj2.states

    # Obj1 moves towards obj2 but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[Attached].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_symmetric_attachment_diff_categories():
    ######################################################################################
    # Symmetric attachment - FAIL because the two objects have different attachment category
    #   can attach if touching and both objects have the symmetric attachment type and the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.SYMMETRIC, "attachment_category": "magnet"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.SYMMETRIC, "attachment_category": "velcro"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[Attached].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_male_female_attachment():
    ######################################################################################
    # Male / female attachment
    #   can attach if touching, both have the opposite end (male / female) but the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.MALE, "attachment_category": "usb"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.FEMALE, "attachment_category": "usb"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[Attached].get_value(obj2)
    assert obj2.states[Attached].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[Attached].set_value(obj2, False)
    assert not obj1.states[Attached].get_value(obj2)
    assert not obj2.states[Attached].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_male_female_attachment_missing_opposite_end():
    ######################################################################################
    # Male / female attachment - FAIL because both objects are male.
    #   can attach if touching, both have the opposite end (male / female) but the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.MALE, "attachment_category": "usb"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.MALE, "attachment_category": "usb"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[Attached].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_failed_male_female_attachment_diff_categories():
    ######################################################################################
    # Male / female attachment - FAIL because the two objects have different attachment category
    #   can attach if touching, both have the opposite end (male / female) but the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.MALE, "attachment_category": "usb"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.FEMALE, "attachment_category": "hdmi"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 and but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[Attached].get_value(obj2)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)


def demo_male_female_attachment_dump_load():
    ######################################################################################
    # Male / female attachment with dump_state and load_state
    #   can attach if touching, both have the opposite end (male / female) but the same attachment category
    ######################################################################################
    env, obj1, obj2 = setup_scene_for_abilities(
        abilities1={"attachable": {"attachment_type": AttachmentType.MALE, "attachment_category": "usb"}},
        abilities2={"attachable": {"attachment_type": AttachmentType.FEMALE, "attachment_category": "usb"}})
    assert Attached in obj1.states
    assert Attached in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[Attached].get_value(obj2)
    assert obj2.states[Attached].get_value(obj1)

    # Save the state.
    state = og.sim.dump_state()

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[Attached].set_value(obj2, False)
    assert not obj1.states[Attached].get_value(obj2)
    assert not obj2.states[Attached].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Load the state where the two objects are attached.
    og.sim.load_state(state)

    # Attached state should be restored correctly
    assert obj1.states[Attached].get_value(obj2)
    assert obj2.states[Attached].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

demo_names_to_demos = {
    "demo_sticky_attachment": demo_sticky_attachment,
    "demo_symmetric_attachment": demo_symmetric_attachment,
    "demo_failed_symmetric_attachment_missing_symmetric": demo_failed_symmetric_attachment_missing_symmetric,
    "demo_failed_symmetric_attachment_diff_categories": demo_failed_symmetric_attachment_diff_categories,
    "demo_male_female_attachment": demo_male_female_attachment,
    "demo_failed_male_female_attachment_missing_opposite_end": demo_failed_male_female_attachment_missing_opposite_end,
    "demo_failed_male_female_attachment_diff_categories": demo_failed_male_female_attachment_diff_categories,
    "demo_male_female_attachment_dump_load": demo_male_female_attachment_dump_load,
}


def main(random_selection=False, headless=False, short_exec=False):
    # Loop indefinitely and choose different examples to run
    for i in range(len(demo_names_to_demos)):
        demo_name = choose_from_options(options=list(demo_names_to_demos.keys()), name="attachment demo")
        # Run the demo
        demo_names_to_demos[demo_name]()


if __name__ == "__main__":
    main()
