import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states import AttachedTo

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def setup_scene():
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

    shelf_back_panel_cfg = dict(
        type="DatasetObject",
        name="shelf_back_panel",
        category="shelf_back_panel",
        model="gjsnrt",
        position=[0, 0, 0.01],
        abilities={"attachable": {}},
    )

    shelf_shelf_0_cfg = dict(
        type="DatasetObject",
        name="shelf_shelf_0",
        category="shelf_shelf",
        model="ymtnqa",
        position=[0, 0, 0.2],
        orientation=[ 0, 0, 0.3826834, 0.9238795 ],
        abilities={"attachable": {}},
    )

    shelf_shelf_1_cfg = dict(
        type="DatasetObject",
        name="shelf_shelf_1",
        category="shelf_shelf",
        model="ymtnqa",
        position=[30, 30, 30],
        abilities={"attachable": {}},
    )

    cfg["objects"] = [light_cfg, shelf_back_panel_cfg, shelf_shelf_0_cfg, shelf_shelf_1_cfg]

    # Recreate the environment (this will automatically override the old environment instance)
    # We load the default config, which is simply an Scene with no objects loaded in by default
    env = og.Environment(configs=cfg)

    # Grab apple and fridge
    shelf_back_panel = env.scene.object_registry("name", "shelf_back_panel")
    shelf_shelf_0 = env.scene.object_registry("name", "shelf_shelf_0")
    shelf_shelf_1 = env.scene.object_registry("name", "shelf_shelf_1")

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.972333, -2.0899, 1.0654]),
        orientation=np.array([0.60682517, -0.24656188, -0.28443909, 0.70004632]),
    )

    return env, shelf_back_panel, shelf_shelf_0, shelf_shelf_1


def demo_male_female_attachment():
    ######################################################################################
    # Male / female attachment
    #   can attach if touching, both have the opposite end (male / female) but the same attachment category
    ######################################################################################
    env, shelf_back_panel, shelf_shelf_0, shelf_shelf_1 = setup_scene()
    assert AttachedTo in shelf_back_panel.states
    assert AttachedTo in shelf_shelf_0.states

    from IPython import embed; print("debug"); embed()
    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[AttachedTo].get_value(obj2)
    assert obj2.states[AttachedTo].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[AttachedTo].set_value(obj2, False)
    assert not obj1.states[AttachedTo].get_value(obj2)
    assert not obj2.states[AttachedTo].get_value(obj1)

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
    assert AttachedTo in obj1.states
    assert AttachedTo in obj2.states

    # Obj1 moves towards obj2 but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[AttachedTo].get_value(obj2)

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
    assert AttachedTo in obj1.states
    assert AttachedTo in obj2.states

    # Obj1 moves towards obj2 but they are NOT attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert not obj1.states[AttachedTo].get_value(obj2)

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
    assert AttachedTo in obj1.states
    assert AttachedTo in obj2.states

    # Obj1 moves towards obj2 and they are attached together.
    obj1.set_linear_velocity(velocity=np.array([3.0, 0, 3.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)
    assert obj1.states[AttachedTo].get_value(obj2)
    assert obj2.states[AttachedTo].get_value(obj1)

    # Save the state.
    state = og.sim.dump_state()

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Unattach obj1 and obj2.
    obj1.states[AttachedTo].set_value(obj2, False)
    assert not obj1.states[AttachedTo].get_value(obj2)
    assert not obj2.states[AttachedTo].get_value(obj1)

    # Obj1 moves away from obj2.
    obj1.set_linear_velocity(velocity=np.array([-2.0, 0, 1.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

    # Load the state where the two objects are attached.
    og.sim.load_state(state)

    # AttachedTo state should be restored correctly
    assert obj1.states[AttachedTo].get_value(obj2)
    assert obj2.states[AttachedTo].get_value(obj1)

    # Apply a large force to obj1 but the two objects cannot move much because obj2 is heavy.
    obj1.set_linear_velocity(velocity=np.array([10.0, 0, 50.0]))
    for i in range(100):
        env.step(np.array([]))  # empty action array since action space is 0 (no robots in the env)

demo_names_to_demos = {
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

    # Shutdown omnigibson
    og.shutdown()


if __name__ == "__main__":
    main()
