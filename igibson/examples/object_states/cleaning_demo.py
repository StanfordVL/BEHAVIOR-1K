import logging

from igibson import object_states
from igibson.objects.ycb_object import YCBObject
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a cleaning task
    Loads an interactive scene and sets all object surface to be dirty
    Loads also a cleaning tool that can be soaked in water and used to clean objects if moved manually
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless", image_width=1280, image_height=720, device_idx=0
    )
    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [-0.7, 2.4, 1.1]
        s.viewer.initial_view_direction = [0, 0.9, -0.5]
        s.viewer.reset_viewer()

    # Only load a few objects
    load_object_categories = ["breakfast_table", "bottom_cabinet", "sink", "stove", "fridge", "window"]
    scene = InteractiveIndoorScene(
        "Rs_int", texture_randomization=False, object_randomization=False, load_object_categories=load_object_categories
    )
    s.import_scene(scene)

    # Load a cleaning tool
    block = YCBObject(name="036_wood_block", abilities={"soakable": {}, "cleaningTool": {}})
    s.import_object(block)
    block.set_position([-1.4, 3.0, 1.5])

    # Set everything that can go dirty and activate the water sources
    stateful_objects = set(
        scene.get_objects_with_state(object_states.Dusty)
        + scene.get_objects_with_state(object_states.Stained)
        + scene.get_objects_with_state(object_states.WaterSource)
    )
    for obj in stateful_objects:
        if object_states.Dusty in obj.states:
            logging.info("Setting object to be Dusty")
            obj.states[object_states.Dusty].set_value(True)

        if object_states.Stained in obj.states:
            logging.info("Setting object to be Stained")
            obj.states[object_states.Stained].set_value(True)

        if object_states.WaterSource in obj.states and object_states.ToggledOn in obj.states:
            logging.info("Setting water source object to be ToggledOn")
            obj.states[object_states.ToggledOn].set_value(True)

    max_steps = -1 if not short_exec else 1000
    step = 0
    try:
        while step != max_steps:
            s.step()
            step += 1
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
