import logging
import os

import numpy as np


from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path
from igibson.utils.utils import restoreState


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a slicing task that resets after everything
    To save/load state it combines pybullet save/load functionality and additional iG functions for the extended states
    Loads an empty scene with a desk and an apple. After a while, the apple gets sliced. Then, the scene resets
    This demo also demonstrates how to create manually a grouped object with multiple URDF models, and a
    multiplexed object (two URDF models for the same object) with a model for the full object and a model
    of the grouped slices
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    s = Simulator(mode="gui_interactive" if not headless else "headless", image_width=1280, image_height=720)

    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [-0.3, -0.3, 1.1]
        s.viewer.initial_view_direction = [0.7, 0.6, -0.4]
        s.viewer.reset_viewer()
    scene = EmptyScene(render_floor_plane=True, floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Load a desk
    model_path = os.path.join(get_ig_model_path("breakfast_table", "19203"), "19203.urdf")
    desk = URDFObject(
        filename=model_path, category="breakfast_table", name="19898", scale=np.array([0.8, 0.8, 0.8]), abilities={}
    )
    s.import_object(desk)
    desk.set_position([0, 0, 0.4])

    # Create an URDF object of an apple, but doesn't load it in the simulator
    model_path = os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf")
    whole_obj = URDFObject(model_path, name="00_0", category="apple", scale=np.array([1.0, 1.0, 1.0]))

    object_parts = []
    # Check the parts that compose the apple and create URDF objects of them
    for i, part in enumerate(whole_obj.metadata["object_parts"]):
        part_category = part["category"]
        part_model = part["model"]
        # Scale the offset accordingly
        part_pos = part["pos"] * whole_obj.scale
        part_orn = part["orn"]
        part_model_path = get_ig_model_path(part_category, part_model)
        part_filename = os.path.join(part_model_path, part_model + ".urdf")
        part_obj_name = whole_obj.name + "_part_{}".format(i)
        part_obj = URDFObject(
            part_filename,
            name=part_obj_name,
            category=part_category,
            model_path=part_model_path,
            scale=whole_obj.scale,
        )
        object_parts.append((part_obj, (part_pos, part_orn)))

    # Group the apple parts into a single grouped object
    grouped_parts_obj = ObjectGrouper(object_parts)

    # Create a multiplexed object: either the full apple, or the parts
    multiplexed_obj = ObjectMultiplexer(whole_obj.name + "_multiplexer", [whole_obj, grouped_parts_obj], 0)

    # Finally, load the multiplexed object
    s.import_object(multiplexed_obj)
    whole_obj.set_position([100, 100, -100])
    for i, (part_obj, _) in enumerate(object_parts):
        part_obj.set_position([101 + i, 100, -100])

    multiplexed_obj.set_position([0, 0, 0.72])
    # Let the apple get stable
    for _ in range(100):
        s.step()

    # Save the initial state.
    initial_state_pb = p.saveState()
    initial_state_multiplexed_obj = multiplexed_obj.dump_state()
    logging.info(multiplexed_obj)

    try:
        max_iterations = -1 if not short_exec else 1
        iteration = 0
        while iteration != max_iterations:
            logging.info("Stepping the simulator")
            for _ in range(100):
                s.step()

            logging.info("Slicing the apple and moving the parts")
            # Slice the apple and set the object parts away
            multiplexed_obj.states[object_states.Sliced].set_value(True)

            # Check that the multiplexed changed to the group of parts
            assert isinstance(multiplexed_obj.current_selection(), ObjectGrouper)

            # Move the parts
            for part_obj in multiplexed_obj.objects:
                part_obj.set_position([0, 0, 1])

            logging.info("Stepping the simulator")
            for _ in range(100):
                s.step()

            logging.info("Restoring the state")
            # Restore the state
            restoreState(initial_state_pb)

            # The apple should become whole again
            multiplexed_obj.load_state(initial_state_multiplexed_obj)

            iteration += 1

            s.sync(force_sync=True)

    finally:
        p.removeState(initial_state_pb)
        s.disconnect()


if __name__ == "__main__":
    main()
