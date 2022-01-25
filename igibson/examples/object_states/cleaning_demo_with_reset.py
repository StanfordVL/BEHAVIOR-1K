import logging
import os

import numpy as np


from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path
from igibson.utils.utils import restoreState


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a cleaning task that resets after everything has been cleaned
    To save/load state it combines pybullet save/load functionality and additional iG functions for the extended states
    Loads an empty scene with a sink, a dusty table and a dirty and stained bowl, and a cleaning tool
    If everything is cleaned, or after N steps, the scene resets to the initial state
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    s = Simulator(mode="gui_interactive" if not headless else "headless", image_width=1280, image_height=720)

    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [-0.5, -0.4, 1.5]
        s.viewer.initial_view_direction = [0.7, 0.1, -0.7]
        s.viewer.reset_viewer()

    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Load sink ON
    model_path = os.path.join(get_ig_model_path("sink", "sink_1"), "sink_1.urdf")
    sink = URDFObject(
        filename=model_path,
        category="sink",
        name="sink_1",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"toggleable": {}, "waterSource": {}},
    )
    s.import_object(sink)
    sink.set_position([1, 1, 0.8])
    assert sink.states[object_states.ToggledOn].set_value(True)

    # Load cleaning tool
    model_path = get_ig_model_path("scrub_brush", "scrub_brush_000")
    model_filename = os.path.join(model_path, "scrub_brush_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
    avg = {"size": max_bbox, "density": 67.0}
    brush = URDFObject(
        filename=model_filename,
        category="scrub_brush",
        name="scrub_brush",
        avg_obj_dims=avg,
        fit_avg_dim_volume=True,
        model_path=model_path,
    )
    s.import_object(brush)
    brush.set_position([0, -2, 0.4])

    # Load table with dust
    model_path = os.path.join(get_ig_model_path("breakfast_table", "19203"), "19203.urdf")
    desk = URDFObject(
        filename=model_path,
        category="breakfast_table",
        name="19898",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"dustyable": {}},
    )
    s.import_object(desk)
    desk.set_position([1, -2, 0.4])
    assert desk.states[object_states.Dusty].set_value(True)

    # Load a bowl with stains
    model_path = os.path.join(get_ig_model_path("bowl", "68_0"), "68_0.urdf")
    bowl = URDFObject(filename=model_path, category="bowl", name="bowl", abilities={"dustyable": {}, "stainable": {}})
    s.import_object(bowl)
    assert bowl.states[object_states.OnTop].set_value(desk, True, use_ray_casting_method=True)
    assert bowl.states[object_states.Stained].set_value(True)

    # Save the initial state.
    pb_initial_state = p.saveState()  # Save pybullet state (kinematics)
    brush_initial_extended_state = brush.dump_state()  # Save brush extended state
    logging.info(brush_initial_extended_state)
    desk_initial_extended_state = desk.dump_state()  # Save desk extended state
    logging.info(desk_initial_extended_state)
    bowl_initial_extended_state = bowl.dump_state()  # Save bowl extended state
    logging.info(bowl_initial_extended_state)

    # Main simulation loop.
    max_steps = 1000
    max_iterations = -1 if not short_exec else 1
    iteration = 0
    try:
        while iteration != max_iterations:
            # Keep stepping until table or bowl are clean, or we reach 1000 steps
            steps = 0
            while (
                desk.states[object_states.Dusty].get_value()
                and bowl.states[object_states.Stained].get_value()
                and steps != max_steps
            ):
                steps += 1
                s.step()
                logging.info("Step {}".format(steps))

            if not desk.states[object_states.Dusty].get_value():
                logging.info("Reset because Table cleaned")
            elif not bowl.states[object_states.Stained].get_value():
                logging.info("Reset because Bowl cleaned")
            else:
                logging.info("Reset because max steps")

            # Reset to the initial state
            restoreState(pb_initial_state)
            brush.load_state(brush_initial_extended_state)
            brush.force_wakeup()
            desk.load_state(desk_initial_extended_state)
            desk.force_wakeup()
            bowl.load_state(bowl_initial_extended_state)
            bowl.force_wakeup()

            iteration += 1

    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
