import logging

import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a cleaning task that resets after everything has been cleaned
    To save/load state it combines pybullet save/load functionality and additional iG functions for the extended states
    Loads an empty scene with a sink, a dusty table and a dirty and stained bowl, and a cleaning tool
    If everything is cleaned, or after N steps, the scene resets to the initial state
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Load sink ON
    sink = DatasetObject(
        prim_path="/World/sink",
        name="sink",
        category="sink",
        model="sink_1",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"toggleable": {}, "waterSource": {}, "waterSink": {}},
    )
    og.sim.import_object(sink)
    sink.set_position([1, 1, 0.8])

    # Load cleaning tool
    avg = {"size": [0.1, 0.1, 0.1], "density": 67.0}
    brush = DatasetObject(
        prim_path="/World/brush",
        name="brush",
        category="scrub_brush",
        model="scrub_brush_000",
        avg_obj_dims=avg,
        fit_avg_dim_volume=True,
    )
    og.sim.import_object(brush)
    brush.set_position([0, -2, 0.4])

    # Load table with dust
    desk = DatasetObject(
        prim_path="/World/desk",
        name="desk",
        category="breakfast_table",
        model="19203",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"dustyable": {}},
    )
    og.sim.import_object(desk)
    desk.set_position([1, -2, 0.4])

    # Load a bowl with stains
    bowl = DatasetObject(
        prim_path="/World/bowl",
        name="bowl",
        category="bowl",
        model="68_0",
        scale=np.array([0.8, 0.8, 0.8]),
        abilities={"dustyable": {}, "stainable": {}},
    )
    og.sim.import_object(bowl)

    # Take a sim step to make sure everything is initialized properly, and then sanity check the initial state
    env.step(np.array([]))              # Empty action since no robots are in the scene

    assert sink.states[object_states.ToggledOn].set_value(True)
    assert desk.states[object_states.Dusty].set_value(True)
    assert bowl.states[object_states.OnTop].set_value(desk, True, use_ray_casting_method=True)
    assert bowl.states[object_states.Stained].set_value(True)

    # Take a step, and save the state
    env.step(np.array([]))
    initial_state = og.sim.dump_state()

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
                env.step(np.array([]))
                logging.info(f"Step {steps}")

            if not desk.states[object_states.Dusty].get_value():
                logging.info("Reset because Table cleaned")
            elif not bowl.states[object_states.Stained].get_value():
                logging.info("Reset because Bowl cleaned")
            else:
                logging.info("Reset because max steps")

            # Reset to the initial state
            og.sim.load_state(initial_state)

            iteration += 1

    finally:
        # Always shut down environment at the end
        env.close()


if __name__ == "__main__":
    main()
