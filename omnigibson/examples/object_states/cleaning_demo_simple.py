import logging
from collections import OrderedDict
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.systems import DustSystem, StainSystem, WaterSystem
from omnigibson.utils.constants import ParticleModifyMethod


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a cleaning task that resets after everything has been cleaned
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

    # Define objects to load into the environment
    sink_cfg = OrderedDict(
        type="DatasetObject",
        name="sink",
        category="sink",
        model="sink_1",
        scale=[0.8, 0.8, 0.8],
        abilities={"toggleable": {}, "waterSource": {}, "waterSink": {}},
        position=[-0.7, 0, 0.53],
    )

    def check_water_saturation(obj):
        return obj.states[object_states.Saturated].get_value(WaterSystem)

    brush_cfg = OrderedDict(
        type="DatasetObject",
        name="brush",
        category="scrub_brush",
        model="scrub_brush_000",
        avg_obj_dims={"size": [0.1, 0.1, 0.1], "density": 67.0},
        fit_avg_dim_volume=True,
        position=[1.0, 0, 0.4],
        abilities={
            "saturable": {},
            "particleRemover": {
                "method": ParticleModifyMethod.ADJACENCY,
                "conditions": {
                    # For a specific particle system, this specifies what conditions are required in order for the
                    # particle applier / remover to apply / remover particles associated with that system
                    # The list should contain functions with signature condition() --> bool,
                    # where True means the condition is satisfied
                    # In this case, we only allow our cleaning tool to remove stains and dust particles if
                    # the object is saturated with water, i.e.: it's "soaked" with water particles
                    StainSystem: [check_water_saturation],
                    DustSystem: [check_water_saturation],
                    WaterSystem: [],
                },
            },
        },
    )

    # Desk that's dusty
    desk_cfg = OrderedDict(
        type="DatasetObject",
        name="desk",
        category="breakfast_table",
        model="19203",
        scale=[0.8, 0.8, 0.8],
        position=[1.0, 0, 0.48],
    )

    # Bowl with stains
    bowl_cfg = OrderedDict(
        type="DatasetObject",
        name="bowl",
        category="bowl",
        model="68_0",
        scale=np.array([0.8, 0.8, 0.8]),
        position=[-1.0, 0, 0.48],
    )

    cfg["objects"] = [sink_cfg, brush_cfg, desk_cfg, bowl_cfg]

    # Create the environment!
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to ideal angle for viewing objects
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.782289, -0.633009,  1.4475  ]),
        orientation=np.array([ 0.48871723, -0.24618907, -0.37654978,  0.74750028]),
    )

    # Take a few steps to let the objects settle, and then sanity check the initial state
    for _ in range(10):
        env.step(np.array([]))              # Empty action since no robots are in the scene

    sink = env.scene.object_registry("name", "sink")
    brush = env.scene.object_registry("name", "brush")
    desk = env.scene.object_registry("name", "desk")
    bowl = env.scene.object_registry("name", "bowl")

    assert sink.states[object_states.ToggledOn].set_value(True)
    assert desk.states[object_states.Covered].set_value(DustSystem, True)
    assert bowl.states[object_states.OnTop].set_value(desk, True, use_ray_casting_method=True)
    assert brush.states[object_states.OnTop].set_value(desk, True, use_ray_casting_method=True)
    assert bowl.states[object_states.Covered].set_value(StainSystem, True)

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
                desk.states[object_states.Covered].get_value(DustSystem)
                and bowl.states[object_states.Covered].get_value(StainSystem)
                and steps != max_steps
            ):
                steps += 1
                env.step(np.array([]))
                logging.info(f"Step {steps}")

            if not desk.states[object_states.Covered].get_value(DustSystem):
                logging.info("Reset because Table cleaned")
            elif not bowl.states[object_states.Covered].get_value(StainSystem):
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
