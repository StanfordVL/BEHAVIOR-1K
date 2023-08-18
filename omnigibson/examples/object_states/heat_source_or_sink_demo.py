import numpy as np
import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main():
    # Create the scene config to load -- empty scene with a stove object added
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "stove",
                "category": "stove",
                "model": "qbjiva",
                "bounding_box": [1.611, 0.769, 1.147],
                "abilities": {
                    "heatSource": {"requires_toggled_on": True},
                    "toggleable": {},
                },
                "position": [0, 0, 0.61],
            }
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Get reference to stove object
    stove = env.scene.object_registry("name", "stove")

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.0792399, -1.30104, 1.51981]),
        orientation=np.array([0.54897692, 0.00110359, 0.00168013, 0.83583509]),
    )

    # Make sure necessary object states are included with the stove
    assert object_states.HeatSourceOrSink in stove.states
    assert object_states.ToggledOn in stove.states

    # Take a few steps so that visibility propagates
    for _ in range(5):
        env.step(np.array([]))

    # Heat source is off.
    print("Heat source is OFF.")
    heat_source_state = stove.states[object_states.HeatSourceOrSink].get_value()
    assert not heat_source_state

    # Toggle on stove, notify user
    input("Heat source will now turn ON: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(True)

    assert stove.states[object_states.ToggledOn].get_value()

    # Need to take a step to update the state.
    env.step(np.array([]))

    # Heat source is on
    heat_source_state = stove.states[object_states.HeatSourceOrSink].get_value()
    assert heat_source_state
    for _ in range(500):
        env.step(np.array([]))

    # Toggle off stove, notify user
    input("Heat source will now turn OFF: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(False)
    assert not stove.states[object_states.ToggledOn].get_value()
    for _ in range(200):
        env.step(np.array([]))

    # Move stove, notify user
    input("Heat source is now moving: Press ENTER to continue.")
    stove.set_position(np.array([0, 1.0, 0.61]))
    for i in range(100):
        env.step(np.array([]))

    # Toggle on stove again, notify user
    input("Heat source will now turn ON: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(True)
    assert stove.states[object_states.ToggledOn].get_value()
    for i in range(500):
        env.step(np.array([]))

    # Shutdown environment at end
    env.close()


if __name__ == "__main__":
    main()
