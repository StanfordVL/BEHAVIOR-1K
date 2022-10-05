import numpy as np
import igibson as ig
from igibson import object_states
from igibson.macros import gm
from igibson.objects import DatasetObject


def main():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to appropriate viewing pose
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.0792399, -1.30104, 1.51981]),
        orientation=np.array([0.54897692, 0.00110359, 0.00168013, 0.83583509]),
    )

    # Load a stove model
    stove = DatasetObject(
        prim_path=f"/World/stove",
        name="stove",
        category="stove",
        model="101908",
        abilities={"heatSource": {"requires_toggled_on": True}, "toggleable": {},},
    )

    # Make sure necessary object states are included with the stove
    assert object_states.HeatSourceOrSink in stove.states
    assert object_states.ToggledOn in stove.states

    # Import this object into the simulator, and take a step to initialize the object
    ig.sim.import_object(stove)
    stove.set_position(np.array([0, 0, 0.4]))
    env.step(np.array([]))

    # Take a few steps so that visibility propagates
    for _ in range(5):
        env.step(np.array([]))

    # Heat source is off.
    print("Heat source is OFF.")
    heat_source_state, heat_source_position = stove.states[object_states.HeatSourceOrSink].get_value()
    assert not heat_source_state

    # Toggle on stove, notify user
    input("Heat source will now turn ON: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(True)
    assert stove.states[object_states.ToggledOn].get_value()

    # Need to take a step to update the state.
    env.step(np.array([]))

    # Heat source is on
    heat_source_state, heat_source_position = stove.states[object_states.HeatSourceOrSink].get_value()
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
    stove.set_position(np.array([0, 1.0, 0.4]))
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
