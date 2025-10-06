import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
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
                "model": "ykretu",
                "abilities": {
                    "heatSource": {"requires_toggled_on": True},
                    "toggleable": {},
                },
                "position": [0, 0, 0.61],
            }
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Get reference to stove object
    stove = env.scene.object_registry("name", "stove")

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.2299, -0.8270, 1.4666]),
        orientation=th.tensor([0.4831, 0.2451, 0.3803, 0.7497]),
    )

    # Make sure necessary object states are included with the stove
    assert object_states.HeatSourceOrSink in stove.states
    assert object_states.ToggledOn in stove.states

    # Take a few steps so that visibility propagates
    for _ in range(40):
        env.step([])

    # Heat source is off.
    print("Heat source is OFF.")
    heat_source_state = stove.states[object_states.HeatSourceOrSink].get_value()
    assert not heat_source_state

    # Toggle on stove, notify user
    if not short_exec:
        input("Heat source will now turn ON: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(True)

    assert stove.states[object_states.ToggledOn].get_value()

    # Need to take a step to update the state.
    env.step([])

    # Heat source is on
    heat_source_state = stove.states[object_states.HeatSourceOrSink].get_value()
    assert heat_source_state
    for _ in range(500):
        env.step([])

    # Toggle off stove, notify user
    if not short_exec:
        input("Heat source will now turn OFF: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(False)
    assert not stove.states[object_states.ToggledOn].get_value()
    for _ in range(200):
        env.step([])

    # Move stove, notify user
    if not short_exec:
        input("Heat source is now moving: Press ENTER to continue.")
    stove.set_position_orientation(position=th.tensor([-0.4, 0.0, 0.61]))
    for i in range(100):
        env.step([])

    # Toggle on stove again, notify user
    if not short_exec:
        input("Heat source will now turn ON: Press ENTER to continue.")
    stove.states[object_states.ToggledOn].set_value(True)
    assert stove.states[object_states.ToggledOn].get_value()
    for i in range(500):
        env.step([])

    # Shutdown environment at end
    og.shutdown()


if __name__ == "__main__":
    main()
