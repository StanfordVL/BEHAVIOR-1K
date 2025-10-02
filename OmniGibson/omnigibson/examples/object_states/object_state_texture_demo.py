import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm, macros
from omnigibson.utils.constants import ParticleModifyMethod

# Make sure object states are enabled, we're using GPU dynamics, and HQ rendering is enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True


def main(random_selection=False, headless=False, short_exec=False):
    # Create the scene config to load -- empty scene plus a steak
    cfg = {
        "env": {
            "rendering_frequency": 60,  # for HQ rendering
        },
        "scene": {
            "type": "Scene",
            "floor_plane_visible": True,
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "steak",
                "category": "steak",
                "model": "ppykkp",
                "abilities": {
                    "freezable": {},
                    "cookable": {},
                    "burnable": {},
                    "saturable": {},
                    "particleRemover": {
                        "method": ParticleModifyMethod.ADJACENCY,
                        "conditions": {
                            # For a specific particle system, this specifies what conditions are required in order for the
                            # particle applier / remover to apply / remover particles associated with that system
                            # The list should contain functions with signature condition() --> bool,
                            # where True means the condition is satisfied
                            # In this case, we only allow our steak to absorb water, with no conditions needed.
                            # This is needed for the Saturated ("saturable") state so that we can modify the texture
                            # according to the water.
                            # NOTE: This will only change color if gm.ENABLE_HQ_RENDERING and gm.USE_GPU_DYNAMICS is
                            # enabled!
                            "water": [],
                        },
                    },
                },
                "position": [0, 0, 0.59],
                "fixed_base": True,
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.2425, 0.0894, 0.7872]),
        orientation=th.tensor([0.2494, 0.3583, 0.7384, 0.5140]),
    )

    # Grab reference to object of interest
    obj = env.scene.object_registry("name", "steak")

    # Make sure all the appropriate states are in the object
    assert object_states.Frozen in obj.states
    assert object_states.Cooked in obj.states
    assert object_states.Burnt in obj.states
    assert object_states.Saturated in obj.states

    def report_states():
        # Make sure states are propagated before printing
        for i in range(5):
            env.step(th.empty(0))

        print("=" * 20)
        print("temperature:", obj.states[object_states.Temperature].get_value())
        print("obj is frozen:", obj.states[object_states.Frozen].get_value())
        print("obj is cooked:", obj.states[object_states.Cooked].get_value())
        print("obj is burnt:", obj.states[object_states.Burnt].get_value())
        print("obj is soaked:", obj.states[object_states.Saturated].get_value(env.scene.get_system("water")))

    # Report default states
    print("==== Initial state ====")
    report_states()

    # Notify user that we're about to freeze the object, and then freeze the object
    if not short_exec:
        input("\nObject will be frozen. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(-50)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Notify user that we're about to cook the object, and then cook the object
    if not short_exec:
        input("\nObject will be cooked. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(100)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Notify user that we're about to burn the object, and then burn the object
    if not short_exec:
        input("\nObject will be burned. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(250)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Notify user that we're about to reset the object to its default state, and then reset state
    if not short_exec:
        input("\nObject will be reset to default state. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(macros.object_states.temperature.DEFAULT_TEMPERATURE)
    obj.states[object_states.MaxTemperature].set_value(macros.object_states.temperature.DEFAULT_TEMPERATURE)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Notify user that we're about to soak the object, and then soak the object
    if not short_exec:
        input("\nObject will be saturated with water. Press ENTER to continue.")
    obj.states[object_states.Saturated].set_value(env.scene.get_system("water"), True)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Notify user that we're about to unsoak the object, and then unsoak the object
    if not short_exec:
        input("\nObject will be unsaturated with water. Press ENTER to continue.")
    obj.states[object_states.Saturated].set_value(env.scene.get_system("water"), False)
    for _ in range(50):
        og.sim.render()
    report_states()

    # Close environment at the end
    if not short_exec:
        input("Demo completed. Press ENTER to shutdown environment.")
    og.shutdown()


if __name__ == "__main__":
    main()
