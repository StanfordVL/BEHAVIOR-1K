import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

# Make sure object states are enabled and GPU dynamics are used
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of ParticleSource and ParticleSink object states, which enable objects to either spawn arbitrary
    particles and remove arbitrary particles from the simulator, respectively.

    Loads an empty scene with a sink, which is enabled with both the ParticleSource and ParticleSink states.
    The sink's particle source is located at the faucet spout and spawns a continuous stream of water particles,
    which is then destroyed ("sunk") by the sink's particle sink located at the drain.

    NOTE: The key difference between ParticleApplier/Removers and ParticleSource/Sinks is that Applier/Removers
    requires contact (if using ParticleProjectionMethod.ADJACENCY) or overlap
    (if using ParticleProjectionMethod.PROJECTION) in order to spawn / remove particles, and generally only spawn
    particles at the contact points. ParticleSource/Sinks are special cases of ParticleApplier/Removers that
    always use ParticleProjectionMethod.PROJECTION and always spawn / remove particles within their projection volume,
    irregardless of overlap with other objects!
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        }
    }

    def check_toggledon(obj):
        return obj.states[object_states.ToggledOn].get_value()

    # Define objects to load into the environment
    sink_cfg = dict(
        type="DatasetObject",
        name="sink",
        category="sink",
        model="yfaufu",
        scale=[0.8, 0.8, 0.8],
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [check_toggledon],   # Must be toggled on for water source to be active
                },
                "source_radius": 0.025,
                "source_height": 0.10,
                "initial_speed": 0.0,               # Water merely falls out of the spout
            },
            "particleSink": {
                "conditions": {
                    "water": None,  # No conditions, always sinking nearby particles
                },
                "sink_radius": 0.20,
                "sink_height": 0.20,
            },
        },
        position=[-0.7, 0, 0.56],
    )

    cfg["objects"] = [sink_cfg]

    # Create the environment!
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to ideal angle for viewing objects
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.71452157, -0.88294428,  1.85640559]),
        orientation=np.array([ 0.44909348, -0.00142818, -0.00284131,  0.89347912]),
    )

    # Take a few steps to let the objects settle, and then turn on the sink
    for _ in range(10):
        env.step(np.array([]))              # Empty action since no robots are in the scene

    sink = env.scene.object_registry("name", "sink")
    assert sink.states[object_states.ToggledOn].set_value(True)

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
            while steps != max_steps:
                steps += 1
                env.step(np.array([]))
            og.log.info("Max steps reached; resetting.")

            # Reset to the initial state
            og.sim.load_state(initial_state)

            iteration += 1

    finally:
        # Always shut down environment at the end
        env.close()


if __name__ == "__main__":
    main()
