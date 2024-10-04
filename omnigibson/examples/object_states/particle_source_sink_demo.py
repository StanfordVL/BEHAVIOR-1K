import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.constants import ParticleModifyCondition

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
        "env": {
            "rendering_frequency": 60,  # for HQ rendering
        },
        "scene": {
            "type": "Scene",
        },
    }

    # Define objects to load into the environment
    sink_cfg = dict(
        type="DatasetObject",
        name="sink",
        category="sink",
        model="egwapq",
        bounding_box=[2.427, 0.625, 1.2],
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [
                        (ParticleModifyCondition.TOGGLEDON, True)
                    ],  # Must be toggled on for water source to be active
                },
                "initial_speed": 0.0,  # Water merely falls out of the spout
            },
            "particleSink": {
                "conditions": {
                    "water": [],  # No conditions, always sinking nearby particles
                },
            },
        },
        position=[0.0, 0, 0.42],
    )

    cfg["objects"] = [sink_cfg]

    # Create the environment!
    env = og.Environment(configs=cfg)

    # Set camera to ideal angle for viewing objects
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.37860532, -0.65396566, 1.4067066]),
        orientation=th.tensor([0.49909498, 0.15201752, 0.24857062, 0.81609284]),
    )

    # Take a few steps to let the objects settle, and then turn on the sink
    for _ in range(10):
        env.step(th.empty(0))  # Empty action since no robots are in the scene

    sink = env.scene.object_registry("name", "sink")
    assert sink.states[object_states.ToggledOn].set_value(True)

    # Take a step, and save the state
    env.step(th.empty(0))
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
                env.step(th.empty(0))
            og.log.info("Max steps reached; resetting.")

            # Reset to the initial state
            og.sim.load_state(initial_state)

            iteration += 1

    finally:
        # Always shut down environment at the end
        og.clear()


if __name__ == "__main__":
    main()
