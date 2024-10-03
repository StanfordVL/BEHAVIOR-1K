import torch as th

import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.object_states import Covered, ToggledOn
from omnigibson.utils.constants import ParticleModifyMethod
from omnigibson.utils.ui_utils import choose_from_options

# Set macros for this example
macros.object_states.particle_modifier.VISUAL_PARTICLES_REMOVAL_LIMIT = 1000
macros.object_states.particle_modifier.PHYSICAL_PARTICLES_REMOVAL_LIMIT = 8000
macros.object_states.particle_modifier.MAX_VISUAL_PARTICLES_APPLIED_PER_STEP = 4
macros.object_states.particle_modifier.MAX_PHYSICAL_PARTICLES_APPLIED_PER_STEP = 40
macros.object_states.covered.MAX_VISUAL_PARTICLES = 300

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for fluids)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_HQ_RENDERING = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of ParticleApplier and ParticleRemover object states, which enable objects to either apply arbitrary
    particles and remove arbitrary particles from the simulator, respectively.

    Loads an empty scene with a table, and starts clean to allow particles to be applied or pre-covers the table
    with particles to be removed. The ParticleApplier / ParticleRemover state is applied to an imported cloth object
    and allowed to interact with the table, applying / removing particles from the table.

    NOTE: The key difference between ParticleApplier/Removers and ParticleSource/Sinks is that Applier/Removers
    requires contact (if using ParticleProjectionMethod.ADJACENCY) or overlap
    (if using ParticleProjectionMethod.PROJECTION) in order to spawn / remove particles, and generally only spawn
    particles at the contact points. ParticleSource/Sinks are special cases of ParticleApplier/Removers that
    always use ParticleProjectionMethod.PROJECTION and always spawn / remove particles within their projection volume,
    irregardless of overlap with other objects!
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose what configuration to load
    modifier_type = choose_from_options(
        options={
            "particleApplier": "Demo object's ability to apply particles in the simulator",
            "particleRemover": "Demo object's ability to remove particles from the simulator",
        },
        name="particle modifier type",
        random_selection=random_selection,
    )
    particle_types = ["salt", "water"]
    particle_type = choose_from_options(
        options={name: f"{name} particles will be applied or removed from the simulator" for name in particle_types},
        name="particle type",
        random_selection=random_selection,
    )

    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="kwmfdg",
        bounding_box=[3.402, 1.745, 1.175],
        position=[0, 0, 0.98],
    )
    tool_cfg = dict(
        type="DatasetObject",
        name="tool",
        visual_only=True,
        position=[0, 0.3, 5.0],
    )

    if modifier_type == "particleRemover":
        if particle_type == "salt":
            # only ask this question if the modifier type is salt particleRemover
            method_type = choose_from_options(
                options={
                    "Adjacency": "Close proximity to the object will be used to determine whether particles can be applied / removed",
                    "Projection": "A Cone or Cylinder shape protruding from the object will be used to determine whether particles can be applied / removed",
                },
                name="modifier method type",
                random_selection=random_selection,
            )
        else:
            # If the particle type is water, the remover is always adjacency type
            method_type = "Adjacency"
        if method_type == "Adjacency":
            # use dishtowel to remove adjacent particles
            tool_cfg["category"] = "dishtowel"
            tool_cfg["model"] = "dtfspn"
            tool_cfg["bounding_box"] = [0.34245, 0.46798, 0.07]
        elif method_type == "Projection":
            # use vacuum to remove projections particles
            tool_cfg["category"] = "vacuum"
            tool_cfg["model"] = "wikhik"
    else:
        # If the modifier type is particleApplier, the applier is always projection type
        method_type = "Projection"

        if particle_type == "salt":
            # use salt shaker to apply salt particles
            tool_cfg["category"] = "salt_shaker"
            tool_cfg["model"] = "iomwtn"
        else:
            # use water atomizer to apply water particles
            tool_cfg["category"] = "water_atomizer"
            tool_cfg["model"] = "lfarai"

    # Create the scene config to load -- empty scene with a light and table
    cfg = {
        "env": {
            "rendering_frequency": 60,  # for HQ rendering
        },
        "scene": {
            "type": "Scene",
        },
        "objects": [table_cfg, tool_cfg],
    }

    # Sanity check inputs: Remover + Adjacency + Fluid will not work because we are using a visual_only
    # object, so contacts will not be triggered with this object

    # Load the environment, then immediately stop the simulator since we need to add in the modifier object
    env = og.Environment(configs=cfg)
    og.sim.stop()

    # Grab references to table and tool
    table = env.scene.object_registry("name", "table")
    tool = env.scene.object_registry("name", "tool")

    # Set the viewer camera appropriately
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-1.61340969, -1.79803028, 2.53167412]),
        orientation=th.tensor([0.46291845, -0.12381886, -0.22679218, 0.84790371]),
    )

    # Play the simulator and take some environment steps to let the objects settle
    og.sim.play()
    for _ in range(25):
        env.step(th.empty(0))

    # If we're removing particles, set the table's covered state to be True
    if modifier_type == "particleRemover":
        table.states[Covered].set_value(env.scene.get_system(particle_type), True)
        # Take a few steps to let particles settle
        for _ in range(25):
            env.step(th.empty(0))

    # If the particle remover/applier is projection type, set the turn on shaker
    if method_type == "Projection":
        tool.states[ToggledOn].set_value(True)

    # Enable camera teleoperation for convenience
    og.sim.enable_viewer_camera_teleoperation()

    tool.keep_still()

    # Set the modifier object to be in position to modify particles
    if modifier_type == "particleRemover" and method_type == "Projection":
        tool.set_position_orientation(
            position=[0, 0.3, 1.45],
            orientation=[0, 0, 0, 1.0],
        )
    elif modifier_type == "particleRemover" and method_type == "Adjacency":
        tool.set_position_orientation(
            position=[0, 0.3, 1.175],
            orientation=[0, 0, 0, 1.0],
        )
    elif modifier_type == "particleApplier" and particle_type == "water":
        tool.set_position_orientation(
            position=[0, 0.3, 1.4],
            orientation=[0.3827, 0, 0, 0.9239],
        )
    else:
        tool.set_position_orientation(
            position=[0, 0.3, 1.5],
            orientation=[0.7071, 0, 0.7071, 0],
        )

    # Move object in square around table
    deltas = [
        [130, th.tensor([-0.01, 0, 0])],
        [60, th.tensor([0, -0.01, 0])],
        [130, th.tensor([0.01, 0, 0])],
        [60, th.tensor([0, 0.01, 0])],
    ]
    for t, delta in deltas:
        for _ in range(t):
            tool.set_position_orientation(position=tool.get_position_orientation()[0] + delta)
            env.step(th.empty(0))

    # Always shut down environment at the end
    og.clear()


if __name__ == "__main__":
    main()
