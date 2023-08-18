import numpy as np
import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm, macros
from omnigibson.systems import get_system
from omnigibson.utils.usd_utils import create_joint
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.constants import ParticleModifyMethod
from pxr import Gf

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

    modification_metalink = {
        "particleApplier": "particleapplier_link",
        "particleRemover": "particleremover_link",
    }

    particle_types = ["stain", "water"]
    particle_type = choose_from_options(
        options={name: f"{name} particles will be applied or removed from the simulator" for name in particle_types},
        name="particle type",
        random_selection=random_selection,
    )

    modification_method = {
        "Adjacency": ParticleModifyMethod.ADJACENCY,
        "Projection": ParticleModifyMethod.PROJECTION,
    }

    projection_mesh_params = {
        "Adjacency": None,
        "Projection": {
            # Either Cone or Cylinder; shape of the projection where particles can be applied / removed
            "type": "Cone",
            # Size of the cone
            "extents": np.array([0.1875, 0.1875, 0.375]),
        },
    }

    method_type = choose_from_options(
        options={
            "Adjacency": "Close proximity to the object will be used to determine whether particles can be applied / removed",
            "Projection": "A Cone or Cylinder shape protruding from the object will be used to determine whether particles can be applied / removed",
        },
        name="modifier method type",
        random_selection=random_selection,
    )

    # Create the ability kwargs to pass to the object state
    abilities = {
        modifier_type: {
            "method": modification_method[method_type],
            "conditions": {
                # For a specific particle system, this specifies what conditions are required in order for the
                # particle applier / remover to apply / remover particles associated with that system
                # The list should contain functions with signature condition() --> bool,
                # where True means the condition is satisified
                particle_type: [],
            },
            "projection_mesh_params": projection_mesh_params[method_type],
        }
    }

    # Define objects to load: a light, table, and cloth
    light_cfg = dict(
        type="LightObject",
        name="light",
        light_type="Sphere",
        radius=0.01,
        intensity=1e8,
        position=[-2.0, -2.0, 2.0],
    )

    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="kwmfdg",
        bounding_box=[3.402, 1.745, 1.175],
        position=[0, 0, 0.98],
    )

    # Create the scene config to load -- empty scene with a light and table
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [light_cfg, table_cfg],
    }

    # Sanity check inputs: Remover + Adjacency + Fluid will not work because we are using a visual_only
    # object, so contacts will not be triggered with this object

    # Load the environment, then immediately stop the simulator since we need to add in the modifier object
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
    og.sim.stop()

    # Grab references to table
    table = env.scene.object_registry("name", "table")

    # Set the viewer camera appropriately
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-1.61340969, -1.79803028,  2.53167412]),
        orientation=np.array([ 0.46291845, -0.12381886, -0.22679218,  0.84790371]),
    )

    # If we're using a projection volume, we manually add in the required metalink required in order to use the volume
    modifier = DatasetObject(
        name="modifier",
        category="dishtowel",
        model="dtfspn",
        bounding_box=[0.341, 0.466, 0.07],
        visual_only=method_type == "Projection",  # Non-fluid adjacency requires the object to have collision geoms active
        abilities=abilities,
    )
    modifier_root_link_path = f"{modifier.prim_path}/base_link"
    modifier._prim = modifier._load()
    if method_type == "Projection":
        metalink_path = f"{modifier.prim_path}/{modification_metalink[modifier_type]}"
        og.sim.stage.DefinePrim(metalink_path, "Xform")
        create_joint(
            prim_path=f"{modifier_root_link_path}/{modification_metalink[modifier_type]}_joint",
            body0=modifier_root_link_path,
            body1=metalink_path,
            joint_type="FixedJoint",
            enabled=True,
        )
    modifier._post_load()
    modifier._loaded = True
    og.sim.import_object(modifier)
    modifier.set_position(np.array([0, 0, 5.0]))

    # Play the simulator and take some environment steps to let the objects settle
    og.sim.play()
    for _ in range(25):
        env.step(np.array([]))

    # If we're removing particles, set the table's covered state to be True
    if modifier_type == "particleRemover":
        table.states[Covered].set_value(get_system(particle_type), True)

        # Take a few steps to let particles settle
        for _ in range(25):
            env.step(np.array([]))

    # Enable camera teleoperation for convenience
    og.sim.enable_viewer_camera_teleoperation()

    # Set the modifier object to be in position to modify particles
    if method_type == "Projection":
        # Higher z to showcase projection volume at work
        z = 1.85
    elif particle_type == "stain":
        # Lower z needed to allow for adjacency bounding box to overlap properly
        z = 1.175
    else:
        # Higher z needed for actual physical interaction to accommodate non-negligible particle radius
        z = 1.22
    modifier.keep_still()
    modifier.set_position_orientation(
        position=np.array([0, 0.3, z]),
        orientation=np.array([0, 0, 0, 1.0]),
    )

    # Move object in square around table
    deltas = [
        [150, np.array([-0.01, 0, 0])],
        [60, np.array([0, -0.01, 0])],
        [150, np.array([0.01, 0, 0])],
        [60, np.array([0, 0.01, 0])],
    ]
    for t, delta in deltas:
        for i in range(t):
            modifier.set_position(modifier.get_position() + delta)
            env.step(np.array([]))

    # Always shut down environment at the end
    env.close()


if __name__ == "__main__":
    main()

