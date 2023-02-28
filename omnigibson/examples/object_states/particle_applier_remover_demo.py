import numpy as np
import omnigibson as og
from omnigibson.object_states import Covered
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm, macros
from omnigibson.systems import *
from omnigibson.utils.usd_utils import create_joint
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.constants import ParticleModifyMethod
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf

# Set macros for this example
macros.object_states.particle_modifier.VISUAL_PARTICLES_REMOVAL_LIMIT = 1000
macros.object_states.particle_modifier.PHYSICAL_PARTICLES_REMOVAL_LIMIT = 8000
macros.object_states.particle_modifier.MAX_VISUAL_PARTICLES_APPLIED_PER_STEP = 10
macros.object_states.particle_modifier.MAX_PHYSICAL_PARTICLES_APPLIED_PER_STEP = 40
StainSystem._N_PARTICLES_PER_GROUP = 300

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for fluids)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of ParticleApplier and ParticleRemover object states, which enable objects to either apply arbitrary
    particles and remove arbitrary particles from the simulator, respectively.

    Loads an empty scene with a table, and starts clean to allow particles to be applied or pre-covers the table
    with particles to be removed. The ParticleApplier / ParticleRemover state is applied to an imported cloth object
    and allowed to interact with the table, applying / removing particles from the table.
    """
    og.log.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

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
        "particleApplier": "particleapplication_link",
        "particleRemover": "particleremover_link",
    }

    particle_mapping = {system.name: system for system in [StainSystem, WaterSystem]}
    particle_type = choose_from_options(
        options={name: f"{name} particles will be applied or removed from the simulator" for name in particle_mapping},
        name="particle type",
        random_selection=random_selection,
    )
    particle_system = particle_mapping[particle_type]

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
            "extents": np.array([0.375, 0.375, 0.75]),
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
                particle_system: [],
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
        intensity=1e5,
        position=[-2.0, -2.0, 2.0],
    )

    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="265851637a59eb2f882f822c83877cbc",
        scale=[4.0, 4.0, 4.0],
        position=[0, 0, 0.7],
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
        prim_path="/World/modifier",
        name="modifier",
        category="dishtowel",
        model="Tag_Dishtowel_Basket_Weave_Red",
        scale=np.ones(3) * 2.0,
        visual_only=method_type == "Projection" or particle_system == StainSystem,  # Fluid + adjacency requires the object to have collision geoms active
        abilities=abilities,
    )
    modifier_root_link_path = f"{modifier.prim_path}/base_link"
    modifier._prim = modifier._load(og.sim)
    if method_type == "Projection":
        metalink_path = f"{modifier.prim_path}/{modification_metalink[modifier_type]}"
        og.sim.stage.DefinePrim(metalink_path, "Xform")
        joint_prim = create_joint(
            prim_path=f"{modifier_root_link_path}/{modification_metalink[modifier_type]}_joint",
            body0=modifier_root_link_path,
            body1=metalink_path,
            joint_type="FixedJoint",
            enabled=True,
        )
        local_area_quat = np.array([0, 0.707, 0, 0.707])    # Needs to rotated so the metalink points downwards from cloth
        joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*(local_area_quat[[3, 0, 1, 2]])))
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
        table.states[Covered].set_value(particle_system, True)

        # Take a few steps to let particles settle
        for _ in range(25):
            env.step(np.array([]))

    # Enable camera teleoperation for convenience
    og.sim.enable_viewer_camera_teleoperation()

    # Set the modifier object to be in position to modify particles
    if method_type == "Projection":
        # Higher z to showcase projection volume at work
        z = 1.85
    elif particle_system == StainSystem:
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
            # env.step(np.array([]))
            og.sim.step()

    # Always shut down environment at the end
    env.close()


if __name__ == "__main__":
    main()

