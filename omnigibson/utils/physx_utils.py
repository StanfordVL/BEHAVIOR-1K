import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.utils.ui_utils import suppress_omni_log

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.PROTOTYPE_GRAVEYARD_POS = (100.0, 100.0, 100.0)


def create_physx_particle_system(
    prim_path,
    physics_scene_path,
    particle_contact_offset,
    visual_only=False,
    smoothing=True,
    anisotropy=True,
    isosurface=True,
):
    """
    Creates an Omniverse physx particle system at @prim_path. For post-processing visualization effects (anisotropy,
    smoothing, isosurface), see the Omniverse documentation
    (https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics.html?highlight=isosurface#post-processing-for-fluid-rendering)
    for more info

    Args:
        prim_path (str): Stage path to where particle system should be created
        physics_scene_path (str): Stage path to where active physicsScene prim is defined
        particle_contact_offset (float): Distance between particles which triggers a collision (m)
        visual_only (bool): If True, will disable collisions between particles and non-particles,
            as well as self-collisions
        smoothing (bool): Whether to smooth particle positions or not
        anisotropy (bool): Whether to apply anisotropy post-processing when visualizing particles. Stretches generated
            particles in order to make the particle cluster surface appear smoother. Useful for fluids
        isosurface (bool): Whether to apply isosurface mesh to visualize particles. Uses a monolithic surface that
            can have materials attached to it, useful for visualizing fluids

    Returns:
        UsdGeom.PhysxParticleSystem: Generated particle system prim
    """
    # TODO: Add sanity check to make sure GPU dynamics are enabled
    # Create particle system
    stage = lazy.omni.isaac.core.utils.stage.get_current_stage()
    particle_system = lazy.pxr.PhysxSchema.PhysxParticleSystem.Define(stage, prim_path)
    particle_system.CreateSimulationOwnerRel().SetTargets([physics_scene_path])

    # Use a smaller particle size for nicer fluid, and let the sim figure out the other offsets
    particle_system.CreateParticleContactOffsetAttr().Set(particle_contact_offset)

    # Possibly disable collisions if we're only visual
    if visual_only:
        particle_system.GetGlobalSelfCollisionEnabledAttr().Set(False)
        particle_system.GetNonParticleCollisionEnabledAttr().Set(False)

    if anisotropy:
        # apply api and use all defaults
        lazy.pxr.PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())

    if smoothing:
        # apply api and use all defaults
        lazy.pxr.PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())

    if isosurface:
        # apply api and use all defaults
        lazy.pxr.PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
        # Make sure we're not casting shadows
        primVarsApi = lazy.pxr.UsdGeom.PrimvarsAPI(particle_system.GetPrim())
        primVarsApi.CreatePrimvar("doNotCastShadows", lazy.pxr.Sdf.ValueTypeNames.Bool).Set(True)
        # tweak anisotropy min, max, and scale to work better with isosurface:
        if anisotropy:
            ani_api = lazy.pxr.PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
            ani_api.CreateScaleAttr().Set(5.0)
            ani_api.CreateMinAttr().Set(1.0)  # avoids gaps in surface
            ani_api.CreateMaxAttr().Set(2.0)

    return particle_system


def bind_material(prim_path, material_path):
    """
    Binds material located at @material_path to the prim located at @prim_path.

    Args:
        prim_path (str): Stage path to prim to bind material to
        material_path (str): Stage path to material to be bound
    """
    lazy.omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=prim_path,
        material_path=material_path,
        strength=None,
    )


def create_physx_particleset_pointinstancer(
    name,
    particle_system_path,
    physx_particle_system_path,
    prototype_prim_paths,
    particle_group,
    positions,
    self_collision=True,
    fluid=False,
    particle_mass=None,
    particle_density=None,
    orientations=None,
    velocities=None,
    angular_velocities=None,
    scales=None,
    prototype_indices=None,
    enabled=True,
):
    """
    Creates a particle set instancer based on a UsdGeom.PointInstancer at @prim_path on the current stage, with
    the specified parameters.

    Args:
        name (str): Name for this point instancer
        particle_system_path (str): Stage path to particle system (Scope)
        physx_particle_system_path (str): Stage path to physx particle system (PhysxParticleSystem)
        prototype_prim_paths (list of str): Stage path(s) to the prototypes to reference for this particle set.
        particle_group (int): ID for this particle set. Particles from different groups will automatically collide
            with each other. Particles in the same group will have collision behavior dictated by @self_collision
        positions (list of 3-tuple or th.tensor): Particle (x,y,z) positions either as a list or a (N, 3) numpy array
        self_collision (bool): Whether to enable particle-particle collision within the set
            (as defined by @particle_group) or not
        fluid (bool): Whether to simulated the particle set as fluid or not
        particle_mass (None or float): If specified, should be per-particle mass. Otherwise, will be
            inferred from @density. Note: Either @particle_mass or @particle_density must be specified!
        particle_density (None or float): If specified, should be per-particle density and is used to compute total
            point set mass. Otherwise, will be inferred from @density. Note: Either @particle_mass or
            @particle_density must be specified!
        orientations (None or list of 4-array or th.tensor): Particle (x,y,z,w) quaternion orientations, either as a
            list or a (N, 4) numpy array. If not specified, all will be set to canonical orientation (0, 0, 0, 1)
        velocities (None or list of 3-array or th.tensor): Particle (x,y,z) velocities either as a list or a (N, 3)
            numpy array. If not specified, all will be set to 0
        angular_velocities (None or list of 3-array or th.tensor): Particle (x,y,z) angular velocities either as a
            list or a (N, 3) numpy array. If not specified, all will be set to 0
        scales (None or list of 3-array or th.tensor): Particle (x,y,z) scales either as a list or a (N, 3)
            numpy array. If not specified, all will be set to 1.0
        prototype_indices (None or list of int): If specified, should specify which prototype should be used for
            each particle. If None, will use all 0s (i.e.: the first prototype created)
        enabled (bool): Whether to enable this particle instancer. If not enabled, then no physics will be used

    Returns:
        UsdGeom.PointInstancer: Created point instancer prim
    """
    stage = og.sim.stage
    n_particles = len(positions)
    particle_system = lazy.omni.isaac.core.utils.prims.get_prim_at_path(physx_particle_system_path)

    # Create point instancer scope
    prim_path = f"{particle_system_path}/{name}"
    assert not stage.GetPrimAtPath(prim_path), f"Cannot create an instancer scope, scope already exists at {prim_path}!"
    stage.DefinePrim(prim_path, "Scope")

    # Create point instancer
    instancer_prim_path = f"{prim_path}/instancer"
    assert not stage.GetPrimAtPath(
        instancer_prim_path
    ), f"Cannot create a PointInstancer prim, prim already exists at {instancer_prim_path}!"
    instancer = lazy.pxr.UsdGeom.PointInstancer.Define(stage, instancer_prim_path)

    is_isosurface = (
        particle_system.HasAPI(lazy.pxr.PhysxSchema.PhysxParticleIsosurfaceAPI)
        and particle_system.GetAttribute("physxParticleIsosurface:isosurfaceEnabled").Get()
    )

    # Add prototype mesh prim paths to the prototypes relationship attribute for this point set
    # We need to make copies of prototypes for each instancer currently because particles won't render properly
    # if multiple instancers share the same prototypes for some reason
    mesh_list = instancer.GetPrototypesRel()
    prototype_prims = []
    for i, original_path in enumerate(prototype_prim_paths):
        prototype_prim_path = f"{prim_path}/prototype{i}"
        lazy.omni.kit.commands.execute("CopyPrim", path_from=original_path, path_to=prototype_prim_path)
        prototype_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prototype_prim_path)
        # Make sure this prim is invisible if we're using isosurface, and vice versa.
        imageable = lazy.pxr.UsdGeom.Imageable(prototype_prim)
        if is_isosurface:
            imageable.MakeInvisible()
        else:
            imageable.MakeVisible()

        # Move the prototype to the graveyard position so that it won't be visible to the agent
        # We can't directly hide the prototype because it will also hide all the generated particles (if not isosurface)
        prototype_prim.GetAttribute("xformOp:translate").Set(m.PROTOTYPE_GRAVEYARD_POS)

        mesh_list.AddTarget(lazy.pxr.Sdf.Path(prototype_prim_path))
        prototype_prims.append(prototype_prim)

    # Set particle instance default data
    prototype_indices = [0] * n_particles if prototype_indices is None else prototype_indices
    if orientations is None:
        orientations = th.zeros((n_particles, 4))
        orientations[:, -1] = 1.0
    orientations = th.tensor(orientations) if not isinstance(orientations, th.Tensor) else orientations
    orientations = orientations[:, [3, 0, 1, 2]]  # x,y,z,w --> w,x,y,z
    velocities = th.zeros((n_particles, 3)) if velocities is None else velocities
    angular_velocities = th.zeros((n_particles, 3)) if angular_velocities is None else angular_velocities
    scales = th.ones((n_particles, 3)) if scales is None else scales
    assert (
        particle_mass is not None or particle_density is not None
    ), "Either particle mass or particle density must be specified when creating particle instancer!"
    particle_mass = 0.0 if particle_mass is None else particle_mass
    particle_density = 0.0 if particle_density is None else particle_density

    # Set particle states
    instancer.GetProtoIndicesAttr().Set(prototype_indices)
    instancer.GetPositionsAttr().Set(lazy.pxr.Vt.Vec3fArray(positions.tolist()))
    instancer.GetOrientationsAttr().Set(lazy.pxr.Vt.QuathArray.FromNumpy(orientations.cpu().numpy()))
    instancer.GetVelocitiesAttr().Set(lazy.pxr.Vt.Vec3fArray(velocities.tolist()))
    instancer.GetAngularVelocitiesAttr().Set(lazy.pxr.Vt.Vec3fArray(angular_velocities.tolist()))
    instancer.GetScalesAttr().Set(lazy.pxr.Vt.Vec3fArray(scales.tolist()))

    # Take a render step to "lock" the visuals of the prototypes at the graveyard position
    # This needs to happen AFTER setting particle states
    # We suppress a known warning that we have no control over where omni complains about a prototype
    # not being populated yet
    with suppress_omni_log(channels=["omni.hydra.scene_delegate.plugin"]):
        og.sim.render()

    # Then we move the prototypes back to zero offset because otherwise all the generated particles will be offset by
    # the graveyard position. At this point, the prototypes themselves no longer appear at the zero offset (locked at
    # the graveyard position), which is desirable because we don't want the agent to see the prototypes themselves.
    for prototype_prim in prototype_prims:
        prototype_prim.GetAttribute("xformOp:translate").Set((0.0, 0.0, 0.0))

    instancer_prim = instancer.GetPrim()

    lazy.omni.physx.scripts.particleUtils.configure_particle_set(
        instancer_prim,
        physx_particle_system_path,
        self_collision,
        fluid,
        particle_group,
        particle_mass * n_particles,
        particle_density,
    )

    # Set whether the instancer is enabled or not
    instancer_prim.GetAttribute("physxParticle:particleEnabled").Set(enabled)

    # Render three more times to fully propagate changes
    # Omni always complains about a low-level USD thing we have no control over
    # so we suppress the warnings
    with suppress_omni_log(channels=["omni.usd"]):
        for i in range(3):
            og.sim.render()

    # Isosurfaces require an additional physics timestep before they're actually rendered
    if is_isosurface:
        og.log.warning(
            f"Creating an instancer that uses isosurface {instancer_prim_path}. "
            f"The rendering of these particles will have a delay of one timestep."
        )

    return instancer_prim


def apply_force_at_pos(prim, force, pos):
    if isinstance(force, th.Tensor):
        force = force.cpu().numpy()
    if isinstance(pos, th.Tensor):
        pos = pos.cpu().numpy()
    prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim.prim_path)
    og.sim.psi.apply_force_at_pos(og.sim.stage_id, prim_id, force, pos)


def apply_torque(prim, foward_vect, roll_torque_scalar):
    if isinstance(foward_vect, th.Tensor):
        foward_vect = foward_vect.cpu().numpy()
    prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim.prim_path)
    og.sim.psi.apply_torque(og.sim.stage_id, prim_id, foward_vect * roll_torque_scalar)
