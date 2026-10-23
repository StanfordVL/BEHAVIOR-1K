"""
NVIDIA PhysX (via Isaac Sim / Omniverse Kit) implementation of the PhysicsBackend interface.

This module is the sole owner of every direct ``omni.physx`` / ``omni.physics.tensors`` /
``isaacsim.core.*`` physics call in the codebase -- almost all of the logic here is relocated
verbatim from ``Simulator`` and ``omnigibson.utils.deprecated_utils``, not new behavior.
"""

import torch as th

import omnigibson.lazy as lazy
from omnigibson.physics_backends.base import PhysicsBackend
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.usd_utils import ensure_usd_api, triangularize_mesh

# These classes subclass Isaac Core classes, which trigger eager loading of Omni modules (carb/omni/isaacsim)
# that only exist on sys.path once the Kit app has been launched. So, like the rest of the codebase, we build
# them lazily on first use instead of importing/subclassing at module load time.
_PhysXArticulationViewCls = None


def _get_physx_articulation_view_cls():
    global _PhysXArticulationViewCls
    if _PhysXArticulationViewCls is None:
        from omnigibson.utils.deprecated_utils import ArticulationView as _ArticulationView

        class PhysXArticulationView(_ArticulationView):
            """
            Thin extension of the (already Isaac-bug-patched) Isaac Core ``Articulation`` view that exposes
            its DOF-layout metadata (joint count/names/dof-counts/dof-offsets, and dof-path lookup) as public
            methods, instead of requiring callers to reach into private ``_metadata``/``_dof_paths``
            attributes.
            """

            @property
            def has_dof_metadata(self):
                # False until the view is initialized against a live physics scene, in which case the
                # DOF-layout accessors below are unusable and callers must fall back to inspecting USD.
                return self._metadata is not None

            @property
            def joint_count(self):
                return self._metadata.joint_count

            @property
            def joint_dof_counts(self):
                return self._metadata.joint_dof_counts

            @property
            def joint_names(self):
                return self._metadata.joint_names

            @property
            def joint_dof_offsets(self):
                return self._metadata.joint_dof_offsets

            def get_dof_is_rotational(self):
                """
                Per-DOF rotational/translational classification -- True for rotational, False for
                translational. Collapses PhysX's own `omni.physics.tensors.DofType` enum (which is
                not importable at all without a live Kit application) so that it never escapes this
                module; `get_dof_types()` itself keeps returning the raw enum, since the Isaac Core
                base class consumes it internally.
                """
                return [x == lazy.omni.physics.tensors.DofType.Rotation for x in self.get_dof_types()]

            def dof_path(self, dof_index, articulation_index=0):
                return self._dof_paths[articulation_index][dof_index]

            def dof_index_of_path(self, prim_path, articulation_index=0):
                return list(self._dof_paths[articulation_index]).index(prim_path)

        _PhysXArticulationViewCls = PhysXArticulationView
    return _PhysXArticulationViewCls


class PhysXBackend(PhysicsBackend):
    def __init__(self, sim=None):
        super().__init__(sim)
        self._sim_context = None

    # ---- Lifecycle ----

    @classmethod
    def create_app(cls):
        # PhysX only exists inside a live Kit application, so it is always the one that launches it.
        from omnigibson.render_backends.kit_backend import launch_kit_app

        return launch_kit_app()

    def assert_clean_start(self):
        assert (
            lazy.isaacsim.core.utils.stage.get_current_stage() is None
        ), "Stage should not exist when creating a new Simulator instance"

    def before_clear(self):
        # Stop the viewport menubar USD watcher before teardown. This revokes the TfNotice listener so
        # that prim deletions during _partial_clear() don't queue deferred callbacks that later fire on
        # an invalid stage and corrupt CUDA/PhysX state.
        lazy.omni.kit.viewport.menubar.core.utils.usd_watch.stop()

    def after_clear(self):
        lazy.omni.kit.viewport.menubar.core.utils.usd_watch.start()

        # Then close the stage, so a new one can be created for the replacement Simulator.
        assert lazy.isaacsim.core.utils.stage.close_stage()

    def finalize(self):
        lazy.isaacsim.core.api.SimulationContext.clear_instance()

    def in_warmup(self):
        return lazy.isaacsim.core.simulation_manager.SimulationManager._warmup_needed

    def add_ground_plane(self, prim_path, visible=True, color=None):
        with self.sim.editing_usd():
            plane = lazy.isaacsim.core.api.objects.ground_plane.GroundPlane(
                prim_path=prim_path,
                name="ground_plane",
                z_position=0,
                size=None,
                color=None if color is None else th.tensor(color),
                visible=visible,
                # TODO: update with new PhysicsMaterial API
                # static_friction=static_friction,
                # dynamic_friction=dynamic_friction,
                # restitution=restitution,
            )

        triangularize_mesh(lazy.pxr.UsdGeom.Mesh.Define(self.sim.stage, plane.prim.GetChildren()[0].GetPath()))

    # ---- Batched DOF target writes ----
    #
    # @view here is an `omni.physics.tensors` articulation view, whose frontend/backend split is
    # PhysX-specific. Writing through it directly skips the Isaac Core wrapper.

    def set_dof_position_targets(self, view, data, indices, cast=False):
        self._write_dof_targets(view, view._backend.set_dof_position_targets, data, indices, cast, "positions")

    def set_dof_velocity_targets(self, view, data, indices, cast=False):
        self._write_dof_targets(view, view._backend.set_dof_velocity_targets, data, indices, cast, "velocities")

    def set_dof_actuation_forces(self, view, data, indices, cast=False):
        self._write_dof_targets(view, view._backend.set_dof_actuation_forces, data, indices, cast, "actuation forces")

    @staticmethod
    def _write_dof_targets(view, backend_fn, data, indices, cast, what):
        # No casting results in better efficiency
        if cast:
            data = view._frontend.as_contiguous_float32(data)
            indices = view._frontend.as_contiguous_uint32(indices)
        data_desc = view._frontend.get_tensor_desc(data)
        indices_desc = view._frontend.get_tensor_desc(indices)

        if not backend_fn(data_desc, indices_desc):
            raise Exception(f"Failed to set DOF {what} in backend")

    # ---- Prim transform reads ----
    #
    # PhysX keeps Fabric current as a side effect of its own stepping, so Fabric is both cheaper and
    # more up to date here than raw USD.

    def get_world_transform_with_scale(self, prim_path):
        # Check that no reads from Fabric are happening during a physics step.
        assert (
            not self.sim.currently_stepping
        ), "Do not read poses from Fabric during a physics step, this is quite slow!"

        return self.sim.fabric_hierarchy.get_world_xform(lazy.usdrt.Sdf.Path(prim_path))

    def get_local_transform_with_scale(self, prim_path):
        assert (
            not self.sim.currently_stepping
        ), "Do not read poses from Fabric during a physics step, this is quite slow!"

        return self.sim.fabric_hierarchy.get_local_xform(lazy.usdrt.Sdf.Path(prim_path))

    # ---- Engine-specific USD schema authoring ----
    #
    # `PhysxSchema` is a PhysX-specific USD schema extension, unavailable in a plain `pxr` install.

    def apply_rigid_body_schemas(self, prim):
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxRigidBodyAPI)

    def apply_collision_schemas(self, prim):
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxCollisionAPI)

    def configure_collision_approximation(self, prim, approximation_type):
        if approximation_type == "convexHull":
            api = ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI)
            # Also make sure the maximum vertex count is 60 (max number compatible with GPU)
            # https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html#collision-settings
            with self.sim.editing_usd():
                if api.GetHullVertexLimitAttr().Get() is None:
                    api.CreateHullVertexLimitAttr()
                api.GetHullVertexLimitAttr().Set(60)
        elif approximation_type == "convexDecomposition":
            ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI)
        elif approximation_type == "meshSimplification":
            ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI)
        elif approximation_type == "sdf":
            ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI)
        elif approximation_type == "none":
            ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI)

    def apply_joint_schemas(self, prim):
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxJointAPI)

    def apply_articulation_schemas(self, prim, self_collisions=False):
        lazy.pxr.PhysxSchema.PhysxArticulationAPI.Apply(prim)
        prim.GetAttribute("physxArticulation:enabledSelfCollisions").Set(bool(self_collisions))

    def strip_articulation_schemas(self, prim):
        prim.RemoveAPI(lazy.pxr.PhysxSchema.PhysxArticulationAPI)

    def enable_contact_reporting(self, prim):
        contact_api = ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxContactReportAPI)
        with self.sim.editing_usd():
            contact_api.GetThresholdAttr().Set(0.0)

    def contact_reporting_enabled(self, prim, visual_only):
        return prim.HasAPI(lazy.pxr.PhysxSchema.PhysxContactReportAPI)

    def get_collision_api(self, prim):
        return (
            lazy.pxr.PhysxSchema.PhysxCollisionAPI(prim)
            if prim.HasAPI(lazy.pxr.PhysxSchema.PhysxCollisionAPI)
            else None
        )

    def is_mimic_joint(self, prim):
        return prim.HasAPI(lazy.pxr.PhysxSchema.PhysxMimicJointAPI)

    def create_physics_context(self, physics_dt, rendering_dt, device):
        self._sim_context = lazy.isaacsim.core.api.SimulationContext(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            backend="torch",
            device=device,
        )
        return self._sim_context

    def start_step_callbacks(self, pre_step_fn, post_step_fn, joint_break_fn):
        self._pre_physics_step_callback = lazy.omni.physx.get_physx_interface().subscribe_physics_on_step_events(
            lambda _: pre_step_fn(),
            pre_step=True,
            order=0,
        )
        self._post_physics_step_callback = lazy.omni.physx.get_physx_interface().subscribe_physics_on_step_events(
            lambda _: post_step_fn(),
            pre_step=False,
            order=0,
        )
        self._simulation_event_callback = (
            lazy.omni.physx.get_physx_interface()
            .get_simulation_event_stream_v2()
            .create_subscription_to_pop(joint_break_fn)
        )

    def stop_step_callbacks(self):
        for attr in ("_pre_physics_step_callback", "_post_physics_step_callback", "_simulation_event_callback"):
            subscription = getattr(self, attr, None)
            if subscription is not None:
                subscription.unsubscribe()
                setattr(self, attr, None)

    def apply_engine_settings(
        self,
        gravity,
        enable_ccd,
        use_gpu_dynamics,
        gpu_pairs_capacity,
        gpu_aggr_pairs_capacity,
        gpu_max_particle_contacts,
        gpu_max_rigid_contact_count,
        gpu_max_rigid_patch_count,
    ):
        physics_context = self._physics_context
        physics_context.set_gravity(value=-gravity)
        # Also make sure we don't invert the collision group filter settings so that different collision groups by
        # default collide with each other, and modify settings for speed optimization
        physics_context.set_invert_collision_group_filter(False)
        physics_context.enable_ccd(enable_ccd)
        physics_context.enable_fabric(True)

        # Enable GPU dynamics based on whether we need omni particles feature
        if use_gpu_dynamics:
            physics_context.enable_gpu_dynamics(True)
            physics_context.set_broadphase_type("GPU")
        else:
            physics_context.enable_gpu_dynamics(False)
            physics_context.set_broadphase_type("MBP")

        # Set GPU Pairs capacity and other GPU settings
        physics_context.set_gpu_found_lost_pairs_capacity(gpu_pairs_capacity)
        physics_context.set_gpu_found_lost_aggregate_pairs_capacity(gpu_aggr_pairs_capacity)
        physics_context.set_gpu_total_aggregate_pairs_capacity(gpu_aggr_pairs_capacity)
        physics_context.set_gpu_max_particle_contacts(gpu_max_particle_contacts)
        physics_context.set_gpu_max_rigid_contact_count(gpu_max_rigid_contact_count)
        physics_context.set_gpu_max_rigid_patch_count(gpu_max_rigid_patch_count)

    def get_physics_context(self):
        return self._sim_context.get_physics_context()

    @property
    def _physics_context(self):
        return self._sim_context._physics_context

    @property
    def stage(self):
        return self._sim_context.stage

    @property
    def current_time(self):
        return self._sim_context.current_time

    @property
    def current_time_step_index(self):
        return self._sim_context.current_time_step_index

    @property
    def initial_physics_dt(self):
        return self._sim_context._initial_physics_dt

    @property
    def initial_rendering_dt(self):
        return self._sim_context._initial_rendering_dt

    def is_playing(self):
        return self._sim_context.is_playing()

    def is_stopped(self):
        return self._sim_context.is_stopped()

    def get_physics_dt(self):
        return self._sim_context.get_physics_dt()

    def get_rendering_dt(self):
        return self._sim_context.get_rendering_dt()

    def set_simulation_dt(self, physics_dt=None, rendering_dt=None):
        self._sim_context.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)

    def get_device(self):
        return lazy.isaacsim.core.simulation_manager.SimulationManager.get_physics_sim_device()

    def set_device(self, device):
        lazy.isaacsim.core.simulation_manager.SimulationManager.set_physics_sim_device(device)

    def play(self):
        self._sim_context.play()

    def pause(self):
        self._sim_context.pause()

    def stop(self):
        self._sim_context.stop()

    def render(self):
        self._sim_context.render()

    def step(self, render):
        self._sim_context.step(render=render)

    def step_physics_once(self, current_time):
        self._physics_context._step(current_time=current_time)

    @property
    def physics_sim_view(self):
        return self._sim_context.physics_sim_view

    def refresh_physics_sim_view(self):
        SimulationManager = lazy.isaacsim.core.simulation_manager.SimulationManager
        IsaacEvents = lazy.isaacsim.core.simulation_manager.IsaacEvents

        stage_id = lazy.isaacsim.core.utils.stage.get_current_stage_id()
        SimulationManager._physics_sim_view = lazy.omni.physics.tensors.create_simulation_view(
            SimulationManager._backend, stage_id=stage_id
        )
        SimulationManager._physics_sim_view.set_subspace_roots("/")
        SimulationManager._physics_sim_view__warp = lazy.omni.physics.tensors.create_simulation_view(
            "warp", stage_id=stage_id
        )
        SimulationManager._simulation_view_created = True
        SimulationManager._message_bus.dispatch_event(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
        SimulationManager._message_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})

    def invalidate_physics_sim_view(self):
        SimulationManager = lazy.isaacsim.core.simulation_manager.SimulationManager
        if SimulationManager._physics_sim_view:
            SimulationManager._physics_sim_view.invalidate()
            SimulationManager._physics_sim_view = None

    def sync_to_render_layer(self):
        self._sim_context._physx_fabric_interface.update(self.current_time, self.get_physics_dt())

    def flush_changes(self):
        self._psi.flush_changes()

    # ---- Per-prim articulation / rigid-body view I/O ----

    def create_articulation_view(self, prim_path):
        return _get_physx_articulation_view_cls()(prim_path)

    def create_rigid_body_view(self, prim_path):
        from omnigibson.utils.deprecated_utils import RigidPrimView as _RigidPrimView

        return _RigidPrimView(prim_path, reset_xform_properties=False)

    # ---- Sleep / wake ----

    def is_asleep(self, prim_path):
        return self._psi.is_sleeping(self.sim.stage_id, lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim_path))

    def wake(self, prim_path):
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim_path)
        self._psi.wake_up(self.sim.stage_id, prim_id)

    def sleep(self, prim_path):
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim_path)
        self._psi.put_to_sleep(self.sim.stage_id, prim_id)

    def apply_force_at_pos(self, prim_path, force, pos):
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim_path)
        self._psi.apply_force_at_pos(self.sim.stage_id, prim_id, force, pos)

    def apply_torque(self, prim_path, torque):
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(prim_path)
        self._psi.apply_torque(self.sim.stage_id, prim_id, torque)

    @property
    def _psi(self):
        return lazy.omni.physx.get_physx_simulation_interface()

    # ---- Scene queries / raycasts ----

    @property
    def _psqi(self):
        return lazy.omni.physx.get_physx_scene_query_interface()

    def raycast_closest(self, *args, **kwargs):
        return self._psqi.raycast_closest(*args, **kwargs)

    def raycast_all(self, *args, **kwargs):
        return self._psqi.raycast_all(*args, **kwargs)

    def overlap_sphere(self, *args, **kwargs):
        return self._psqi.overlap_sphere(*args, **kwargs)

    def overlap_box(self, *args, **kwargs):
        return self._psqi.overlap_box(*args, **kwargs)

    def overlap_mesh(self, *args, **kwargs):
        return self._psqi.overlap_mesh(*args, **kwargs)

    def overlap_shape(self, *args, **kwargs):
        return self._psqi.overlap_shape(*args, **kwargs)

    def overlap_sphere_any(self, radius, pos):
        return self._psqi.overlap_sphere_any(radius, pos)

    # ---- Cloth particle state I/O ----
    # points/velocities are raw USD attrs on the cloth mesh prim -- PhysX's own particle-cloth solver
    # keeps them in sync with its internal simulation state each step via Fabric/USD live-sync (Kit-only
    # behavior; a backend without an equivalent has to instead read/
    # write its own solver state directly). points are authored in the mesh's LOCAL frame; velocities
    # are already world-frame (PhysX's own convention). This class exposes only the WORLD-frame contract
    # documented in PhysicsBackend, so local<->world conversion happens here, not in ClothPrim.

    def _cloth_local_to_world_matrix(self, prim_path):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        matrix = lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        # Gf.Matrix4d is row-vector convention (v' = v * M); rows 0-2 are the linear part, row 3 is
        # translation.
        linear = th.tensor([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=th.float32)
        translation = th.tensor([matrix[3][j] for j in range(3)], dtype=th.float32)
        return linear, translation

    def get_cloth_particle_positions(self, prim_path, idxs=None):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        p_local = vtarray_to_torch(prim.GetAttribute("points").Get())
        p_local = p_local[idxs] if idxs is not None else p_local
        linear, translation = self._cloth_local_to_world_matrix(prim_path)
        return p_local @ linear + translation

    def set_cloth_particle_positions(self, prim_path, positions, idxs=None):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        linear, translation = self._cloth_local_to_world_matrix(prim_path)
        p_local = (th.as_tensor(positions, dtype=th.float32, device=translation.device) - translation) @ th.linalg.inv(
            linear
        )
        if idxs is not None:
            p_local_full = vtarray_to_torch(prim.GetAttribute("points").Get())
            p_local_full[idxs] = p_local
            p_local = p_local_full
        with self.sim.editing_usd():
            prim.GetAttribute("points").Set(lazy.pxr.Vt.Vec3fArray(p_local.tolist()))

    def get_cloth_particle_velocities(self, prim_path):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        return vtarray_to_torch(prim.GetAttribute("velocities").Get())

    def set_cloth_particle_velocities(self, prim_path, velocities):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        velocities = th.as_tensor(velocities, dtype=th.float32)
        with self.sim.editing_usd():
            prim.GetAttribute("velocities").Set(lazy.pxr.Vt.Vec3fArray(velocities.tolist()))

    _CLOTH_STIFFNESS_ATTRS = {
        "bend": "physxAutoParticleCloth:springBendStiffness",
        "damping": "physxAutoParticleCloth:springDamping",
        "shear": "physxAutoParticleCloth:springShearStiffness",
        "stretch": "physxAutoParticleCloth:springStretchStiffness",
    }

    def get_cloth_stiffness(self, prim_path):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        return {key: prim.GetAttribute(attr).Get() for key, attr in self._CLOTH_STIFFNESS_ATTRS.items()}

    def set_cloth_stiffness(self, prim_path, bend=None, damping=None, shear=None, stretch=None):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        values = {"bend": bend, "damping": damping, "shear": shear, "stretch": stretch}
        # One editing_usd() block for all four, rather than per-attribute: nesting is forbidden and
        # each entry into the context costs a USD->Fabric sync.
        with self.sim.editing_usd():
            for key, value in values.items():
                if value is not None:
                    prim.GetAttribute(self._CLOTH_STIFFNESS_ATTRS[key]).Set(value)

    # ---- Joint-break events ----

    def is_joint_break_event(self, event):
        return event.type == int(lazy.omni.physx.bindings._physx.SimulationEvent.JOINT_BREAK)

    def decode_joint_break_event(self, event):
        return str(
            lazy.pxr.PhysicsSchemaTools.decodeSdfPath(event.payload["jointPath"][0], event.payload["jointPath"][1])
        )
