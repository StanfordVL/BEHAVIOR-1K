"""
Abstract interface for a pluggable physics-engine backend.

The Simulator constructs exactly one PhysicsBackend instance (exposed as ``og.sim.physics_backend``),
chosen via ``gm.PHYSICS_BACKEND``. All physics-engine-specific behavior is confined behind this
interface -- ``Simulator`` and the ``prims`` / ``utils`` layers above it talk only to
``og.sim.physics_backend``, never to a concrete engine module directly.

Rendering (the Kit/RTX render loop, cameras, materials, lighting) is out of scope for this abstraction
and lives instead in the sibling ``render_backends.RenderBackend`` interface (``og.sim.render_backend``,
chosen via ``gm.RENDER_BACKEND``). The two are independent axes -- a given run picks a physics backend
and a render backend separately, and ``Simulator._launch_app()`` launches whichever application a
backend hands back from ``create_app()``. ``PhysicsBackend.sync_to_render_layer()`` below remains
the one physics-side hook that
crosses this boundary (pushing computed poses toward wherever the render backend reads them from), since
that's fundamentally the physics engine's own responsibility, not the renderer's.
"""

from abc import ABC, abstractmethod


class PhysicsBackend(ABC):
    """
    Abstract base class for a physics engine backend.
    """

    def __init__(self, sim=None):
        self.sim = sim

    # ---- Application lifecycle ----

    @classmethod
    def create_app(cls):
        """
        Creates and returns the application this backend needs to run at all. Called on the class,
        before any Simulator or backend instance exists, from `Simulator._launch_app()`; making it
        the backend's job is what keeps the decision to launch Kit out of `Simulator` itself.

        Returns:
            isaacsim.SimulationApp: The application this backend runs inside.
        """
        raise NotImplementedError

    def assert_clean_start(self):
        """
        Raises if process-global state left over from a previous Simulator would break this backend.
        Called at the very top of `Simulator.__init__`. Default no-op.
        """

    def before_clear(self):
        """
        Hook called by `og.clear()` immediately before the Simulator tears down everything it owns,
        for state that must be quiesced before prims start disappearing. Default no-op.
        """

    def after_clear(self):
        """
        Hook called by `og.clear()` immediately after `Simulator._partial_clear()`, to undo
        `before_clear()` and release whatever engine-global state would otherwise leak into the
        replacement Simulator. Default no-op.
        """

    def finalize(self):
        """
        Hook called by `og.clear()` after `og.sim` has already been dropped, for engine-global
        teardown that must outlive the Simulator it belonged to. Default no-op.
        """

    def in_warmup(self):
        """
        Whether the engine is currently inside its own internal warmup stepping, during which
        OmniGibson's per-step controller/view bookkeeping must be skipped (those buffers are not yet
        valid and get rebuilt right afterwards).

        Returns:
            bool: True if warming up. Default False -- a backend with no such concept always steps
                for real.
        """
        return False

    @abstractmethod
    def add_ground_plane(self, prim_path, visible=True, color=None):
        """
        Creates this backend's ground-plane collider at @prim_path.

        Args:
            prim_path (str): Absolute prim path to create the ground plane at.
            visible (bool): Whether the plane should be visible to a renderer.
            color (None or 3-array): Optional RGB color.
        """

    def enable_extensions(self):
        """
        One-time, process-level setup performed once before any Simulator/Kit app exists (called from
        `Simulator._launch_app()` on a transient, sim-less backend instance). Default no-op; a backend
        that needs to set process-global state before its own package is imported (e.g. an env var that
        must be set before `pxr` is first imported) overrides this.
        """

    def before_play(self, sim):
        """
        Hook called every time `Simulator.play()` transitions from stopped to playing (including the
        one-off warmup play/stop cycle `Simulator.__init__` itself does before any objects exist).
        Default no-op; a backend that needs to (re)build its engine-side scene representation from the
        live USD stage before physics can run overrides this -- such a backend should rebuild from
        scratch on every call rather than caching across calls, since the object/robot set may have
        changed while stopped.
        """

    # ---- Lifecycle ----

    @abstractmethod
    def create_physics_context(self, physics_dt, rendering_dt, device):
        """Construct the sim-lifecycle context this backend uses for play/pause/stop/current_time bookkeeping."""

    @abstractmethod
    def stop_step_callbacks(self):
        """
        Tears down the subscriptions made by start_step_callbacks(). Must be called before the owning
        Simulator goes away: a live subscription keeps firing into the old Simulator's callbacks, so
        skipping this leaves the next Simulator with two of everything.
        """

    @abstractmethod
    def start_step_callbacks(self, pre_step_fn, post_step_fn, joint_break_fn):
        """
        Register the given callables to be invoked right before / after each physics step, and whenever
        a simulation event occurs (``joint_break_fn`` is called with the raw backend event; use
        ``is_joint_break_event``/``decode_joint_break_event`` to interpret it).
        """

    @abstractmethod
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
        """Apply global engine settings (gravity, CCD, GPU dynamics, buffer capacities)."""

    @abstractmethod
    def get_physics_context(self):
        """Return the low-level physics scene context object (used by particle/cloth authoring code)."""

    @property
    @abstractmethod
    def stage(self):
        """The USD stage backing this simulation."""

    @property
    @abstractmethod
    def current_time(self):
        pass

    @property
    @abstractmethod
    def current_time_step_index(self):
        pass

    @property
    @abstractmethod
    def initial_physics_dt(self):
        pass

    @property
    @abstractmethod
    def initial_rendering_dt(self):
        pass

    @abstractmethod
    def is_playing(self):
        pass

    @abstractmethod
    def is_stopped(self):
        pass

    @abstractmethod
    def get_physics_dt(self):
        pass

    @abstractmethod
    def get_rendering_dt(self):
        pass

    @abstractmethod
    def set_simulation_dt(self, physics_dt=None, rendering_dt=None):
        pass

    @abstractmethod
    def get_device(self):
        pass

    @abstractmethod
    def set_device(self, device):
        pass

    @abstractmethod
    def play(self):
        pass

    @abstractmethod
    def pause(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self, render):
        """Advance the underlying sim context/renderer by one step."""

    @abstractmethod
    def step_physics_once(self, current_time):
        """Advance physics only (no rendering), used for manual single-step physics."""

    @property
    @abstractmethod
    def physics_sim_view(self):
        """
        The engine's batch-view factory object, exposing ``create_articulation_view(pattern)``,
        ``create_rigid_contact_view(pattern, filter_patterns, max_contact_data_count)``, and
        ``create_rigid_body_view(pattern)`` for pattern-based, scene-wide batch views (as opposed to the
        single-prim views returned by ``create_articulation_view``/``create_rigid_body_view`` below).
        """

    @abstractmethod
    def refresh_physics_sim_view(self):
        """Rebuild/invalidate the batch physics views after the set of prims in the stage has changed."""

    @abstractmethod
    def invalidate_physics_sim_view(self):
        """
        Preemptively invalidate/de-initialize the batch physics views, e.g. before an operation that is
        known to invalidate them as a side effect (such as adding new objects to the stage). A no-op if
        no view currently exists.
        """

    @abstractmethod
    def flush_changes(self):
        """Flush any pending USD changes into the physics engine's own simulation state."""

    @abstractmethod
    def sync_to_render_layer(self):
        """
        Push physics-side pose changes into whatever render-facing scene mirror the renderer uses
        (e.g. Fabric). A backend with no such mirror may no-op this.
        """

    # ---- Batched DOF target writes ----
    #
    # @view is one of this backend's own articulation views. These take it explicitly because the
    # caller (`BatchControlViewAPIImpl`) holds one view spanning every controllable object rather
    # than one per prim.

    @abstractmethod
    def set_dof_position_targets(self, view, data, indices, cast=False):
        """Writes DOF position targets @data at @indices through @view."""

    @abstractmethod
    def set_dof_velocity_targets(self, view, data, indices, cast=False):
        """Writes DOF velocity targets @data at @indices through @view."""

    @abstractmethod
    def set_dof_actuation_forces(self, view, data, indices, cast=False):
        """Writes DOF actuation forces @data at @indices through @view."""

    # ---- Prim transform reads ----

    @abstractmethod
    def get_world_transform_with_scale(self, prim_path):
        """
        Returns:
            Gf.Matrix4d: @prim_path's world transform, including scale, read from wherever this
                backend keeps live poses.
        """

    @abstractmethod
    def get_local_transform_with_scale(self, prim_path):
        """
        Returns:
            Gf.Matrix4d: @prim_path's local (parent-relative) transform, including scale.
        """

    # ---- Engine-specific USD schema authoring ----
    #
    # OmniGibson authors the generic `UsdPhysics.*` schemas itself -- those are engine-agnostic and
    # every backend understands them. These hooks are for whatever an engine needs *on top* of that
    # (PhysX's `PhysxSchema.*` extensions, which do not even exist in a plain `pxr` install), and all
    # default to doing nothing.

    def apply_rigid_body_schemas(self, prim):
        """Applies any engine-specific rigid-body schema to @prim. Default no-op."""

    def apply_collision_schemas(self, prim):
        """Applies any engine-specific collider schema to @prim. Default no-op."""

    def configure_collision_approximation(self, prim, approximation_type):
        """
        Applies the engine-specific schema and tuning for @approximation_type (one of "convexHull",
        "convexDecomposition", "meshSimplification", "sdf", "none", ...) on @prim. The backend-agnostic
        `UsdPhysics.MeshCollisionAPI` approximation attribute is recorded by the caller either way.
        Default no-op.
        """

    def apply_joint_schemas(self, prim):
        """Applies any engine-specific joint schema to @prim. Default no-op."""

    def apply_articulation_schemas(self, prim, self_collisions=False):
        """
        Applies any engine-specific articulation-root schema to @prim, configured for
        @self_collisions. Default no-op.
        """

    def strip_articulation_schemas(self, prim):
        """
        Removes any engine-specific articulation-root schema from @prim (the generic
        `UsdPhysics.ArticulationRootAPI` is removed by the caller). Default no-op.
        """

    def enable_contact_reporting(self, prim):
        """
        Opts @prim into per-body contact reporting, for an engine that needs it requested per body.
        Default no-op -- an engine that reports contacts for every body regardless has nothing to
        opt into.
        """

    def contact_reporting_enabled(self, prim, visual_only):
        """
        Returns:
            bool: Whether contact reporting is active for @prim. Defaults to "every
                non-visual-only body", matching an engine with no per-body opt-in (see
                enable_contact_reporting()).
        """
        return not visual_only

    def get_collision_api(self, prim):
        """
        Returns:
            None or object: @prim's engine-specific collider API handle, whose attributes the caller
                writes collider settings through, or None if this engine has no such API (in which
                case those settings have no engine-side equivalent and are skipped). Default None.
        """
        return None

    def is_mimic_joint(self, prim):
        """
        Returns:
            bool: Whether @prim is a mimic joint (one whose motion is slaved to another joint's).
                Default False -- no engine-agnostic equivalent exists for this.
        """
        return False

    # ---- Per-prim articulation view I/O (backs EntityPrim / JointPrim) ----

    @abstractmethod
    def create_articulation_view(self, prim_path):
        """
        Return a handle for the articulation rooted at ``prim_path``, exposing (at least): joint
        position/velocity/effort get+set, position/velocity targets, coriolis/gravity/mass-matrix/
        jacobian, world-pose set, solver iteration counts, and the public ``joint_count``/
        ``joint_dof_counts``/``joint_names``/``joint_dof_offsets``/``dof_path``/``dof_index_of_path``
        DOF-layout accessors, plus ``has_dof_metadata`` reporting whether those accessors are usable
        yet (deliberately public -- no caller should need to reach into backend-private metadata).
        """

    # ---- Per-prim rigid-body view I/O (backs RigidDynamicPrim) ----

    @abstractmethod
    def create_rigid_body_view(self, prim_path):
        """
        Return a handle for the rigid body at ``prim_path``, exposing (at least): linear/angular
        velocity get+set, world-pose get+set, center-of-mass/mass/density get+set, and gravity
        enable/disable.
        """

    # ---- Sleep / wake ----

    @abstractmethod
    def is_asleep(self, prim_path):
        pass

    @abstractmethod
    def wake(self, prim_path):
        pass

    @abstractmethod
    def sleep(self, prim_path):
        pass

    @abstractmethod
    def apply_force_at_pos(self, prim_path, force, pos):
        pass

    @abstractmethod
    def apply_torque(self, prim_path, torque):
        pass

    # ---- Scene queries / raycasts ----
    # These mirror the underlying scene-query interface's own signatures 1:1 (pass-through), since they
    # are one-for-one replacements for existing direct call sites.

    @abstractmethod
    def raycast_closest(self, *args, **kwargs):
        pass

    def raycast_closest_batch(self, origins, dirs, distances):
        """
        Closest hit for a batch of rays. Optional fast path: this default implementation simply calls
        raycast_closest() once per ray, so a backend only needs to override it when issuing a single
        query for the whole batch is cheaper than N individual ones -- see
        a backend whose raycast_closest_batch() pays a large fixed
        cost independent of how many rays it carries.

        Args:
            origins (list of 3-array): per-ray (x,y,z) world-frame ray origin
            dirs (list of 3-array): per-ray unit-length world-frame ray direction
            distances (list of float): per-ray maximum hit distance

        Returns:
            list of dict: one raycast_closest() result per input ray, in the same order
        """
        return [
            self.raycast_closest(origin=origin, dir=dir, distance=distance)
            for origin, dir, distance in zip(origins, dirs, distances)
        ]

    @abstractmethod
    def raycast_all(self, *args, **kwargs):
        pass

    @abstractmethod
    def overlap_sphere(self, *args, **kwargs):
        pass

    @abstractmethod
    def overlap_box(self, *args, **kwargs):
        pass

    @abstractmethod
    def overlap_mesh(self, *args, **kwargs):
        pass

    @abstractmethod
    def overlap_shape(self, *args, **kwargs):
        pass

    @abstractmethod
    def overlap_sphere_any(self, radius, pos):
        pass

    # ---- Cloth particle state I/O (backs ClothPrim) ----
    # Positions/velocities are always in the WORLD frame -- each backend hides its own internal
    # convention (e.g. a local-mesh-frame USD attribute vs. an already-world-space solver state array).

    @abstractmethod
    def get_cloth_particle_positions(self, prim_path, idxs=None):
        """Return an (N, 3) (or (len(idxs), 3) if idxs given) world-frame particle position tensor."""

    @abstractmethod
    def set_cloth_particle_positions(self, prim_path, positions, idxs=None):
        """Set world-frame particle positions (shape must match get_cloth_particle_positions)."""

    @abstractmethod
    def get_cloth_particle_velocities(self, prim_path):
        """Return an (N, 3) world-frame particle velocity tensor."""

    @abstractmethod
    def set_cloth_particle_velocities(self, prim_path, velocities):
        """Set world-frame particle velocities (shape must match get_cloth_particle_velocities)."""

    @abstractmethod
    def get_cloth_stiffness(self, prim_path):
        """Return a dict with 'bend'/'damping'/'shear'/'stretch' stiffness values for this cloth."""

    @abstractmethod
    def set_cloth_stiffness(self, prim_path, bend=None, damping=None, shear=None, stretch=None):
        """Set any subset of this cloth's bend/damping/shear/stretch stiffness values (None = unchanged)."""

    # ---- Joint-break events ----

    @abstractmethod
    def is_joint_break_event(self, event):
        pass

    @abstractmethod
    def decode_joint_break_event(self, event):
        """Return the broken joint's prim path as a string, given a raw backend event."""
