"""
Abstract interface for a pluggable render backend.

The Simulator constructs exactly one RenderBackend instance (exposed as ``og.sim.render_backend``),
chosen via ``gm.RENDER_BACKEND``. This is the sibling abstraction to ``physics_backends.PhysicsBackend``
(see that module's docstring) -- where PhysicsBackend owns everything about advancing physics state,
RenderBackend owns everything about turning that state into pixels (or, in Phase 1, simply owns whether
a live Kit/Omniverse application needs to exist at all).

Physics and rendering are independent axes: ``gm.PHYSICS_BACKEND`` picks how physics is simulated,
``gm.RENDER_BACKEND`` picks how (or whether) it gets rendered, and either backend may independently
require a live Kit application -- which it expresses by returning one from ``create_app()`` rather
than by advertising a flag for callers to branch on (see ``simulator.py::_launch_app()``).

The interface covers application and Simulator-setup lifecycle, the render-graph "tick" primitive,
and camera/modality capture. Viewport/GUI management and material authoring are still only partly
relocated here; the remaining Kit-specific call sites are being moved behind this interface
incrementally.
"""

from abc import ABC, abstractmethod


class RenderBackend(ABC):
    """
    Abstract base class for a render backend.
    """

    def __init__(self, sim=None):
        self.sim = sim

    # ---- Application lifecycle ----

    def enable_extensions(self):
        """
        One-time, process-level setup performed once before any Simulator/Kit app exists (called from
        `Simulator._launch_app()` on a transient, sim-less backend instance, mirroring
        `PhysicsBackend.enable_extensions()`). Default no-op.
        """

    def before_play(self, sim):
        """
        Hook called every time `Simulator.play()` transitions from stopped to playing (including the
        one-off warmup play/stop cycle `Simulator.__init__` itself does before any objects exist),
        mirroring `PhysicsBackend.before_play(sim)`. Default no-op.
        """

    # ---- Simulator setup hooks ----

    def apply_renderer_settings(self):
        """
        Applies this backend's renderer/lighting settings. Called once from `Simulator.__init__`.
        Default no-op.
        """

    def create_fabric_hierarchy(self):
        """
        Creates the Fabric/usdrt stage mirror and hierarchy interface this backend reads poses
        through, if it has one.

        Returns:
            2-tuple: (usdrt_stage, fabric_hierarchy), both None if this backend has no Fabric mirror
                (pose reads then go through raw pxr / the physics backend's own state instead).
        """
        return None, None

    def setup_viewer_camera(self, viewer_width, viewer_height):
        """
        Creates the interactive viewer camera at the given resolution, if this backend has a viewport.
        Called from `Simulator.__init__` when `gm.RENDER_VIEWER_CAMERA` is set. Default no-op.
        """

    # ---- Stage / application services ----
    #
    # A live rendering application brings its own versions of a handful of USD, bounds, semantics and
    # logging utilities. They live on the render backend rather than the physics backend because what
    # picks between implementations is which application is drawing, not which engine is simulating.

    @abstractmethod
    def create_default_pbr_material(self, scope_path, material_path, target_prim_path):
        """
        Creates a default PBR material at @material_path (under a Scope authored at @scope_path) and
        binds it to @target_prim_path.
        """

    @abstractmethod
    def get_stage(self):
        """
        Returns:
            pxr.Usd.Stage: The stage this backend renders from.
        """

    @abstractmethod
    def copy_prim(self, source_prim_path, dest_prim_path):
        """Copies the prim at @source_prim_path, and its whole subtree, to @dest_prim_path."""

    @abstractmethod
    def copy_mesh_prim(self, source_prim_path, dest_prim_path):
        """
        Copies the single, childless Mesh prim at @source_prim_path to @dest_prim_path. Narrower than
        copy_prim(); the caller guarantees there is no subtree to carry along.
        """

    @abstractmethod
    def deactivate_prim(self, prim):
        """Removes @prim from the scene as far as anything reading the stage is concerned."""

    @abstractmethod
    def delete_prim(self, prim_path, destructive=True):
        """
        Deletes the prim at @prim_path. @destructive=False deactivates it instead of removing it,
        which is all that is possible for a prim whose definition comes from a reference arc.
        """

    @abstractmethod
    def is_prim_ancestral(self, prim):
        """
        Returns:
            bool: Whether @prim's own definition was introduced via a reference arc (e.g. a link
                inside a dataset object's referenced USD), as opposed to being authored directly on
                the live stage. Such a prim can only be deactivated, never removed.
        """

    @abstractmethod
    def add_reference_to_stage(self, asset_path, prim_path):
        """
        References the USD at @asset_path in at @prim_path.

        Returns:
            Usd.Prim: The referencing prim.
        """

    @abstractmethod
    def create_primitive_mesh(self, primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
        """
        Authors a @primitive_type mesh (one of `PRIMITIVE_MESH_TYPES`) at @prim_path, optionally
        tessellated to @u_patches x @v_patches.
        """

    @abstractmethod
    def compute_world_aabb(self, prim_path):
        """
        Returns:
            2-tuple: (aabb_min, aabb_max) world-frame axis-aligned bounds of @prim_path, each a
                3-tuple of floats.
        """

    @abstractmethod
    def recompute_extents(self, prim):
        """Re-syncs @prim's authored extent attribute to its actual geometry."""

    @abstractmethod
    def add_semantic_labels(self, prim, label, instance_name="class"):
        """Tags @prim with semantic @label under @instance_name, for segmentation rendering."""

    @abstractmethod
    def suppress_log(self, channels=None):
        """
        Returns:
            contextmanager: Suppresses @channels (or all logging, if None) for its duration.
        """

    # ---- Render-graph lifecycle ----

    @abstractmethod
    def render(self):
        """
        Advance/tick the render graph by one frame (a Kit app-update + RTX draw, or equivalent). Does
        NOT itself return pixel data -- that is a later phase's `capture_modality()`-style method.
        Called by `Simulator.render()`/`step()` whenever the physics backend does not already own the
        Kit tick itself (e.g. PhysX's own `SimulationContext.render()` already does this, so this
        method is only invoked in addition by a physics backend that does not own that tick).
        """

    # ---- Camera capture pipeline (Phase 3) ----
    #
    # These back VisionSensor's capture pipeline (create a per-camera render resource, attach/detach
    # per-modality annotators to it, pull each annotator's latest data). Callers must guard every call

    @abstractmethod
    def create_camera_resource(self, prim_path, resolution, force_new=False):
        """
        Creates a backend-specific rendering resource (e.g. a Replicator render product) for the camera
        prim at @prim_path, sized to @resolution = (width, height). @force_new forces creation of a new
        resource even if the backend has a stale cached one at the same @prim_path (needed when
        recreating a resource right after destroying the previous one at the same path).

        Returns:
            object: Opaque handle to be passed to attach_modality/destroy_camera_resource.
        """

    @abstractmethod
    def destroy_camera_resource(self, camera_resource):
        """Destroys a camera resource previously returned by create_camera_resource."""

    @abstractmethod
    def create_annotator(self, raw_modality_name):
        """
        Creates (but does not yet attach) a backend-specific annotator for @raw_modality_name (one of
        VisionSensor.RAW_SENSOR_TYPES' values).

        Returns:
            object: Opaque handle to be passed to attach_modality/detach_modality/get_modality_data.
        """

    @abstractmethod
    def attach_modality(self, annotator, camera_resource):
        """Attaches @annotator (from create_annotator) to @camera_resource so it starts collecting data."""

    @abstractmethod
    def detach_modality(self, annotator, camera_resource):
        """Detaches @annotator from @camera_resource. Tolerates backend-specific teardown quirks."""

    @abstractmethod
    def get_modality_data(self, annotator, device=None):
        """
        Returns:
            dict or array: The current raw observation for @annotator -- either a dict with "data"/"info"
                keys or a direct array/tensor, depending on modality (mirrors the underlying annotator's
                own return convention). @device, if given, requests data placed on that device.
        """
