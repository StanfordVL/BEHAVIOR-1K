import torch as th
import warp as wp

from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


def _wp_from_torch(t):
    """
    Wrap a torch tensor as a wp.array. Bool tensors are wrapped as wp.uint8
    (storage is byte-per-bool in torch already).
    """
    if t.dtype == th.bool:
        return wp.from_torch(t.view(th.uint8), dtype=wp.uint8)
    return wp.from_torch(t)


class TensorizedState:
    """
    Mixin holding the shared lifecycle and class-level tensors for tensorized states.

    Concrete tensorized states should inherit through one of two front-ends:
      - ``TensorizedAbsoluteState`` for single-object values, VALUES shape (S, N, *value_shape)
      - ``TensorizedRelativeState`` for pairwise values, VALUES shape (S, N, N, *value_shape)

    Both front-ends provide their own ``global_initialize`` / ``initialize_view`` to allocate
    their respective VALUES tensor; everything else (pre/post update, change detection,
    wp.array wrapping, graph_dirty flag) lives here and is shared.

    Subclasses MUST implement ``_update_values``.
    """

    # Tensor of raw internally tracked values — GPU-resident during computation.
    # Shape is (S, N, ...) for absolute or (S, N, N, ...) for relative.
    VALUES = None

    # Pinned CPU mirror of VALUES — updated via async DMA after each global_update pass.
    # _get_value() always reads from here to avoid GPU stalls for Python callers.
    VALUES_CPU = None

    # CPU tensor to store VALUES_CPU from the previous step — used for change detection.
    PREV_VALUES = None

    # Dictionary mapping relative prim path to index in the N dimension of VALUES
    OBJ_IDXS = None

    # 2-D list: IDX_OBJS[scene_idx][obj_idx] = object instance (or None if absent in that scene)
    IDX_OBJS = None

    # Int representing per-object state size (flattened size of value_shape)
    STATE_SIZE = None

    # wp.array views over VALUES for use inside Warp kernels and graph capture.
    # Wrapped once in initialize_view; never re-created per call. Re-wrapped on every
    # initialize_view because torch storage is reallocated there.
    VALUES_WP = None

    # wp.array view over the (pinned) VALUES_CPU mirror. Used so that the GPU→CPU mirror
    # copy inside the captured wp.graph runs as a wp.copy (graph-safe), not as torch's
    # .copy_ which uses the legacy stream and is forbidden during capture.
    VALUES_CPU_WP = None

    # Set to True any time any subclass's initialize_view runs, signalling that the
    # captured wp.graph holds stale pointers/shapes and must be re-captured. The simulator
    # checks-and-resets this before each step. update_handles() always calls the view APIs'
    # initialize_view before every TensorizedState's initialize_view, so any viewAPI
    # buffer reallocation is automatically covered.
    graph_dirty = True

    # Set to True whenever something has happened that could leave VALUES_CPU one frame
    # behind the live world: a physics step, a direct pose mutation, a joint mutation, etc.
    # Cleared by ``og.sim._refresh_state_caches()``. The lazy-refresh gate in
    # ``TensorizedAbsoluteState._get_value`` / ``TensorizedRelativeState._get_value`` checks
    # this flag and triggers a refresh on first read so callers always observe fresh state
    # without having to know about the cache lifecycle.
    caches_dirty = False

    # Re-entrance guard for the lazy refresh: while `_refresh_state_caches` is running, any
    # nested `_get_value` should read the (now-being-recomputed) cache directly without
    # re-triggering a refresh. Set/cleared inside the simulator helper.
    _refresh_in_progress = False

    # Simulator time (seconds) at the previous _update_values call. Used by pre_update
    # to compute the seconds elapsed since the last tick and store it in _dt for subclasses
    # that do time-dependent math (Temperature, ToggledOn, SlicerActive). Reset on initialize_view
    # so a stop→play cycle (which resets og.sim.current_time to 0) does not produce a negative dt.
    _last_update_time = 0.0

    # Single-element wp.array on cuda holding the seconds elapsed since the last _update_values
    # tick. Written in pre_update (outside the captured graph) and read by time-dependent
    # subclasses' kernels inside the captured graph. We use a wp.array (not a wp.float32 scalar
    # passed at launch) so per-frame mutation is visible to graph replays without re-capture —
    # scalars passed via wp.float32(...) bake the value at capture time.
    _dt = None

    @classmethod
    def maybe_refresh_caches(cls):
        """Helper for `get_value`: if caches are dirty (and we're not already
        mid-refresh), run the lazy refresh so the upcoming read sees fresh values.
        """
        if cls.caches_dirty and not cls._refresh_in_progress:
            import omnigibson as og  # local import to avoid module-level cycle

            og.sim._refresh_state_caches()

    @classmethod
    def initialize_view(cls):
        """
        Δt-tracking setup shared across tensorized states. Subclasses' initialize_view should
        call ``super().initialize_view()`` after they finish allocating VALUES.

        Snapshots the current sim time so the next _update_values invocation sees Δt=0, and
        allocates the 1-element wp.array that pre_update writes into each step.
        """
        import omnigibson as og  # local import to avoid module-level cycle

        cls._last_update_time = og.sim.current_time
        if cls._dt is None:
            cls._dt = wp.zeros(shape=(1,), dtype=wp.float32, device="cuda")

    def get_value(self, *args, **kwargs):
        self.maybe_refresh_caches()
        return super().get_value(*args, **kwargs)

    @classmethod
    def pre_update(cls):
        """
        CPU-side prep run BEFORE global_update each step. Snapshots VALUES_CPU into
        PREV_VALUES so post_update() can detect changes after the warp work completes,
        and refreshes cls._dt with the seconds elapsed since the previous update.

        Lives outside the captured wp.graph.

        Subclasses may override to add per-state CPU prep (e.g. ToggledOn refreshing
        marker world poses).
        """
        if cls.VALUES_CPU is None or cls.VALUES_CPU.numel() == 0:
            return
        cls.PREV_VALUES.copy_(cls.VALUES_CPU)

        # Refresh dt for time-dependent kernels. max(0, …) guards against a stop→play cycle,
        # which resets Isaac's current_time to 0 — without the clamp the next dt would be negative.
        import omnigibson as og  # local import to avoid module-level cycle

        curr_time = og.sim.current_time
        dt_value = max(0.0, curr_time - cls._last_update_time)
        cls._last_update_time = curr_time
        if cls._dt is not None:
            cls._dt.fill_(dt_value)

    @classmethod
    def global_update(cls):
        """
        Globally update all values via _update_values() and async-copy to VALUES_CPU.
        Change detection and post_update() are called separately by the simulator after synchronize().
        Skips if there are no tracked objects.

        Should be capturable inside wp.graph: only emits CUDA work via Warp
        kernel launches and Warp memcpy.
        """
        if cls.VALUES is None or cls.VALUES.numel() == 0:
            return

        cls._update_values(values=cls.VALUES)
        # Mirror VALUES → VALUES_CPU. Use wp.copy when both wp.array handles exist (graph-safe);
        # fall back to torch's non-blocking copy otherwise (e.g. partial init).
        # TODO(vector): Why would one (torch) exist and not the other (wp)?
        if cls.VALUES_WP is not None and cls.VALUES_CPU_WP is not None:
            wp.copy(cls.VALUES_CPU_WP, cls.VALUES_WP)
        else:
            cls.VALUES_CPU.copy_(cls.VALUES, non_blocking=True)

    @classmethod
    def post_update(cls):
        """
        Called by the simulator after th.cuda.synchronize(). Compares VALUES_CPU with PREV_VALUES
        (both CPU) to detect changes, fires state_updated() for affected objects, then updates PREV_VALUES.

        Reduce semantics:
        - Absolute (S, N, *value_shape): reduces over value_shape dims to get per-(scene, obj) row mask.
        - Relative (S, N, N, *value_shape): reduces over (other-obj, *value_shape) dims to get per-(scene, self-obj)
          row mask. A pairwise-cell change fires state_updated() once on the row's self-object.
        """
        if cls.VALUES_CPU is None or cls.VALUES_CPU.numel() == 0:
            return
        S = cls.VALUES_CPU.shape[0]

        diff = cls.VALUES_CPU != cls.PREV_VALUES
        changed_mask = th.any(diff, dim=tuple(range(2, diff.ndim))) if diff.ndim > 2 else diff
        for s_idx in range(S):
            for obj_idx in th.where(changed_mask[s_idx])[0].tolist():
                obj = cls.IDX_OBJS[s_idx][obj_idx]
                obj.state_updated()

    @classmethod
    def _update_values(cls, values):
        """
        Updates all internally tracked @values for this object state. Should be implemented by subclass.
        Mutates @values in-place. Must not return anything.

        Args:
            values (th.tensor): Tensorized value array. Shape depends on the front-end:
                (S, N, *value_shape) for TensorizedAbsoluteState,
                (S, N, N, *value_shape) for TensorizedRelativeState.
        """
        raise NotImplementedError

    @classproperty
    def value_shape(cls):
        """
        Returns:
            tuple: Expected shape of the per-object state instance value. Default is () (scalar).
        """
        return ()

    @classproperty
    def value_type(cls):
        """
        Returns:
            type: Type of the internal value array, e.g., bool, th.uint, th.float32, etc. Default is th.float32
        """
        return th.float32

    @classproperty
    def value_name(cls):
        """
        Returns:
            str: Name of the value key to assign when dumping / loading the state. Should be implemented by subclass
        """
        raise NotImplementedError

    @property
    def state_size(self):
        # This is merely the class state size
        return self.STATE_SIZE
