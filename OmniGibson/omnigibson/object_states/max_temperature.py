import warp as wp

from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.utils.python_utils import classproperty


@wp.kernel
def _max_temperature_kernel(
    max_values: wp.array2d(dtype=wp.float32),  # cls.VALUES (S, O_max)
    temperature_values: wp.array2d(dtype=wp.float32),  # Temperature.VALUES (S, O_temp)
    temperature_idxs: wp.array(dtype=wp.int32),  # (O_max,) — O_max → O_temp
):
    """
    Element-wise max with index gather:
        max_values[s, o] = max(max_values[s, o], temperature_values[s, temperature_idxs[o]])
    One thread per (scene, max_obj_idx).
    """
    s, o = wp.tid()
    t = temperature_values[s, temperature_idxs[o]]
    if t > max_values[s, o]:
        max_values[s, o] = t


class MaxTemperature(TensorizedAbsoluteState):
    """
    This state remembers the highest temperature reached by an object.
    """

    # wp.array(int32) on CUDA: maps each MaxTemperature N index to the matching Temperature N index.
    # Built directly as a wp.array (no torch view) — only the kernel reads it.
    TEMPERATURE_IDXS_WP = None
    # Reference to Temperature.VALUES_WP grabbed in _refresh_external_wp_handles().
    TEMPERATURE_VALUES_WP = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Temperature)
        return deps

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        # Rebuild TEMPERATURE_IDXS_WP: for each MaxTemp N index, find matching Temperature N index.
        # MaxTemp objects are a subset of Temperature objects with the same relative paths.
        # cls.OBJ_IDXS is insertion-ordered (Python 3.7+), so iterating yields keys in N=0,1,2,... order.
        # Allocated directly as a Warp array as only the kernel reads it.
        idxs = [Temperature.OBJ_IDXS[rel_path] for rel_path in cls.OBJ_IDXS]
        cls.TEMPERATURE_IDXS_WP = wp.array(idxs, dtype=wp.int32, device="cuda") if idxs else None

        # Initialize new VALUE slots (not carried over) to -inf
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(len(cls.IDX_OBJS)):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = -float("inf")
                        cls.VALUES_CPU[s_idx, obj_idx] = -float("inf")

    @classmethod
    def _refresh_external_wp_handles(cls):
        """
        Snapshot the current Temperature.VALUES_WP reference. Called by the simulator
        immediately before wp.ScopedCapture so the captured graph kernels read from the
        live Temperature buffer (Temperature.initialize_view may have re-wrapped its
        VALUES_WP since the last graph capture).
        """
        cls.TEMPERATURE_VALUES_WP = Temperature.VALUES_WP

    @classmethod
    def _update_values(cls, values):
        # Value is max between stored values and current temperature values.
        # Temperature.VALUES is (S, N_temp); cls.TEMPERATURE_IDXS maps MaxTemp N → Temperature N,
        # so Temperature.VALUES[:, cls.TEMPERATURE_IDXS] has shape (S, N_max).

        # cls.TEMPERATURE_VALUES_WP is refreshed by the simulator.
        if cls.VALUES_WP is None or cls.TEMPERATURE_IDXS_WP is None or cls.TEMPERATURE_VALUES_WP is None:
            return
        S, O = cls.VALUES.shape[:2]
        wp.launch(
            kernel=_max_temperature_kernel,
            dim=(S, O),
            inputs=[cls.VALUES_WP, cls.TEMPERATURE_VALUES_WP, cls.TEMPERATURE_IDXS_WP],
            device="cuda",
        )

    @classproperty
    def value_name(cls):
        return "max_temperature"
