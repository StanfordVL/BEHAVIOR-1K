import torch as th
import warp as wp

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import _wp_from_torch
from omnigibson.utils.python_utils import classproperty


@wp.kernel
def _temperature_decay_kernel(
    values: wp.array2d(dtype=wp.float32),
    incoming_heat_rate: wp.array2d(dtype=wp.float32),
    default_temp: wp.float32,
    decay_rate: wp.float32,
    dt: wp.array(dtype=wp.float32),
):
    """
    Fused exponential decay toward `default_temp` plus consumption of the per-step
    incoming heat rate scratch buffer. One thread per (scene, obj).
        values += (default_temp - values) * decay_rate * dt[0] + incoming_heat_rate * dt[0]
    Then zero the consumed entry so the buffer is fresh for the next-step writers
    (HSS at the start of the next step plus any lag-1 writes from OnFire / cloth
    post-pass that happen after this kernel).

    `dt` is a single-element warp array (read as `dt[0]`) rather than a scalar so the
    per-frame value written in pre_update is visible inside the captured graph without
    re-capturing it (time-dependent-state mechanism).
    """
    s, o = wp.tid()
    values[s, o] = values[s, o] + (default_temp - values[s, o]) * decay_rate * dt[0] + incoming_heat_rate[s, o] * dt[0]
    incoming_heat_rate[s, o] = wp.float32(0.0)


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(TensorizedAbsoluteState):
    # (S, N) float32 — per-step rate accumulator. HeatSourceOrSink (and cloth post-pass)
    # atomic_add into this; the decay kernel consumes it and zeros it at the end of the same
    # kernel so contributions from later-running states (OnFire, cloth post-pass) accumulate
    # for the *next* step's consumption (lag-1, matches the pre-tensorized behavior).
    INCOMING_HEAT_RATE = None
    INCOMING_HEAT_RATE_WP = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        # HSS stays *optional*: the topo sort still places it before Temperature (so the
        # captured graph runs HSS's atomic_add into INCOMING_HEAT_RATE before this kernel
        # reads it), but objects without HSS on them (e.g. cookable food items) are still
        # eligible to receive a Temperature state.
        deps.add(HeatSourceOrSink)
        return deps

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        # Initialize new VALUE slots (not carried over) to DEFAULT_TEMPERATURE
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(len(cls.IDX_OBJS)):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE
                        cls.VALUES_CPU[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE

        # Allocate the per-step heat-rate scratch buffer. No carry-over: a partial step from
        # the previous configuration would be applied to the wrong indices.
        if cls.VALUES.numel() > 0:
            cls.INCOMING_HEAT_RATE = th.zeros(cls.VALUES.shape, dtype=th.float32, device="cuda")
            cls.INCOMING_HEAT_RATE_WP = _wp_from_torch(cls.INCOMING_HEAT_RATE)
        else:
            cls.INCOMING_HEAT_RATE = None
            cls.INCOMING_HEAT_RATE_WP = None

    @classmethod
    def _update_values(cls, values):
        # Apply temperature decay + incoming heat rate consumption via Warp kernel. dt is read
        # from cls._dt at kernel-launch time inside the captured graph, so the per-frame value
        # written in pre_update is visible without re-capturing the graph.
        if cls.VALUES_WP is None or cls.INCOMING_HEAT_RATE_WP is None:
            return
        S, O = cls.VALUES.shape[:2]
        wp.launch(
            kernel=_temperature_decay_kernel,
            dim=(S, O),
            inputs=[
                cls.VALUES_WP,
                cls.INCOMING_HEAT_RATE_WP,
                wp.float32(m.DEFAULT_TEMPERATURE),
                wp.float32(m.TEMPERATURE_DECAY_SPEED),
                cls._dt,
            ],
            device="cuda",
        )

    @classproperty
    def value_name(cls):
        return "temperature"
