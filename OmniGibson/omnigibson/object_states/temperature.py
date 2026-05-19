import warp as wp

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.utils.python_utils import classproperty


@wp.kernel
def _temperature_decay_kernel(
    values: wp.array2d(dtype=wp.float32),
    default_temp: wp.float32,
    decay_rate: wp.float32,
    dt: wp.float32,
):
    """
    In-place exponential decay toward `default_temp`. One thread per (scene, obj).
    Equivalent to: values += (default_temp - values) * decay_rate * dt
    """
    s, o = wp.tid()
    values[s, o] = values[s, o] + (default_temp - values[s, o]) * decay_rate * dt


# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(TensorizedAbsoluteState):
    # Index of the last og.sim.step() call during which _update_values ran. The gate in
    # _update_values uses this to skip ticking during lazy refreshes triggered between
    # real sim steps (e.g. by sample_kinematics).
    _LAST_UPDATE_STEP_INDEX = -1

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        cls._LAST_UPDATE_STEP_INDEX = og.sim.step_call_index

        # Initialize new VALUE slots (not carried over) to DEFAULT_TEMPERATURE
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(len(cls.IDX_OBJS)):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE
                        cls.VALUES_CPU[s_idx, obj_idx] = m.DEFAULT_TEMPERATURE

    @classmethod
    def update_temperature_from_heatsource_or_sink(cls, objs, temperature, rate):
        """
        Updates @objs' internal temperatures based on @temperature and @rate

        Args:
            objs (Iterable of StatefulObject): Objects whose temperatures should be updated
            temperature (float): Heat source / sink temperature
            rate (float): Heating rate of the source / sink
        """
        # Get (scene, obj) index pairs for the affected objects
        s_idxs = [obj.scene.idx for obj in objs]
        n_idxs = [cls.OBJ_IDXS[obj.relative_prim_path] for obj in objs]
        cls.VALUES[s_idxs, n_idxs] += (temperature - cls.VALUES[s_idxs, n_idxs]) * rate * og.sim.get_sim_step_dt()
        # Mirror to CPU immediately — this runs in Pass 2 (after the async sync point), so
        # VALUES_CPU would otherwise lag by one step for heatsource contributions.
        cls.VALUES_CPU[s_idxs, n_idxs] = cls.VALUES[s_idxs, n_idxs].cpu()

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        deps.add(HeatSourceOrSink)
        return deps

    @classmethod
    def _update_values(cls, values):
        # Apply temperature decay via Warp kernel
        if cls.VALUES_WP is None:
            return
        # Out-of-step update
        step = og.sim.step_call_index
        if step == cls._LAST_UPDATE_STEP_INDEX:
            return
        cls._LAST_UPDATE_STEP_INDEX = step
        S, O = cls.VALUES.shape[:2]
        wp.launch(
            kernel=_temperature_decay_kernel,
            dim=(S, O),
            inputs=[
                cls.VALUES_WP,
                wp.float32(m.DEFAULT_TEMPERATURE),
                wp.float32(m.TEMPERATURE_DECAY_SPEED),
                wp.float32(og.sim.get_sim_step_dt()),
            ],
            device="cuda",
        )

    @classproperty
    def value_name(cls):
        return "temperature"
