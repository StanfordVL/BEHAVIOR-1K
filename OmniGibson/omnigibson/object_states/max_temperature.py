import torch as th

from omnigibson.object_states.temperature import Temperature
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty


class MaxTemperature(TensorizedValueState):
    """
    This state remembers the highest temperature reached by an object.
    """

    # th.tensor: Array of Temperature.OBJ_IDXS N-indices that correspond to each MaxTemperature object
    TEMPERATURE_IDXS = None

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

        # Rebuild TEMPERATURE_IDXS: for each MaxTemp N index, find matching Temperature N index.
        # MaxTemp objects are a subset of Temperature objects with the same relative paths.
        # cls.OBJ_IDXS is insertion-ordered (Python 3.7+), so iterating yields keys in N=0,1,2,... order.
        cls.TEMPERATURE_IDXS = th.tensor(
            [Temperature.OBJ_IDXS[rel_path] for rel_path in cls.OBJ_IDXS],
            dtype=th.long,
        )

        # Initialize new VALUE slots (not carried over) to -inf
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(len(cls.IDX_OBJS)):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = -float("inf")

    @classmethod
    def _update_values(cls, values):
        # Value is max between stored values and current temperature values.
        # Temperature.VALUES is (S, N_temp); cls.TEMPERATURE_IDXS maps MaxTemp N → Temperature N,
        # so Temperature.VALUES[:, cls.TEMPERATURE_IDXS] has shape (S, N_max).
        return th.maximum(values, Temperature.VALUES[:, cls.TEMPERATURE_IDXS])

    @classproperty
    def value_name(cls):
        return "max_temperature"
