import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.utils.python_utils import classproperty, torch_delete

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HEATSOURCE_META_LINK_TYPE = "heatsource"
m.HEATING_ELEMENT_MARKER_SCALE = [1.0] * 3

# TODO: Delete default values for this and make them required.
m.DEFAULT_TEMPERATURE = 200
m.DEFAULT_HEATING_RATE = 0.04
m.DEFAULT_DISTANCE_THRESHOLD = 0.2


class HeatSourceOrSink(LinkBasedStateMixin, TensorizedValueState):
    """
    This state indicates the heat source or heat sink state of the object.

    Tensorized storage
    ------------------
    VALUES[:] — (N,) bool — active gate per heat source. True when the source is
    currently emitting heat (toggled on if required, not open if required).

    All geometry and config tensors are public (no leading underscore) so that
    Temperature._apply_heat_source_influence() can read them directly without
    violating encapsulation, following the MaxTemperature.TEMPERATURE_IDXS pattern.
    """

    # ── Construction-time constants (resized at _add_obj / _remove_obj) ──────
    TEMPERATURES = None  # (N,) float32 — target temperature of each heat source
    HEATING_RATES = None  # (N,) float32 — rate of temperature change per second
    DISTANCE_THRESHOLDS = None  # (N,) float32 — spatial range (near mode); unused for inside mode
    REQUIRES_INSIDE = None  # (N,) bool
    REQUIRES_TOGGLED_ON = None  # (N,) bool
    REQUIRES_CLOSED = None  # (N,) bool
    TOGGLED_ON_IDXS = None  # (N,) int64 — index into ToggledOn.VALUES; -1 if not applicable

    # ── Per-step geometry (refreshed in global_update, read by Temperature) ──
    POSITIONS = None  # (N, 3) float32 — emitter position (near) or AABB center (inside)
    AABB_LO = None  # (N, 3) float32
    AABB_HI = None  # (N, 3) float32

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "active"

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls.TEMPERATURES = th.empty(0, dtype=th.float32)
        cls.HEATING_RATES = th.empty(0, dtype=th.float32)
        cls.DISTANCE_THRESHOLDS = th.empty(0, dtype=th.float32)
        cls.REQUIRES_INSIDE = th.empty(0, dtype=th.bool)
        cls.REQUIRES_TOGGLED_ON = th.empty(0, dtype=th.bool)
        cls.REQUIRES_CLOSED = th.empty(0, dtype=th.bool)
        cls.TOGGLED_ON_IDXS = th.empty(0, dtype=th.int64)
        cls.POSITIONS = th.empty((0, 3), dtype=th.float32)
        cls.AABB_LO = th.empty((0, 3), dtype=th.float32)
        cls.AABB_HI = th.empty((0, 3), dtype=th.float32)

        # Register callback to keep TOGGLED_ON_IDXS consistent when a ToggledOn is removed.
        # Pattern mirrors MaxTemperature's TEMPERATURE_IDXS update on Temperature removal.
        def _update_toggled_on_idxs(obj):
            if obj not in ToggledOn.OBJ_IDXS:
                return
            deleted_idx = ToggledOn.OBJ_IDXS[obj]
            # Only decrement valid indices (>= 0) that point past the deleted slot.
            valid = cls.TOGGLED_ON_IDXS >= 0
            cls.TOGGLED_ON_IDXS = th.where(
                valid & (cls.TOGGLED_ON_IDXS >= deleted_idx),
                cls.TOGGLED_ON_IDXS - 1,
                cls.TOGGLED_ON_IDXS,
            )

        ToggledOn.add_callback_on_remove(
            name="HeatSourceOrSink_toggled_on_idx_update", callback=_update_toggled_on_idxs
        )

    @classmethod
    def _add_obj(cls, obj):
        super()._add_obj(obj=obj)
        # Append placeholder rows; actual values are written in __init__ after super() returns.
        cls.TEMPERATURES = th.cat([cls.TEMPERATURES, th.zeros(1, dtype=th.float32)])
        cls.HEATING_RATES = th.cat([cls.HEATING_RATES, th.zeros(1, dtype=th.float32)])
        cls.DISTANCE_THRESHOLDS = th.cat([cls.DISTANCE_THRESHOLDS, th.zeros(1, dtype=th.float32)])
        cls.REQUIRES_INSIDE = th.cat([cls.REQUIRES_INSIDE, th.zeros(1, dtype=th.bool)])
        cls.REQUIRES_TOGGLED_ON = th.cat([cls.REQUIRES_TOGGLED_ON, th.zeros(1, dtype=th.bool)])
        cls.REQUIRES_CLOSED = th.cat([cls.REQUIRES_CLOSED, th.zeros(1, dtype=th.bool)])
        cls.TOGGLED_ON_IDXS = th.cat([cls.TOGGLED_ON_IDXS, th.full((1,), -1, dtype=th.int64)])
        cls.POSITIONS = th.cat([cls.POSITIONS, th.zeros((1, 3), dtype=th.float32)], dim=0)
        cls.AABB_LO = th.cat([cls.AABB_LO, th.zeros((1, 3), dtype=th.float32)], dim=0)
        cls.AABB_HI = th.cat([cls.AABB_HI, th.zeros((1, 3), dtype=th.float32)], dim=0)

    @classmethod
    def _remove_obj(cls, obj):
        deleted_idx = cls.OBJ_IDXS[obj]
        cls.TEMPERATURES = torch_delete(cls.TEMPERATURES, [deleted_idx])
        cls.HEATING_RATES = torch_delete(cls.HEATING_RATES, [deleted_idx])
        cls.DISTANCE_THRESHOLDS = torch_delete(cls.DISTANCE_THRESHOLDS, [deleted_idx])
        cls.REQUIRES_INSIDE = torch_delete(cls.REQUIRES_INSIDE, [deleted_idx])
        cls.REQUIRES_TOGGLED_ON = torch_delete(cls.REQUIRES_TOGGLED_ON, [deleted_idx])
        cls.REQUIRES_CLOSED = torch_delete(cls.REQUIRES_CLOSED, [deleted_idx])
        cls.TOGGLED_ON_IDXS = torch_delete(cls.TOGGLED_ON_IDXS, [deleted_idx])
        cls.POSITIONS = torch_delete(cls.POSITIONS, [deleted_idx])
        cls.AABB_LO = torch_delete(cls.AABB_LO, [deleted_idx])
        cls.AABB_HI = torch_delete(cls.AABB_HI, [deleted_idx])
        super()._remove_obj(obj=obj)

    @classmethod
    def _update_values(cls, values):
        active = th.ones(len(cls.IDX_OBJS), dtype=th.bool)
        # ToggledOn IS a TensorizedValueState — pure tensor index lookup, no loop
        if cls.REQUIRES_TOGGLED_ON.any():
            toggled = ToggledOn.VALUES[cls.TOGGLED_ON_IDXS[cls.REQUIRES_TOGGLED_ON], 0] > 0.5
            active[cls.REQUIRES_TOGGLED_ON] &= toggled
        # Open is NOT a TensorizedValueState — per-object call is unavoidable
        for i in th.where(cls.REQUIRES_CLOSED)[0].tolist():
            if cls.IDX_OBJS[i].states[Open].get_value():
                active[i] = False
        return active

    @classmethod
    def global_update(cls):
        if not cls.IDX_OBJS:
            return

        # Refresh per-step geometry (loop over N_hs, typically < 10).
        # Will become a pure tensor op once AABB is tensorized.
        for i, obj in enumerate(cls.IDX_OBJS):
            lo, hi = obj.states[AABB].get_value()
            cls.AABB_LO[i] = lo
            cls.AABB_HI[i] = hi
            if cls.REQUIRES_INSIDE[i]:
                cls.POSITIONS[i] = (lo + hi) / 2.0
            else:
                hs_state = obj.states[cls]
                link = hs_state.link
                cls.POSITIONS[i] = (
                    link.aabb_center if link == hs_state._default_link else link.get_position_orientation()[0]
                )

        super().global_update()  # calls _update_values → refreshes VALUES (active flags)

    def __init__(
        self,
        obj,
        temperature=None,
        heating_rate=None,
        distance_threshold=None,
        requires_toggled_on=False,
        requires_closed=False,
        requires_inside=False,
    ):
        """
        Args:
            obj (StatefulObject): The object with the heat source ability.
            temperature (float): The temperature of the heat source.
            heating_rate (float): Fraction in [0, 1] of the temperature difference with the
                heat source temperature should be received every step, per second.
            distance_threshold (float): The distance threshold which an object needs
                to be closer than in order to receive heat from this heat source.
            requires_toggled_on (bool): Whether the heat source object needs to be
                toggled on to emit heat. Requires toggleable ability if set to True.
            requires_closed (bool): Whether the heat source object needs to be
                closed (e.g. in terms of the joints) to emit heat. Requires openable
                ability if set to True.
            requires_inside (bool): Whether an object needs to be `inside` the
                heat source to receive heat. See the Inside state for details. This
                will mean that the "heating element" link for the object will be
                ignored.
        """
        # super().__init__ calls _add_obj, which appends placeholder rows to class tensors.
        super(HeatSourceOrSink, self).__init__(obj)

        # Set per-instance config (same semantics as original).
        self._temperature = temperature if temperature is not None else m.DEFAULT_TEMPERATURE
        self._heating_rate = heating_rate if heating_rate is not None else m.DEFAULT_HEATING_RATE
        self.distance_threshold = distance_threshold if distance_threshold is not None else m.DEFAULT_DISTANCE_THRESHOLD

        if requires_toggled_on:
            assert ToggledOn in self.obj.states
        self.requires_toggled_on = requires_toggled_on

        if requires_closed:
            assert Open in self.obj.states
        self.requires_closed = requires_closed

        self.requires_inside = requires_inside

        # Write actual config into class-level tensors at our index.
        idx = type(self).OBJ_IDXS[obj]
        type(self).TEMPERATURES[idx] = self._temperature
        type(self).HEATING_RATES[idx] = self._heating_rate
        type(self).DISTANCE_THRESHOLDS[idx] = self.distance_threshold
        type(self).REQUIRES_INSIDE[idx] = requires_inside
        type(self).REQUIRES_TOGGLED_ON[idx] = requires_toggled_on
        type(self).REQUIRES_CLOSED[idx] = requires_closed
        if requires_toggled_on and ToggledOn in obj.states:
            type(self).TOGGLED_ON_IDXS[idx] = ToggledOn.OBJ_IDXS[obj]
        # else: stays -1 as set by _add_obj

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and state_type not in obj.states:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"

        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        for kwarg, state_type in zip(("requires_toggled_on", "requires_closed"), (ToggledOn, Open)):
            if kwargs.get(kwarg, False) and not state_type.is_compatible_asset(prim=prim, **kwargs)[0]:
                return False, f"{cls.__name__} has {kwarg} but obj has no {state_type.__name__} state!"

        return True, None

    @classproperty
    def meta_link_types(cls):
        return [m.HEATSOURCE_META_LINK_TYPE]

    @classmethod
    def requires_meta_link(cls, **kwargs):
        # No meta link required if inside
        return not kwargs.get("requires_inside", False)

    @property
    def _default_link(self):
        # Only supported if we require inside
        return self.obj.root_link if self.requires_inside else super()._default_link

    @property
    def heating_rate(self):
        """
        Returns:
            float: Temperature changing rate of this heat source / sink
        """
        return self._heating_rate

    @property
    def temperature(self):
        """
        Returns:
            float: Temperature of this heat source / sink
        """
        return self._temperature

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB, Inside})
        return deps

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        deps.update({ToggledOn, Open})
        return deps

    def _initialize(self):
        # Run super first
        super()._initialize()
        self.initialize_link_mixin()

    # Nothing needs to be done to save/load HeatSource beyond the inherited
    # TensorizedValueState._dump_state / _load_state (which handle VALUES["active"]).
