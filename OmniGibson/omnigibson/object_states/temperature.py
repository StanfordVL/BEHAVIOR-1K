import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.heat_source_or_sink import HeatSourceOrSink
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty

# Create settings for this module
m = create_module_macros(module_path=__file__)

# TODO: Consider sourcing default temperature from scene
# Default ambient temperature.
m.DEFAULT_TEMPERATURE = 23.0  # degrees Celsius

# What fraction of the temperature difference with the default temperature should be decayed every step.
m.TEMPERATURE_DECAY_SPEED = 0.02  # per second. We'll do the conversion to steps later.


class Temperature(TensorizedValueState):
    def __init__(self, obj):
        # Run super first
        super(Temperature, self).__init__(obj)

        # Set value to be default
        self._set_value(m.DEFAULT_TEMPERATURE)

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
        # Apply temperature decay toward ambient
        return values + (m.DEFAULT_TEMPERATURE - values) * m.TEMPERATURE_DECAY_SPEED * og.sim.get_sim_step_dt()

    @classmethod
    def global_update(cls):
        # Lazy import to avoid circular dependency: on_fire → temperature → on_fire
        from omnigibson.object_states.on_fire import OnFire

        super().global_update()  # applies ambient decay via _update_values()

        # Pin active OnFire objects to fire_temperature so decay cannot extinguish them.
        if OnFire.IDX_OBJS and OnFire.VALUES.any():
            active_fire = th.where(OnFire.VALUES)[0]
            temp_idxs = OnFire.TEMPERATURE_IDXS[active_fire]
            cls.VALUES[temp_idxs] = OnFire.TEMPERATURES[active_fire]

        cls._apply_heat_source_influence(HeatSourceOrSink)
        cls._apply_heat_source_influence(OnFire)

    @classmethod
    def _get_obj_positions(cls):
        """
        Compute AABB centers for all tracked temperature objects.

        Returns:
            th.Tensor: (T, 3) float32 — AABB center of each object in cls.IDX_OBJS order.
        """
        return th.stack([sum(obj.states[AABB].get_value()) / 2.0 for obj in cls.IDX_OBJS])

    @classmethod
    def _get_affected_matrix(cls, obj_positions, heatsource_cls):
        """
        Compute which active heat sources affect which temperature objects.

        Near mode uses batched cdist; inside mode uses vectorized AABB containment
        followed by a semantic Inside check for spatially plausible candidates.

        Args:
            obj_positions (th.Tensor): (T, 3) float32 — AABB centers of all temperature
                objects in cls.IDX_OBJS order.
            heatsource_cls: HeatSourceOrSink or OnFire class whose tensors to read.

        Returns:
            th.Tensor: (H, T) bool — affected[h, t] is True if heat source h affects
                temperature object t. H = len(heatsource_cls.IDX_OBJS),
                T = len(cls.IDX_OBJS). Inactive heat sources have all-False rows.
        """
        H = len(heatsource_cls.IDX_OBJS)
        T = len(cls.IDX_OBJS)
        affected = th.zeros((H, T), dtype=th.bool)

        active_mask = heatsource_cls.VALUES  # (H,) bool

        # ── Near mode: batched cdist ──────────────────────────────────────────
        near_mask = active_mask & ~heatsource_cls.REQUIRES_INSIDE
        if near_mask.any():
            near_idxs = th.where(near_mask)[0]
            positions = heatsource_cls.POSITIONS[near_idxs]  # (H_near, 3)
            thresholds = heatsource_cls.DISTANCE_THRESHOLDS[near_idxs]  # (H_near,)

            dists = th.cdist(positions, obj_positions)  # (H_near, T)
            affected[near_idxs] = dists < thresholds.unsqueeze(1)  # (H_near, T)

        # ── Inside mode: AABB containment + semantic Inside filter ────────────
        inside_mask = active_mask & heatsource_cls.REQUIRES_INSIDE
        if inside_mask.any():
            inside_idxs = th.where(inside_mask)[0]
            aabb_lo = heatsource_cls.AABB_LO[inside_idxs]  # (H_inside, 3)
            aabb_hi = heatsource_cls.AABB_HI[inside_idxs]  # (H_inside, 3)

            # Vectorized AABB containment: (H_inside, T)
            in_aabb = (
                (obj_positions.unsqueeze(0) >= aabb_lo.unsqueeze(1))
                & (obj_positions.unsqueeze(0) <= aabb_hi.unsqueeze(1))
            ).all(dim=-1)

            # use state Inside as filter again.
            # TODO: (andi) tensorized after Inside is tensorized.
            for inside_idx, heatsource_idx in enumerate(inside_idxs.tolist()):
                heatsource_obj = heatsource_cls.IDX_OBJS[heatsource_idx]
                for temperature_idx in th.where(in_aabb[inside_idx])[0].tolist():
                    temperature_obj = cls.IDX_OBJS[temperature_idx]
                    if not temperature_obj.states[Inside].get_value(heatsource_obj):
                        in_aabb[inside_idx, temperature_idx] = False

            affected[inside_idxs] = in_aabb

        return affected

    @classmethod
    def _apply_heat_source_influence(cls, heatsource_cls):
        if not heatsource_cls.IDX_OBJS or not cls.IDX_OBJS:
            return
        if not heatsource_cls.VALUES.any():
            return

        dt = og.sim.get_sim_step_dt()
        temperature_obj_positions = cls._get_obj_positions()  # (T, 3)
        affected = cls._get_affected_matrix(temperature_obj_positions, heatsource_cls)  # (H, T)

        # (H, T) temperature delta, zeroed for unaffected pairs
        temp_diff = heatsource_cls.TEMPERATURES.unsqueeze(1) - cls.VALUES.unsqueeze(0)  # (H, T)
        deltas = temp_diff * heatsource_cls.HEATING_RATES.unsqueeze(1) * dt  # (H, T)
        deltas[~affected] = 0.0

        # Multiple heat sources sum (order-independent)
        cls.VALUES += deltas.sum(dim=0)  # (T,)

    @classmethod
    def get_objs_affected_by_heatsource(cls, heatsources, objs):
        """
        Returns the subset of @heatsources that are actively heating at least one object in @objs.

        Args:
            heatsources (list of StatefulObject): Candidate heat source objects to test.
                May contain HeatSourceOrSink and/or OnFire objects.
            objs (list of StatefulObject): Target objects to check whether they are being heated.

        Returns:
            list of StatefulObject: Heat sources from @heatsources that affect at least one
                object in @objs.
        """
        # Lazy import to avoid circular dependency: on_fire → temperature → on_fire
        from omnigibson.object_states.on_fire import OnFire

        if not cls.IDX_OBJS:
            return []
        # Only track those with Temperature state
        objs_tracked = [obj for obj in objs if obj in cls.OBJ_IDXS]
        if not objs_tracked:
            return []

        result = []
        temperature_obj_positions = cls._get_obj_positions()

        # Partition heatsources into HeatSourceOrSink vs OnFire
        heatsource_only = []
        on_fires = []
        for heatsource in heatsources:
            if heatsource in HeatSourceOrSink.OBJ_IDXS:
                heatsource_only.append(heatsource)
            else:
                on_fires.append(heatsource)

        for heatsource_cls, heatsource_tracked in ((HeatSourceOrSink, heatsource_only), (OnFire, on_fires)):
            if not heatsource_tracked or not heatsource_cls.IDX_OBJS or not heatsource_cls.VALUES.any():
                continue
            affected = cls._get_affected_matrix(temperature_obj_positions, heatsource_cls)
            result.extend(
                hs
                for hs in heatsource_tracked
                if any(affected[heatsource_cls.OBJ_IDXS[hs], cls.OBJ_IDXS[obj]] for obj in objs_tracked)
            )

        return result

    @classproperty
    def value_name(cls):
        return "temperature"
