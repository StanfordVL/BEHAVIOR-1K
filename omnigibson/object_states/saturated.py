import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanStateMixin
from omnigibson.systems.system_base import UUID_TO_SYSTEMS, REGISTERED_SYSTEMS
from omnigibson.utils.python_utils import get_uuid


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Default saturation limit
m.DEFAULT_SATURATION_LIMIT = 1e6


class ModifiedParticles(RelativeObjectState):
    """
    Object state tracking number of modified particles for a given object
    """
    def __init__(self, obj):
        # Run super first
        super().__init__(obj=obj)

        # Set internal values
        self.particle_counts = None

    def _initialize(self):
        super()._initialize()

        # Set internal variables
        self.particle_counts = dict()

    def _get_value(self, system):
        # If system isn't stored, return 0, otherwise, return the actual value
        return self.particle_counts.get(system, 0)

    def _set_value(self, system, new_value):
        assert new_value >= 0, "Cannot set ModifiedParticles value to be less than 0!"
        # Remove the value from the dictionary if we're setting it to zero (save memory)
        if new_value == 0 and system in self.particle_counts:
            self.particle_counts.pop(system)
        else:
            self.particle_counts[system] = new_value

    def _sync_systems(self, systems):
        """
        Helper function for forcing internal systems to be synchronized with external list of @systems.

        NOTE: This may override internal state

        Args:
            systems (list of BaseSystem): List of system(s) that should be actively tracked internally
        """
        self.particle_counts = {system: -1 for system in systems}

    @property
    def state_size(self):
        # Two entries per system (name + count) + number of systems
        return len(self.particle_counts) * 2 + 1

    def _dump_state(self):
        state = dict(n_systems=len(self.particle_counts))
        for system, val in self.particle_counts.items():
            state[system.name] = val
        return state

    def _load_state(self, state):
        self.particle_counts = {REGISTERED_SYSTEMS[system_name]: val for system_name, val in state.items() if system_name != "n_systems" and val > 0}

    def _serialize(self, state):
        state_flat = np.array([state["n_systems"]], dtype=float)
        if state["n_systems"] > 0:
            system_names = tuple(state.keys())[1:]
            state_flat = np.concatenate(
                [state_flat,
                 np.concatenate([(get_uuid(system_name), state[system_name]) for system_name in system_names])]
            ).astype(float)
        return state_flat

    def _deserialize(self, state):
        n_systems = int(state[0])
        state_shaped = state[1:1 + n_systems * 2].reshape(-1, 2)
        state_dict = dict(n_systems=n_systems)
        systems = []
        for uuid, val in state_shaped:
            system = UUID_TO_SYSTEMS[int(uuid)]
            state_dict[system.name] = int(val)
            systems.append(system)

        # Sync systems so that state size sanity check succeeds
        self._sync_systems(systems=systems)

        return state_dict, n_systems * 2 + 1


class Saturated(RelativeObjectState, BooleanStateMixin):
    def __init__(self, obj, default_limit=m.DEFAULT_SATURATION_LIMIT):
        # Run super first
        super().__init__(obj=obj)

        # Limits
        self._default_limit = default_limit
        self._limits = None

    def _initialize(self):
        super()._initialize()

        # Set internal variables
        self._limits = dict()

    @property
    def limits(self):
        """
        Returns:
            dict: Maps system to limit count for that system, if it exists
        """
        return self._limits

    def get_limit(self, system):
        """
        Grabs the internal particle limit for @system

        Args:
            system (BaseSystem): System to limit

        Returns:
            init: Number of particles representing limit for the given @system
        """
        return self._limits.get(system, self._default_limit)

    def set_limit(self, system, limit):
        """
        Sets internal particle limit @limit for @system

        Args:
            system (BaseSystem): System to limit
            limit (int): Number of particles representing limit for the given @system
        """
        self._limits[system] = limit

    def _get_value(self, system):
        limit = self.get_limit(system=system)

        # If requested, run sanity check to make sure we're not over the limit with this system's particles
        count = self.obj.states[ModifiedParticles].get_value(system)
        assert count <= limit, f"{self.__class__.__name__} should not be over the limit! Max: {limit}, got: {count}"

        return count == limit

    def _set_value(self, system, new_value):
        # Only set the value if it's different than what currently exists
        if new_value != self.get_value(system):
            self.obj.states[ModifiedParticles].set_value(system, self.get_limit(system=system) if new_value else 0)
        return True

    def get_texture_change_params(self):
        colors = []

        for system in self._limits.keys():
            if self.get_value(system):
                colors.append(system.color)

        if len(colors) == 0:
            # If no fluid system has Soaked=True, keep the default albedo value
            albedo_add = 0.0
            diffuse_tint = [1.0, 1.0, 1.0]
        else:
            albedo_add = 0.1
            avg_color = np.mean(colors, axis=0)
            # Add a tint of avg_color
            # We want diffuse_tint to sum to 2.5 to result in the final RGB to sum to 1.5 on average
            # This is because an average RGB color sum to 1.5 (i.e. [0.5, 0.5, 0.5])
            # (0.5 [original avg RGB per channel] + 0.1 [albedo_add]) * 2.5 = 1.5
            diffuse_tint = np.array([0.5, 0.5, 0.5]) + avg_color / np.sum(avg_color)
            diffuse_tint = diffuse_tint.tolist()

        return albedo_add, diffuse_tint

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ModifiedParticles)
        return deps

    def _sync_systems(self, systems):
        """
        Helper function for forcing internal systems to be synchronized with external list of @systems.

        NOTE: This may override internal state

        Args:
            systems (list of BaseSystem): List of system(s) that should be actively tracked internally
        """
        self._limits = {system: m.DEFAULT_SATURATION_LIMIT for system in systems}

    @property
    def state_size(self):
        # Limit per entry * 2 (UUID, value) + default limit + n limits
        return len(self._limits) * 2 + 2

    def _dump_state(self):
        state = dict(n_systems=len(self._limits), default_limit=self._default_limit)
        for system, limit in self._limits.items():
            state[system.name] = limit
        return state

    def _load_state(self, state):
        self._limits = dict()
        for k, v in state.items():
            if k == "n_systems":
                continue
            elif k == "default_limit":
                self._default_limit = v
            # TODO: Make this an else once fresh round of sampling occurs (i.e.: no more outdated systems stored)
            elif k in REGISTERED_SYSTEMS:
                self._limits[REGISTERED_SYSTEMS[k]] = v

    def _serialize(self, state):
        state_flat = np.array([state["n_systems"], state["default_limit"]], dtype=float)
        if state["n_systems"] > 0:
            system_names = tuple(state.keys())[2:]
            state_flat = np.concatenate(
                [state_flat,
                 np.concatenate([(get_uuid(system_name), state[system_name]) for system_name in system_names])]
            ).astype(float)
        return state_flat

    def _deserialize(self, state):
        n_systems = int(state[0])
        state_dict = dict(n_systems=n_systems, default_limit=int(state[1]))
        state_shaped = state[2:2 + n_systems * 2].reshape(-1, 2)
        systems = []
        for uuid, val in state_shaped:
            system = UUID_TO_SYSTEMS[int(uuid)]
            state_dict[system.name] = int(val)
            systems.append(system)

        # Sync systems so that state size sanity check succeeds
        self._sync_systems(systems=systems)

        return state_dict, 2 + n_systems * 2
