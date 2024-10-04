import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.systems.system_base import UUID_TO_SYSTEM_NAME

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
        return self.particle_counts.get(system.name, 0)

    def _set_value(self, system, new_value):
        assert new_value >= 0, "Cannot set ModifiedParticles value to be less than 0!"
        # Remove the value from the dictionary if we're setting it to zero (save memory)
        if new_value == 0 and system.name in self.particle_counts:
            self.particle_counts.pop(system.name)
        else:
            self.particle_counts[system.name] = new_value

    def _sync_systems(self, system_names):
        """
        Helper function for forcing internal systems to be synchronized with external list of @systems.

        NOTE: This may override internal state

        Args:
            system_names (list of str): List of system name(s) that should be actively tracked internally
        """
        self.particle_counts = {name: -1 for name in system_names}

    @property
    def state_size(self):
        # Two entries per system (name + count) + number of systems
        return len(self.particle_counts) * 2 + 1

    def _dump_state(self):
        state = dict(n_systems=len(self.particle_counts))
        for system_name, val in self.particle_counts.items():
            state[system_name] = val
        return state

    def _load_state(self, state):
        self.particle_counts = {
            system_name: val for system_name, val in state.items() if system_name != "n_systems" and val > 0
        }

    def serialize(self, state):
        state_flat = th.tensor([state["n_systems"]], dtype=th.float32)
        if state["n_systems"] > 0:
            system_names = tuple(state.keys())[1:]
            state_flat = th.cat(
                [
                    state_flat,
                    th.cat(
                        [
                            (
                                self.obj.scene.get_system(system_name, force_init=False).uuid,
                                state[system_name],
                            )
                            for system_name in system_names
                        ]
                    ),
                ]
            )
        return state_flat

    def deserialize(self, state):
        n_systems = int(state[0])
        state_shaped = state[1 : 1 + n_systems * 2].reshape(-1, 2)
        state_dict = dict(n_systems=n_systems)
        system_names = []
        for uuid, val in state_shaped:
            system_name = UUID_TO_SYSTEM_NAME[int(uuid)]
            state_dict[system_name] = int(val)
            system_names.append(system_name)

        # Sync systems so that state size sanity check succeeds
        self._sync_systems(system_names=system_names)

        return state_dict, n_systems * 2 + 1


class Saturated(RelativeObjectState, BooleanStateMixin):
    def __init__(self, obj, default_limit=None):
        # Run super first
        super().__init__(obj=obj)

        # Limits
        self._default_limit = default_limit if default_limit is not None else m.DEFAULT_SATURATION_LIMIT
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
        return self._limits.get(system.name, self._default_limit)

    def set_limit(self, system, limit):
        """
        Sets internal particle limit @limit for @system

        Args:
            system (BaseSystem): System to limit
            limit (int): Number of particles representing limit for the given @system
        """
        self._limits[system.name] = limit

        # Add this object to the current state update set in its scene
        self.obj.state_updated()

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

        for system_name in self._limits.keys():
            system = self.obj.scene.get_system(system_name, force_init=False)
            if self.get_value(system):
                colors.append(system.color)

        if len(colors) == 0:
            # If no fluid system has Soaked=True, keep the default albedo value
            albedo_add = 0.0
            diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        else:
            albedo_add = 0.1
            avg_color = th.mean(th.stack(colors), dim=0)
            # Add a tint of avg_color
            # We want diffuse_tint to sum to 2.5 to result in the final RGB to sum to 1.5 on average
            # This is because an average RGB color sum to 1.5 (i.e. [0.5, 0.5, 0.5])
            # (0.5 [original avg RGB per channel] + 0.1 [albedo_add]) * 2.5 = 1.5
            diffuse_tint = th.tensor([0.5, 0.5, 0.5]) + avg_color / th.sum(avg_color)

        return albedo_add, diffuse_tint

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ModifiedParticles)
        return deps

    def _sync_systems(self, system_names):
        """
        Helper function for forcing internal systems to be synchronized with external list of @systems.

        NOTE: This may override internal state

        Args:
            system_names (list of str): List of system name(s) that should be actively tracked internally
        """
        self._limits = {system_name: m.DEFAULT_SATURATION_LIMIT for system_name in system_names}

    @property
    def state_size(self):
        # Limit per entry * 2 (UUID, value) + default limit + n limits
        return len(self._limits) * 2 + 2

    def _dump_state(self):
        state = dict(n_systems=len(self._limits), default_limit=self._default_limit)
        for system_name, limit in self._limits.items():
            state[system_name] = limit
        return state

    def _load_state(self, state):
        self._limits = dict()
        for k, v in state.items():
            if k == "n_systems":
                continue
            elif k == "default_limit":
                self._default_limit = v
            else:
                self._limits[k] = v

    def serialize(self, state):
        state_flat = th.tensor([state["n_systems"], state["default_limit"]], dtype=th.float32)
        if state["n_systems"] > 0:
            system_names = tuple(state.keys())[2:]
            state_flat = th.cat(
                [
                    state_flat,
                    th.cat(
                        [
                            th.tensor(
                                [self.obj.scene.get_system(system_name, force_init=False).uuid, state[system_name]],
                                dtype=th.float32,
                            )
                            for system_name in system_names
                        ]
                    ),
                ]
            )
        return state_flat

    def deserialize(self, state):
        n_systems = int(state[0])
        state_dict = dict(n_systems=n_systems, default_limit=int(state[1]))
        state_shaped = state[2 : 2 + n_systems * 2].reshape(-1, 2)
        system_names = []
        for uuid, val in state_shaped:
            system_name = UUID_TO_SYSTEM_NAME[int(uuid)]
            state_dict[system_name] = int(val)
            system_names.append(system_name)

        # Sync systems so that state size sanity check succeeds
        self._sync_systems(system_names=system_names)

        return state_dict, 2 + n_systems * 2
