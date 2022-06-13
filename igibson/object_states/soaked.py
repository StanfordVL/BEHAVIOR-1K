import numpy as np
from collections import OrderedDict
from igibson.systems.micro_particle_system import get_fluid_systems
from igibson.systems.system_base import get_system_from_element_name, get_element_name_from_system
import igibson.macros as m
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.water_source import WaterSource
from igibson.utils.python_utils import assert_valid_key
from igibson.utils.constants import PrimType
from omni.usd import get_shader_from_material
from pxr import Sdf


if m.ENABLE_OMNI_PARTICLES:
    from igibson.systems import SYSTEMS_REGISTRY


# Proportion of fluid particles per group required to intersect with this object for them all to be absorbed
# by a soakable object
PARTICLE_GROUP_PROPORTION = 0.7

# Soak threshold -- number of fluid particle required to be "absorbed" in order for the object to be
# considered soaked
SOAK_PARTICLE_THRESHOLD = 40

# TODO: SOOOOOOOOOOOO Hacky for cloth objects, because we can't use the normal way to detect particle collisions
# Update this once we have a better way!!


class Soaked(AbsoluteObjectState, BooleanState):
    def __init__(self, obj):
        super(Soaked, self).__init__(obj)

        # Map of fluid system to number of absorbed particles for this object corresponding
        # To that fluid system
        self.absorbed_particle_system_count = {}

        for system in get_fluid_systems().values():
            self.absorbed_particle_system_count[system] = 0

        self.absorbed_particle_threshold = SOAK_PARTICLE_THRESHOLD
        self.cloth_heuristic_update_fcn = lambda soaked_state, fluid_system: None              # Should update the absorbed particle count appropriately

    def _get_value(self, fluid_system):
        assert self.absorbed_particle_system_count[fluid_system] <= self.absorbed_particle_threshold
        return self.absorbed_particle_system_count[fluid_system] == self.absorbed_particle_threshold

    def _set_value(self, new_value, fluid_system):
        assert_valid_key(key=fluid_system, valid_keys=self.absorbed_particle_system_count, name="fluid element name")
        self.absorbed_particle_system_count[fluid_system] = self.absorbed_particle_threshold if new_value else 0
        return True

    def _update(self):
        # Handle cloth in a special way
        if self.obj._prim_type == PrimType.CLOTH:
            for fluid_system in self.absorbed_particle_system_count.keys():
                self.cloth_heuristic_update_fcn(self, fluid_system)

            return

        # Iterate over all fluid systems
        for fluid_system in self.absorbed_particle_system_count.keys():
            # fluid_name = get_element_name_from_system(fluid_system)
            # Map of obj_id -> (system, system_particle_id)
            particle_contacts = fluid_system.state_cache['particle_contacts']

            # For each particle hit, hide then add to total particle count of soaked object
            for instancer, particle_idxs in particle_contacts.get(self.obj, {}).items():
                assert self.absorbed_particle_threshold >= self.absorbed_particle_system_count[fluid_system]

                particles_to_absorb = min(len(particle_idxs), self.absorbed_particle_threshold - self.absorbed_particle_system_count[fluid_system])
                particle_idxs_to_absorb = particle_idxs[:particles_to_absorb]

                # Hide particles in contact with the object
                particle_visibilities = fluid_system.particle_instancers[instancer].particle_visibilities
                particle_visibilities[particle_idxs_to_absorb] = 0

                # Absorb the particles
                self.absorbed_particle_system_count[fluid_system] += particles_to_absorb

                # If above threshold, soak the object and stop absorbing
                if self.absorbed_particle_system_count[fluid_system] >= self.absorbed_particle_threshold:
                    break

        # TODO: remove this
        for system, count in self.absorbed_particle_system_count.items():
            assert count <= self.absorbed_particle_threshold, \
                f"{system.name} contains {count} particles violating threshold: {self.absorbed_particle_threshold}"

    def get_texture_change_params(self):
        albedo_add = 0.1
        colors = []

        for fluid_system in self.absorbed_particle_system_count.keys():
            if self.get_value(fluid_system):
                color = [0.0, 0.0, 1.0]
                colors.append(color)
            # TODO: Figure out how to read shader type
            # shader = get_shader_from_material(fluid_system.particle_material)
            # transmission_weight = shader.GetInput("enable_specular_transmission") * shader.GetInput("specular_transmission_weight")
            # total_weight = base_color_weight + transmission_weight
            # if total_weight == 0.0:
            #     # If the fluid doesn't have any color, we add a "blue" tint by default 
            #     color = np.array([0.0, 0.0, 1.0])
            # else:
            #     base_color_weight /= total_weight
            #     transmission_weight /= total_weight
            #     # Weighted sum of base color and transmission color
            #     color = base_color_weight * shader.GetInput("diffuse_reflection_color") + transmission_weight * (0.5 * shader.GetInput("specular_transmission_color") + 0.5 * shader.GetInput("specular_transmission_scattering_color"))

            # We want diffuse_tint to sum to 2.5 to keep the final sum of RGB to 1.5 on average: (0.5 (original average RGB) + 0.5 (albedo_add)) * 2.5 = 1.5 (original sum RGB)

        if len(colors) == 0:
            # If no fluid system has Soaked=True, keep the default albedo value
            albedo_add = 0.0
            diffuse_tint = [1.0, 1.0, 1.0]
        else:
            albedo_add = 0.1
            avg_color = np.mean(colors, axis=0)
            # Add a tint of avg_color
            diffuse_tint = np.array([0.5, 0.5, 0.5]) + avg_color / np.sum(avg_color)
            diffuse_tint = diffuse_tint.tolist()


        return albedo_add, diffuse_tint

    @property
    def settable(self):
        return True

    @property
    def state_size(self):
        return len(self.absorbed_particle_system_count)

    def _dump_state(self):
        state = OrderedDict()
        for system, val in self.absorbed_particle_system_count.items():
            state[get_element_name_from_system(system)] = val
        return state

    def _load_state(self, state):
        for system_name, val in state.items():
            self.absorbed_particle_system_count[get_system_from_element_name(system_name)] = val

    def _serialize(self, state):
        return np.stack(list(state.values()))

    def _deserialize(self, state):
        state_dict = OrderedDict()
        for i, system in enumerate(self.absorbed_particle_system_count.keys()):
            state_dict[get_element_name_from_system(system)] = state[i]

        return state_dict, len(self.absorbed_particle_system_count)

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]
