import numpy as np
from collections import OrderedDict
from omnigibson.systems.micro_particle_system import get_fluid_systems
from omnigibson.systems.system_base import get_system_from_element_name, get_element_name_from_system
from omnigibson.macros import gm, create_module_macros
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.water_source import WaterSource
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.constants import PrimType
from pxr import Sdf
from omnigibson.systems import SYSTEMS_REGISTRY


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Proportion of fluid particles per group required to collide with this object for them all to be absorbed
m.PARTICLE_GROUP_PROPORTION = 0.7

# Soak threshold -- number of fluid particle required to be "absorbed" in order for the object to be considered soaked
m.SOAK_PARTICLE_THRESHOLD = 40


class Soaked(RelativeObjectState, BooleanState):
    def __init__(self, obj):
        super(Soaked, self).__init__(obj)

        # Map of fluid system to number of absorbed particles for this object corresponding
        # To that fluid system
        self.absorbed_particle_system_count = {}

        for system in get_fluid_systems().values():
            self.absorbed_particle_system_count[system] = 0


    def _get_value(self, fluid_system):
        assert self.absorbed_particle_system_count[fluid_system] <= m.SOAK_PARTICLE_THRESHOLD
        return self.absorbed_particle_system_count[fluid_system] == m.SOAK_PARTICLE_THRESHOLD

    def _set_value(self, fluid_system, new_value):
        assert_valid_key(key=fluid_system, valid_keys=self.absorbed_particle_system_count, name="fluid element name")
        self.absorbed_particle_system_count[fluid_system] = m.SOAK_PARTICLE_THRESHOLD if new_value else 0
        return True

    def _update(self):
        # Iterate over all fluid systems
        for fluid_system in self.absorbed_particle_system_count.keys():
            # We should never "over-absorb" particles from any fluid system above the threshold
            assert self.absorbed_particle_system_count[fluid_system] <= m.SOAK_PARTICLE_THRESHOLD, \
                f"over-absorb particles from {fluid_system.name}"

            # If we already reach the absorption limit, continue
            if self.absorbed_particle_system_count[fluid_system] == m.SOAK_PARTICLE_THRESHOLD:
                continue

            # Dict[Object, Dict[instancer_name, List[particle_idxs]]
            instancer_name_to_particle_idxs = {}
            if self.obj.prim_type == PrimType.RIGID:
                # The fluid system caches contact information for each of its particles with rigid bodies
                particle_contacts = fluid_system.state_cache['particle_contacts']
                instancer_name_to_particle_idxs = particle_contacts.get(self.obj, {})
            elif self.obj.prim_type == PrimType.CLOTH:
                # Scene query interface overlap sphere API doesn't work for cloth objects, so no contact will be cached.
                # A reasonable heuristics is to detect if the fluid particles lie within the AABB of the object
                aabb_low, aabb_high = self.obj.states[AABB].get_value()
                for instancer_name, instancer in fluid_system.particle_instancers.items():
                    inbound = ((aabb_low < instancer.particle_positions) & (instancer.particle_positions < aabb_high))
                    inbound_idxs = inbound.all(axis=1).nonzero()[0]
                    instancer_name_to_particle_idxs[instancer_name] = inbound_idxs
            else:
                raise ValueError(f"Unknown prim type {self.obj.prim_type} when handling Soaked state.")

            # For each particle hit, hide then add to total particle count of soaked object
            for instancer, particle_idxs in instancer_name_to_particle_idxs.items():
                max_particle_absorbed = m.SOAK_PARTICLE_THRESHOLD - self.absorbed_particle_system_count[fluid_system]
                particles_to_absorb = min(len(particle_idxs), max_particle_absorbed)
                particle_idxs_to_absorb = list(particle_idxs)[:particles_to_absorb]

                # Hide particles that have been absorbed
                particle_visibilities = fluid_system.particle_instancers[instancer].particle_visibilities
                particle_visibilities[particle_idxs_to_absorb] = 0

                # Keep track of the particles that have been absorbed
                self.absorbed_particle_system_count[fluid_system] += particles_to_absorb

                # If above threshold, soak the object and stop absorbing
                if self.absorbed_particle_system_count[fluid_system] == m.SOAK_PARTICLE_THRESHOLD:
                    break

    def get_texture_change_params(self):
        albedo_add = 0.1
        colors = []

        for fluid_system in self.absorbed_particle_system_count.keys():
            if self.get_value(fluid_system):
                # Figure out the color for this particular fluid
                mat = fluid_system.material
                base_color_weight = mat.diffuse_reflection_weight
                transmission_weight = mat.enable_specular_transmission * mat.specular_transmission_weight
                total_weight = base_color_weight + transmission_weight
                if total_weight == 0.0:
                    # If the fluid doesn't have any color, we add a "blue" tint by default
                    color = np.array([0.0, 0.0, 1.0])
                else:
                    base_color_weight /= total_weight
                    transmission_weight /= total_weight
                    # Weighted sum of base color and transmission color
                    color = base_color_weight * mat.diffuse_reflection_color + \
                            transmission_weight * (0.5 * mat.specular_transmission_color + \
                                                   0.5 * mat.specular_transmission_scattering_color)
                colors.append(color)

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
