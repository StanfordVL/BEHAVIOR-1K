import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from omnigibson.systems.micro_particle_system import get_fluid_systems
from omnigibson.systems.system_base import get_system_from_element_name, get_element_name_from_system
from omnigibson.systems.macro_particle_system import VisualParticleSystem
from omnigibson.systems.micro_particle_system import FluidSystem
from omnigibson.macros import gm, create_module_macros
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.particle_remover import ParticleRemover
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.constants import PrimType
from pxr import Sdf
from omnigibson.systems import SYSTEMS_REGISTRY


# Create settings for this module
m = create_module_macros(module_path=__file__)


class Saturated(RelativeObjectState, BooleanState):
    def __init__(self, obj):
        # Run super init
        super().__init__(obj)

    def _get_value(self, system):
        return self.obj.states[ParticleRemover].check_saturation(system=system)

    def _set_value(self, system, new_value):
        assert_valid_key(key=system, valid_keys=ParticleRemover.supported_systems, name="particle system")
        # Only set the value if it's different than what currently exists
        if new_value != self.get_value(system):
            self.obj.states[ParticleRemover].set_value(new_value)
        return True

    def get_texture_change_params(self):
        albedo_add = 0.1
        colors = []

        for system in ParticleRemover.supported_systems:
            if self.get_value(system):
                if issubclass(system, VisualParticleSystem):
                    # If an texture of the particle exists, grab its mean color and set that
                    # Otherwise, set the RGB value
                    diffuse_texture = system.particle_object.material.diffuse_texture
                    color = plt.imread(diffuse_texture).mean(axis=(0, 1)) if diffuse_texture else system.particle_object.material.diffuse_color_constant
                elif issubclass(system, FluidSystem):
                    # Figure out the color for this particular fluid
                    mat = system.material
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
                else:
                    ParticleRemover.unsupported_system_error(system=system)
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

    @staticmethod
    def get_dependencies():
        return RelativeObjectState.get_dependencies() + [ParticleRemover]
