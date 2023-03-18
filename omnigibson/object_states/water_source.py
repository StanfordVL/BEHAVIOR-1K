from omnigibson.object_states.fluid_source import FluidSource
from omnigibson.systems import get_system

class WaterSource(FluidSource):
    @property
    def fluid_system(self):
        return get_system("water")

    @property
    def n_particles_per_group(self):
        return 5

    @property
    def n_steps_per_group(self):
        return 5

    @staticmethod
    def get_state_link_name():
        return "watersource_link"
