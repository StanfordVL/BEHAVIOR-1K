from igibson.macros import gm
from igibson.object_states.fluid_sink import FluidSink
from igibson.systems.micro_particle_system import WaterSystem


class WaterSink(FluidSink):

    @property
    def fluid_system(self):
        return WaterSystem

    @staticmethod
    def get_state_link_name():
        return "water_sink_link"
