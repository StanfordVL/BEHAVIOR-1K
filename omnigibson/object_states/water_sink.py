from omnigibson.object_states.fluid_sink import FluidSink
from omnigibson.systems import get_system


class WaterSink(FluidSink):

    @property
    def fluid_system(self):
        return get_system("water")

    @staticmethod
    def get_state_link_name():
        return "watersink_link"
