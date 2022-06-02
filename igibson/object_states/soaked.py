import numpy as np
from collections import OrderedDict
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.object_states.water_source import WaterSource

# TODO: Have BooleanState automatically set the dump / load / serialize / deserialize functions
class Soaked(AbsoluteObjectState, BooleanState, TextureChangeStateMixin):
    def __init__(self, obj):
        super(Soaked, self).__init__(obj)
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True

    def _update(self):
        # TODO!
        return
        # water_source_objs = self.simulator.scene.get_objects_with_state(WaterSource)
        # for water_source_obj in water_source_objs:
        #     contacted_water_prim_paths = set(
        #         item.bodyUniqueIdB for item in list(self.obj.states[ContactBodies].get_value())
        #     )
        #     for particle in water_source_obj.states[WaterSource].water_stream.get_active_particles():
        #         if not set(particle.link_prim_paths).isdisjoint(contacted_water_prim_paths):
        #             self.value = Tru
        # self.update_texture()

    @property
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(soaked=self.value)

    def _load_state(self, state):
        self.value = state["soaked"]

    def _serialize(self, state):
        return np.array([float(state["soaked"])])

    def _deserialize(self, state):
        return OrderedDict(soaked=(state[0] == 1.0)), 1

    @staticmethod
    def get_optional_dependencies():
        return [WaterSource]