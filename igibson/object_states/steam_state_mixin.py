from igibson.object_states.object_state_base import BaseObjectState


class SteamStateMixin(BaseObjectState):
    def __init__(self, obj):
        super(SteamStateMixin, self).__init__(obj)
        self.value = False

    def update_steam(self):
        # Assume only state evaluated True will need non-default texture.
        if self.get_value() != self.value:
            self.value = self.get_value()
            self.obj.set_emitter_temperature(self.value)
            self.obj.set_emitter_enabled(self.value)