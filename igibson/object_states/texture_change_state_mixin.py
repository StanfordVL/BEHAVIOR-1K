from igibson.object_states.object_state_base import BaseObjectState


class TextureChangeStateMixin(BaseObjectState):
    def __init__(self, obj):
        super(TextureChangeStateMixin, self).__init__(obj)
        self.value = False

    def update_texture(self):
        # Assume only state evaluated True will need non-default texture.
        if self.get_value() != self.value:
            self.value = self.get_value()
            self.obj.update_textures_for_state(self.__class__, self.value)