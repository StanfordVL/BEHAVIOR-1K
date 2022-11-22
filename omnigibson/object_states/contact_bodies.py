from omnigibson.object_states.object_state_base import AbsoluteObjectState


class ContactBodies(AbsoluteObjectState):
    def _get_value(self):
        return self.obj.contact_list()

    def _set_value(self, new_value):
        raise NotImplementedError("ContactBodies state currently does not support setting.")
