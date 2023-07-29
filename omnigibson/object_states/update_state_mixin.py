from omnigibson.object_states.object_state_base import BaseObjectState


class UpdateStateMixin(BaseObjectState):
    """
    A state-mixin that allows for per-sim-step updates via the update() call
    """
    def update(self):
        """
        Updates the object state. This function will be called for every simulator step
        """
        assert self._initialized, "Cannot update uninitialized state."
        return self._update()

    def _update(self):
        """
        This function will be called once for every simulator step. Must be implemented by subclass.
        """
        # Explicitly raise not implemented error to avoid silent bugs -- update should never be called otherwise
        raise NotImplementedError()
