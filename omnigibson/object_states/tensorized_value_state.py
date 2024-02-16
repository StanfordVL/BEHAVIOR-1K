from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.update_state_mixin import GlobalUpdateStateMixin
from omnigibson.utils.python_utils import classproperty
import numpy as np


class TensorizedValueState(AbsoluteObjectState, GlobalUpdateStateMixin):
    """
    A state-mixin that implements optimized global value updates across all object state instances
    of this type, i.e.: all values across all object state instances are updated at once, rather than per
    individual instance update() call.
    """
    # Numpy array of raw internally tracked values
    # Shape is (N, ...), where the ith entry in the first dimension corresponds to the ith object state instance's value
    VALUES = None

    # Dictionary mapping object name to index in VALUES
    OBJ_IDXS = None

    # Dict of callbacks that can be added to when an object is removed
    CALLBACKS_ON_REMOVE = None

    @classmethod
    def global_initialize(cls):
        # Call super first
        super().global_initialize()

        # Initialize the global variables
        cls.VALUES = np.array([], dtype=cls.value_type).reshape(0, *cls.value_shape)
        cls.OBJ_IDXS = dict()
        cls.CALLBACKS_ON_REMOVE = dict()

    @classmethod
    def global_update(cls):
        # Call super first
        super().global_update()

        # This should be globally update all values. If there are no values, we skip by default since there is nothing
        # being tracked currently
        if len(cls.VALUES) == 0:
            return

        cls.VALUES = cls._update_values(values=cls.VALUES)

    @classmethod
    def global_clear(cls):
        # Call super first
        super().global_clear()

        # Clear internal state
        cls.VALUES = None
        cls.OBJ_IDXS = None
        cls.CALLBACKS_ON_REMOVE = None

    @classmethod
    def _update_values(cls, values):
        """
        Updates all internally tracked @values for this object state. Should be implemented by subclass.

        Args:
            values (np.array): Tensorized value array

        Returns:
            np.array: Updated tensorized value array
        """
        raise NotImplementedError

    @classmethod
    def _add_obj(cls, obj):
        """
        Adds object @obj to be tracked internally in @VALUES array.

        Args:
            obj (StatefulObject): Object to add
        """
        assert obj.name not in cls.OBJ_IDXS, \
            f"Tried to add object {obj.name} to the global tensorized value array but the object already exists!"

        # Add this object to the tracked global state
        cls.OBJ_IDXS[obj.name] = len(cls.VALUES)
        cls.VALUES = np.concatenate([cls.VALUES, np.zeros((1, *cls.value_shape), dtype=cls.value_type)], axis=0)

    @classmethod
    def _remove_obj(cls, obj):
        """
        Removes object @obj from the internally tracked @VALUES array.
        This also removes the corresponding tracking idx in @OBJ_IDXS

        Args:
            obj (StatefulObject): Object to remove
        """
        # Removes this tracked object from the global value array
        assert obj.name in cls.OBJ_IDXS, \
            f"Tried to remove object {obj.name} from the global tensorized value array but the object does not exist!"
        deleted_idx = cls.OBJ_IDXS.pop(obj.name)

        # Re-standardize the indices
        for i, name in enumerate(cls.OBJ_IDXS.keys()):
            cls.OBJ_IDXS[name] = i
        cls.VALUES = np.delete(cls.VALUES, [deleted_idx])

    @classmethod
    def add_callback_on_remove(cls, name, callback):
        """
        Adds a callback that will be triggered when @self.remove is called

        Args:
            name (str): Name of the callback to trigger
            callback (function): Function to execute. Should have signature callback(obj: BaseObject) --> None
        """
        cls.CALLBACKS_ON_REMOVE[name] = callback

    @classmethod
    def remove_callback_on_remove(cls, name):
        """
        Removes callback with name @name from the internal set of callbacks

        Args:
            name (str): Name of the callback to remove
        """
        cls.CALLBACKS_ON_REMOVE.pop(name)

    @classproperty
    def value_shape(cls):
        """
        Returns:
            tuple: Expected shape of the per-object state instance value. If empty (), this assumes
                that each entry is a single (non-array) value. Default is ()
        """
        return ()

    @classproperty
    def value_type(cls):
        """
        Returns:
            type: Type of the internal value array, e.g., bool, np.uint, float, etc. Default is float
        """
        return float

    @classproperty
    def value_name(cls):
        """
        Returns:
            str: Name of the value key to assign when dumping / loading the state. Should be implemented by subclass
        """
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        # Run super first
        super().__init__(*args, **kwargs)

        self._add_obj(obj=self.obj)

    def remove(self):
        # Execute all callbacks
        for callback in self.CALLBACKS_ON_REMOVE.values():
            callback(self.obj)

        # Removes this tracked object from the global value array
        self._remove_obj(obj=self.obj)

    def _get_value(self):
        # Directly access value from global register
        return self.value_type(self.VALUES[self.OBJ_IDXS[self.obj.name]])

    def _set_value(self, new_value):
        # Directly set value in global register
        self.VALUES[self.OBJ_IDXS[self.obj.name]] = new_value
        return True

    @property
    def state_size(self):
        # This is the flattened size of @self.value_shape
        # Note that np.product(()) returns 1, which is also correct for a non-arrayed value
        return int(np.product(self.value_shape))

    # For this state, we simply store its value.
    def _dump_state(self):
        return {self.value_name: self._get_value()}

    def _load_state(self, state):
        self._set_value(state[self.value_name])

    def _serialize(self, state):
        # If the state value is not an iterable, wrap it in a numpy array
        val = state[self.value_name] if isinstance(state[self.value_name], np.ndarray) else np.array([state[self.value_name]])
        return val.flatten().astype(float)

    def _deserialize(self, state):
        value_length = int(np.product(self.value_shape))
        value = state[:value_length].reshape(self.value_shape) if len(self.value_shape) > 0 else state[0]
        return {self.value_name: value}, value_length

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("TensorizedValueState")
        return classes
