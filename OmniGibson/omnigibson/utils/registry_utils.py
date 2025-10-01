"""
A set of utility functions for registering and tracking objects
"""

from collections.abc import Iterable
from inspect import isclass

import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.utils.python_utils import Serializable, SerializableNonInstance, get_uuid
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.objects.object_base import BaseObject

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Token identifier for default values if a key doesn't exist in a given object
m.DOES_NOT_EXIST = "DOES_NOT_EXIST"

m.STABILIZE_SKIPPED_OBJECTS = False


class Registry:
    """
    Simple class for easily registering and tracking arbitrary objects of the same (or very similar) class types.

    Elements added are automatically organized by attributes specified by @unique_keys and @group_keys, and
    can be accessed at runtime by specifying the desired key and indexing value to grab the object(s).

    Default_key is a 1-to-1 mapping: i.e.: a single indexing value will return a single object.
        default: "name" -- indexing by object.name (i.e.: every object's name should be unique)
    Unique_keys are other 1-to-1 mappings: i.e.: a single indexing value will return a single object.
        example: indexing by object.name (every object's name should be unique)
    Group_keys are 1-to-many mappings: i.e.: a single indexing value will return a set of objects.
        example: indexing by object.in_rooms (many objects can be in a single room)

    Note that if a object's attribute is an array of values, then it will be stored under ALL of its values.
        example: object.in_rooms = ["kitchen", "living_room"], indexing by in_rooms with a value of either kitchen OR
            living room will return this object as part of its set!

    You can also easily check for membership in this registry, via either the object's name OR the object itself,
    e.g.:

        > object.name in registry
        > object in registry

        If the latter, note that default_key attribute will automatically be used to search for the object
    """

    def __init__(
        self,
        name,
        class_types=object,
        default_key="name",
        unique_keys=None,
        group_keys=None,
        default_value=None,
    ):
        """
        Args:
            name (str): name of this registry
            class_types (class or list of class): class expected for all entries in this registry. Default is `object`,
                meaning any object entered will be accepted. This is used to sanity check added entries using add()
                to make sure their type is correct (either that the entry itself is a valid class, or that they are an
                object of the valid class). Note that if a list of classes are passed, any one of the classes are
                considered a valid type for added objects
            default_key (str): default key by which to reference a given object. This key should be a
                publically accessible attribute in a given object (e.g.: object.name) and uniquely identify
                any entries
            unique_keys (None or list of str or set of str): keys by which to reference a given object. Any key should be a
                publically accessible attribute in a given object (e.g.: object.name)
                i.e.: these keys should map to a single object

            group_keys (None or list of str): keys by which to reference a group of objects, based on the key
                (e.g.: object.room)
                i.e.: these keys can map to multiple objects

                e.g.: default is "name" key only, so we will store objects by their object.name attribute

            default_value (any): Default value to use if the attribute @key does not exist in the object
        """
        self._name = name
        self.class_types = class_types if isinstance(class_types, Iterable) else [class_types]
        self.default_key = default_key
        self.unique_keys = set([] if unique_keys is None else unique_keys)
        self.group_keys = set([] if group_keys is None else group_keys)
        self.default_value = default_value

        # We always add in the "name" attribute as well
        self.unique_keys.add(self.default_key)

        # Make sure there's no overlap between the unique and group keys
        assert len(self.unique_keys.intersection(self.group_keys)) == 0, (
            f"Cannot create registry with unique and group object keys that are the same! "
            f"Unique keys: {self.unique_keys}, group keys: {self.group_keys}"
        )

        # Create the dicts programmatically
        for k in self.unique_keys.union(self.group_keys):
            self.__setattr__(f"_objects_by_{k}", dict())

        # Run super init
        super().__init__()

    @property
    def name(self):
        return self._name

    def add(self, obj):
        """
        Adds Instance @obj to this registry

        Args:
            obj (any): Instance to add to this registry
        """
        # Make sure that obj is of the correct class type
        assert any(
            [isinstance(obj, class_type) or issubclass(obj, class_type) for class_type in self.class_types]
        ), f"Added object must be either an instance or subclass of one of the following classes: {self.class_types}!"
        self._add(obj=obj, keys=self.all_keys)

    def _add(self, obj, keys=None):
        """
        Same as self.add, but allows for selective @keys for adding this object to. Useful for internal things,
        such as internal updating of mappings

        Args:
            obj (any): Instance to add to this registry
            keys (None or set or list of str): Which object keys to use for adding the object to mappings.
                None is default, which corresponds to all keys
        """
        keys = self.all_keys if keys is None else keys

        for k in keys:
            obj_attr = self._get_obj_attr(obj=obj, attr=k)
            # Standardize input as a list
            obj_attr = obj_attr if isinstance(obj_attr, Iterable) and not isinstance(obj_attr, str) else [obj_attr]

            # Loop over all values in this attribute and add to all mappings
            for attr in obj_attr:
                mapping = self.get_dict(k)
                if k in self.unique_keys:
                    # Handle unique case
                    if attr in mapping:
                        log.warning(
                            f"Instance identifier '{k}' should be unique for adding to this registry mapping! Existing {k}: {attr}"
                        )
                        # Special case for "name" attribute, which should ALWAYS be unique
                        assert k != "name", "For name attribute, objects MUST be unique."
                    mapping[attr] = obj
                else:
                    # Not unique case
                    # Possibly initialize list
                    if attr not in mapping:
                        mapping[attr] = set()
                    mapping[attr].add(obj)

    def remove(self, obj):
        """
        Removes object @object from this registry

        Args:
            obj (any): Instance to remove from this registry
        """
        # Iterate over all keys
        for k in self.all_keys:
            # Grab the attribute from the object
            obj_attr = self._get_obj_attr(obj=obj, attr=k)
            # Standardize input as a list
            obj_attr = obj_attr if isinstance(obj_attr, Iterable) and not isinstance(obj_attr, str) else [obj_attr]

            # Loop over all values in this attribute and remove them from all mappings
            for attr in obj_attr:
                mapping = self.get_dict(k)
                if k in self.unique_keys:
                    # Handle unique case -- in this case, we just directly pop the value from the dictionary
                    mapping.pop(attr)
                else:
                    # Not unique case
                    # We remove a value from the resulting set
                    mapping[attr].remove(obj)

    def clear(self):
        """
        Removes all owned objects from this registry
        """
        # Re-create the owned dicts programmatically
        for k in self.unique_keys.union(self.group_keys):
            self.__setattr__(f"_objects_by_{k}", dict())

    def update(self, keys=None):
        """
        Updates this registry, refreshing all internal mappings in case an object's value was updated

        Args:
            keys (None or str or set or list of str): Which object keys to update. None is default, which corresponds
                to all keys
        """
        objects = self.objects
        keys = self.all_keys if keys is None else (keys if type(keys) in {tuple, list} else [keys])

        # Delete and re-create all keys mappings
        for k in keys:
            self.__delattr__(f"_objects_by_{k}")
            self.__setattr__(f"_objects_by_{k}", dict())

            # Iterate over all objects and re-populate the mappings
            for obj in objects:
                self._add(obj=obj, keys=[k])

    def object_is_registered(self, obj):
        """
        Check if a given object @object is registered

        Args:
            obj (any): Instance to check if it is internally registered
        """
        return obj in self.objects

    def get_dict(self, key):
        """
        Specific mapping dictionary within this registry corresponding to the mappings of @key.
            e.g.: if key = "name", this will return the dictionary mapping object.name to objects

        Args:
            key (str): Key with which to grab mapping dict from

        Returns:
            dict: Mapping from identifiers to object(s) based on @key
        """
        return getattr(self, f"_objects_by_{key}")

    def get_ids(self, key):
        """
        All identifiers within this registry corresponding to the mappings of @key.
            e.g.: if key = "name", this will return all "names" stored internally that index into a object
        Args:
            key (str): Key with which to grab all identifiers from

        Returns:
            set: All identifiers within this registry corresponding to the mappings of @key.
        """
        return set(self.get_dict(key=key).keys())

    def _get_obj_attr(self, obj, attr):
        """
        Grabs object's @obj's attribute @attr. Additionally checks to see if @obj is a class or a class instance, and
        uses the correct logic

        Args:
            obj (any): Object to grab attribute from
            attr (str): String name of the attribute to grab

        Return:
            any: Attribute @k of @obj
        """
        # We try to grab the object's attribute, and if it fails we fallback to the default value
        try:
            val = getattr(obj, attr)

        except:
            val = self.default_value if self.default_value is not None else m.DOES_NOT_EXIST

        return val

    @property
    def objects(self):
        """
        Get the objects in this registry

        Returns:
            list of any: Instances owned by this registry
        """
        return list(self.get_dict(self.default_key).values())

    @property
    def object_names(self):
        """
        Get the names of the objects in this registry

        Returns:
            set of str: Names of the instances owned by this registry
        """
        return {obj.name for obj in self.objects}

    @property
    def all_keys(self):
        """
        Returns:
            set of str: All object keys that are valid identification methods to index object(s)
        """
        return self.unique_keys.union(self.group_keys)

    def __call__(self, key, value, default_val=None):
        """
        Grab the object in this registry based on @key and @value

        Args:
            key (str): What identification type to use to grab the requested object(s).
                Should be one of @self.all_keys.
            value (any): Value to grab. Should be the value of your requested object.<key> attribute
            default_val (any): Default value to return if @value is not found

        Returns:
            any or set of any: requested unique object if @key is one of unique_keys, else a set if
                @key is one of group_keys
        """
        assert key in self.all_keys, f"Invalid key requested! Valid options are: {self.all_keys}, got: {key}"

        return self.get_dict(key).get(value, default_val)

    def __contains__(self, obj):
        # Instance can be either a string (default key) OR the object itself
        if isinstance(obj, str):
            obj = self(self.default_key, obj)
        return self.object_is_registered(obj=obj)


class SerializableRegistry(Registry, Serializable):
    """
    Registry that is serializable, i.e.: entries contain states that can themselves be serialized /deserialized.

    Note that this assumes that any objects added to this registry are themselves of @Serializable type!
    """

    def __init__(
        self,
        name,
        class_types=object,
        default_key="name",
        hash_key="uuid",
        unique_keys=None,
        group_keys=None,
        default_value=None,
    ):
        """
        Args:
            name (str): name of this registry
            class_types (class or list of class): class expected for all entries in this registry. Default is `object`,
                meaning any object entered will be accepted. This is used to sanity check added entries using add()
                to make sure their type is correct (either that the entry itself is a valid class, or that they are an
                object of the valid class). Note that if a list of classes are passed, any one of the classes are
                considered a valid type for added objects
            default_key (str): default key by which to reference a given object. This key should be a
                publically accessible attribute in a given object (e.g.: object.name) and uniquely identify
                any entries
            hash_key (str): key by which to reference a given object when serializing / deserializing its state.
                This key should be a publically accessible attribute in a given object (e.g.: object.name) and
                uniquely identify any entries via a hash value (i.e.: integer)
            unique_keys (None or list of str or set of str): keys by which to reference a given object. Any key should be a
                publically accessible attribute in a given object (e.g.: object.name)
                i.e.: these keys should map to a single object

            group_keys (None or list of str): keys by which to reference a group of objects, based on the key
                (e.g.: object.room)
                i.e.: these keys can map to multiple objects

                e.g.: default is "name" key only, so we will store objects by their object.name attribute

            default_value (any): Default value to use if the attribute @key does not exist in the object
        """
        # Store hash key
        self.hash_key = hash_key

        # We always add in the hash_key attribute as well
        unique_keys = set() if unique_keys is None else set(unique_keys)
        unique_keys.add(self.hash_key)

        # Set the default dump and load filters, which is a pass-through
        self._dump_filter = lambda obj: True
        self._load_filter = lambda obj: True

        # Run super
        super().__init__(
            name=name,
            class_types=class_types,
            default_key=default_key,
            unique_keys=unique_keys,
            group_keys=group_keys,
            default_value=default_value,
        )

    def add(self, obj):
        # In addition to any other class types, we make sure that the object is a serializable instance / class
        validate_class = issubclass if isclass(obj) else isinstance
        assert any(
            [validate_class(obj, class_type) for class_type in (Serializable, SerializableNonInstance)]
        ), "Added object must be either an instance or subclass of Serializable or SerializableNonInstance!"
        # Run super like normal
        super().add(obj=obj)

    def set_dump_filter(self, dump_filter):
        """
        Sets the internal filter that determines whether an object should be dumped or not.

        Args:
            dump_filter (function): Function that determines whether an object should be dumped or not.
                Expected signature is:

                def dump_filter(obj) -> bool

                where it takes in a given registered object @obj and returns True if the object should have its state
                dumped
        """
        self._dump_filter = dump_filter

    def set_load_filter(self, load_filter):
        """
        Sets the internal filter that determines whether an object's state should be loaded or not.

        Args:
            load_filter (function): Function that determines whether an object should have its state loaded or not
                Expected signature is:

                def load_filter(obj) -> bool

                where it takes in a given registered object @obj and returns True if the object should have its
                state loaded or not
        """
        self._load_filter = load_filter

    def _dump_state(self):
        # Iterate over all objects and grab their states
        state = dict()
        for obj in self.objects:
            if self._dump_filter(obj):
                state[obj.name] = obj.dump_state(serialized=False)
        return state

    def _load_state(self, state):
        # Iterate over all objects and load their states. Currently the objects and the state don't have to match, i.e.
        # there might be objects in the scene that do not appear in the state dict (a warning will be printed), or
        # the state might contain additional information about objects that are NOT in the scene. For both cases, state
        # loading will be skipped.
        for obj in self.objects:
            if self._load_filter(obj):
                if obj.name not in state:
                    log.debug(f"Object '{obj.name}' is not in the state dict to load from. Skip loading its state.")
                    if m.STABILIZE_SKIPPED_OBJECTS and isinstance(obj, BaseObject) and not obj.kinematic_only:
                        obj.keep_still()
                    continue
                obj.load_state(state[obj.name], serialized=False)

    def serialize(self, state):
        # Iterate over the entire dict and flatten
        # We keep track of how many objects are being saved, as well as the unique identifier for each object so that
        # the saved flattened array is agnostic to object ordering
        # self("name", name) grabs the corresponding object from this registry so that we can serialize its state
        # Each object's state in the flattened array is composed of [hash_key, serialized_state]
        n_objs = len(state)

        if len(state) > 0:
            state_flat = []
            for name in state:
                obj = self("name", name)
                serialized = obj.serialize(state[name])

                # Handle the case where serialize returns an empty tensor
                if serialized.numel() == 0:
                    hash_key_tensor = th.tensor([getattr(obj, self.hash_key)], dtype=th.float32)
                    state_flat.append(hash_key_tensor)
                else:
                    hash_key_tensor = th.tensor([getattr(obj, self.hash_key)], dtype=serialized.dtype)
                    state_flat.append(th.cat([hash_key_tensor, serialized.flatten()]))

            state_flat = th.cat(state_flat)
        else:
            state_flat = th.tensor([])

        return th.cat([th.tensor([n_objs], dtype=th.float32), state_flat])

    def deserialize(self, state):
        state_dict = dict()
        # Iterate over all the objects and deserialize their individual states, incrementing the index counter
        # along the way
        n_objects = int(state[0])
        idx = 1
        for _ in range(n_objects):
            # Infer obj based on UUID
            obj = self(self.hash_key, int(state[idx]))
            assert (
                obj is not None
            ), f"Could not find object while deserializing with hash_key {self.hash_key}: {int(state[idx])}"
            idx += 1
            log.debug(f"obj: {obj.name}, idx: {idx}, passing in state length: {len(state[idx:])}")
            # We pass in the entire remaining state vector, assuming the object only parses the relevant states
            # at the beginning
            state_dict[obj.name], deserialized_items = obj.deserialize(state[idx:])
            idx += deserialized_items
        return state_dict, idx

    @property
    def uuid(self):
        """
        Returns:
            int: Unique hashed ID for this registry
        """
        return get_uuid(self.name)
