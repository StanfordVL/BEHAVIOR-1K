import logging
from collections import OrderedDict, Iterable


class PrimRegistry:
    """
    Simple class for easily registering and tracking prim objects.

    Prims added are automatically organized by attributes specified by @unique_prim_keys and @group_prim_keys, and
    can be accessed at runtime by specifying the desired key and indexing value to grab the prim(s).

    Unique_prim_keys are 1-to-1 mappings: i.e.: a single indexing value will return a single prim.
        example: indexing by prim.name (every prim's name should be unique)
    Group_prim_keys are 1-to-many mappings: i.e.: a single indexing value will return a set of prims.
        example: indexing by prim.in_rooms (many prims can be in a single room)

    Note that if a prim's attribute is an array of values, then it will be stored under ALL of its values.
        example: prim.in_rooms = ["kitchen", "living_room"], indexing by in_rooms with a value of either kitchen OR
            living room will return this prim as part of its set!

    You can also easily check for membership in this registry, via either the prim's name OR the prim itself, e.g.:

        > prim.name in registry
        > prim in registry
    """
    def __init__(
            self,
            unique_prim_keys=None,
            group_prim_keys=None,
    ):
        """
        Args:
            unique_prim_keys (None or list of str): keys by which to reference a given prim. Any key should be a
                publically accessible attribute in a given prim (e.g.: prim.name)
                i.e.: these keys should map to a single prim

            group_prim_keys (None or list of str): keys by which to reference a group of prims, based on the key
                (e.g.: prim.room)
                i.e.: these keys can map to multiple prims

                e.g.: default is "name" key only, so we will store objects by their prim.name attribute
        """
        self.unique_prim_keys = set([] if unique_prim_keys is None else unique_prim_keys)
        self.group_prim_keys = set([] if group_prim_keys is None else group_prim_keys)

        # We always add in the "name" attribute as well
        self.unique_prim_keys.add("name")

        # Make sure there's no overlap between the unique and group keys
        assert len(self.unique_prim_keys.intersection(self.group_prim_keys)) == 0,\
            "Cannot create registry with unique and group prim keys that are the same!"

        # Create the ordered dicts programmatically
        for k in self.unique_prim_keys.union(self.group_prim_keys):
            self.__setattr__(f"_objects_by_{k}", OrderedDict())

    def add(self, prim):
        """
        Adds Prim @prim to this registry

        Args:
            prim (BasePrim): Prim to add to this registry
        """
        self._add(prim=prim, prim_keys=self.all_prim_keys)

    def _add(self, prim, prim_keys=None):
        """
        Same as self.add, but allows for selective @prim_keys for adding this prim to. Useful for internal things,
        such as internal updating of mappings

        Args:
            prim (BasePrim): Prim to add to this registry
            prim_keys (None or set or list of str): Which prim keys to use for adding the prim to mappings.
                None is default, which corresponds to all keys
        """
        prim_keys = self.all_prim_keys if prim_keys is None else prim_keys
        for k in prim_keys:
            prim_attr = prim.__getattribute__(k)
            # Standardize input as a list
            prim_attr = prim_attr if isinstance(prim_attr, Iterable) and not isinstance(prim_attr, str) else [prim_attr]

            # Loop over all values in this attribute and add to all mappings
            for attr in prim_attr:
                mapping = self.get_dict(k)
                if k in self.unique_prim_keys:
                    # Handle unique case
                    if attr in mapping:
                        logging.warning(f"Prim identifier '{k}' should be unique for adding to this registry mapping! Existing {k}: {attr}")
                        # Special case for "name" attribute, which should ALWAYS be unique
                        if k == "name":
                            logging.error(f"For name attribute, prims MUST be unique. Exiting.")
                            exit(-1)
                    mapping[attr] = prim
                else:
                    # Not unique case
                    # Possibly initialize list
                    if attr not in mapping:
                        mapping[attr] = set()
                    mapping[attr].add(prim)

    def remove(self, prim):
        """
        Removes prim @prim from this registry

        Args:
            prim (BasePrim): Prim to remove from this registry
        """
        for k in self.unique_prim_keys:
            self.get_dict(k).pop(prim.__getattribute__(k))
        for k in self.group_prim_keys:
            self.get_dict(k)[prim.__getattribute__(k)].remove(prim)

    def update(self, prim_keys=None):
        """
        Updates this registry, refreshing all internal mappings in case an object's value was updated

        Args:
            prim_keys (None or str or set or list of str): Which prim keys to update. None is default, which corresponds
                to all keys
        """
        prims = self.prims
        prim_keys = self.all_prim_keys if prim_keys is None else \
            (prim_keys if type(prim_keys) in {tuple, list} else [prim_keys])

        # Delete and re-create all prim_keys mappings
        for k in prim_keys:
            self.__delattr__(f"_objects_by_{k}")
            self.__setattr__(f"_objects_by_{k}", OrderedDict())

            # Iterate over all prims and re-populate the mappings
            for prim in prims:
                self._add(prim=prim, prim_keys=[k])

    def prim_is_registered(self, prim):
        """
        Check if a given prim @prim is registered

        Args:
            prim (Prim): Prim to check if it is internally registered
        """
        return prim in self.prims

    def get_dict(self, prim_key):
        """
        Specific mapping dictionary within this registry corresponding to the mappings of @prim_key.
            e.g.: if prim_key = "name", this will return the ordered dictionary mapping prim.name to prims

        Args:
            prim_key (str): Key with which to grab mapping dict from

        Returns:
            OrderedDict: Mapping from identifiers to prim(s) based on @prim_key
        """
        return self.__getattribute__(f"_objects_by_{prim_key}")

    def get_ids(self, prim_key):
        """
        All identifiers within this registry corresponding to the mappings of @prim_key.
            e.g.: if prim_key = "name", this will return all "names" stored internally that index into a prim
        Args:
            prim_key (str): Key with which to grab all identifiers from

        Returns:
            set: All identifiers within this registry corresponding to the mappings of @prim_key.
        """
        return set(self.get_dict(prim_key=prim_key).keys())

    @property
    def prims(self):
        """
        Get the prims in this registry

        Returns:
            list of Prim: Prims owned by this registry
        """
        return list(self.get_dict("name").values())

    @property
    def all_prim_keys(self):
        """
        Returns:
            set of str: All prim keys that are valid identification methods to index prim(s)
        """
        return self.unique_prim_keys.union(self.group_prim_keys)

    def __call__(self, prim_key, prim_value, default_val=None):
        """
        Grab the prim in this registry based on @prim_key and @prim_value

        Args:
            prim_key (str): What identification type to use to grab the requested prim.
                Should be one of @self.prim_keys.
            prim_value (any): Value to grab. Should be the value of your requested prim.<prim_key> attribute
            default_val (any): Default value to return if @prim_value is not found

        Returns:
            Prim: requested prim
        """
        assert prim_key in self.all_prim_keys,\
            f"Invalid prim_key requested! Valid options are: {self.all_prim_keys}, got: {prim_key}"

        return self.get_dict(prim_key).get(prim_value, default_val)

    def __contains__(self, prim):
        # Prim can be either a string OR the prim itself
        if isinstance(prim, str):
            prim = self("name", prim)
        return self.prim_is_registered(prim=prim)
