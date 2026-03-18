from omnigibson.object_states.object_state_base import BaseObjectRequirement


class SliceableRequirement(BaseObjectRequirement):
    """
    Class for sanity checking objects that request the "sliceable" ability
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Avoid circular imports
        from omnigibson.objects.dataset_object import DatasetObject

        # Make sure object is dataset object
        if not isinstance(obj, DatasetObject):
            return False, f"Only compatible with DatasetObject, but {obj} is of type {type(obj)}"
        # Check to make sure object parts are properly annotated in this object's metadata
        if not obj.metadata["object_parts"]:
            return False, "Missing required metadata 'object_parts'."

        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Check to make sure object parts are properly annotated in this object's metadata
        metadata = prim.GetCustomData().get("metadata", dict())
        if not metadata.get("object_parts", None):
            return False, "Missing required metadata 'object_parts'."

        return True, None
