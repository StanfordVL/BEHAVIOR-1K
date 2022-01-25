import itertools



from igibson.object_states.factory import get_state_name
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.objects.object_base import BaseObject


class ObjectGrouper(BaseObject):
    """A multi-object wrapper that groups multiple objects and applies operations to all of them in parallel."""

    class ProceduralMaterialAggregator(object):
        """A fake procedural material that updates the procedural material of all ObjectGrouper objects."""

        def __init__(self, object_grouper):
            self.object_grouper = object_grouper

        def update(self):
            for obj in self.object_grouper.objects:
                obj.procedural_material.update()

    class BaseStateAggregator(object):
        """A fake state that aggregates state between ObjectGrouper objects and propagates updates."""

        def __init__(self, state_type, object_grouper):
            self.state_type = state_type
            self.object_grouper = object_grouper

        def update(self):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].update()

    class AbsoluteStateAggregator(BaseStateAggregator):
        def get_value(self):
            if not issubclass(self.state_type, BooleanState):
                raise ValueError("Aggregator can only aggregate boolean states.")

            return all(obj.states[self.state_type].get_value() for obj in self.object_grouper.objects)

        def set_value(self, new_value):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].set_value(new_value)

        # We declare a list subtype here so that we can identify dumps produced through ObjectGrouper's
        # dump function vs. dumps produced otherwise, so that we can identify one-to-many loads vs.
        # many-to-many loads.
        class GroupedStateDump(list):
            pass

        # Since we don't inherit from ObjectStateBase, we don't implement _dump()
        # but dump() directly.
        def dump(self):
            return ObjectGrouper.AbsoluteStateAggregator.GroupedStateDump(
                obj.states[self.state_type].dump() for obj in self.object_grouper.objects
            )

        def load(self, data):
            if isinstance(data, ObjectGrouper.AbsoluteStateAggregator.GroupedStateDump):
                # We're loading a many-to-many data dump.
                assert len(data) == len(self.object_grouper.objects)

                for obj, dump in zip(self.object_grouper.objects, data):
                    obj.states[self.state_type].load(dump)
            else:
                # We're loading a one-to-many data dump.
                # An example of when this happens is when an object goes from non-sliced to
                # sliced. In the Sliceable setter, we dump the state of the non-sliced object
                # (which will be a single object) and we then attempt to load it into the
                # ObjectGrouper representing the halves.
                for obj in self.object_grouper.objects:
                    obj.states[self.state_type].load(data)

    class RelativeStateAggregator(BaseStateAggregator):
        def get_value(self, other):
            if not issubclass(self.state_type, BooleanState):
                raise ValueError("Aggregator can only aggregate boolean states.")

            return all(obj.states[self.state_type].get_value(other) for obj in self.object_grouper.objects)

        def set_value(self, other, new_value):
            for obj in self.object_grouper.objects:
                obj.states[self.state_type].set_value(other, new_value)

    def __init__(self, objects_with_pose_offsets):
        objects = [obj for obj, _ in objects_with_pose_offsets]
        pose_offsets = [trans for _, trans in objects_with_pose_offsets]
        assert objects and all(isinstance(obj, BaseObject) for obj in objects)
        self.objects = objects

        # Pose offsets are the transformation of the object parts to the whole
        # object in base link frame
        self.pose_offsets = pose_offsets

        state_types = [obj.states.keys() for obj in self.objects]
        if state_types.count(state_types[0]) != len(state_types):
            raise ValueError("Grouped objects have different states.")
        state_types = state_types[0]

        self.states = {
            state_type: ObjectGrouper.AbsoluteStateAggregator(state_type, self)
            if issubclass(state_type, AbsoluteObjectState)
            else ObjectGrouper.RelativeStateAggregator(state_type, self)
            for state_type in state_types
        }

        self.procedural_material = (
            ObjectGrouper.ProceduralMaterialAggregator(self)
            if self.objects[0].procedural_material is not None
            else None
        )

    def __getattr__(self, item):
        # Check if the attr is the same for everything
        attrs = [getattr(obj, item) for obj in self.objects]

        # If the attribute is a method, let's return a wrapper function that calls the method
        # on each object.
        if callable(attrs[0]):

            def grouped_function(*args, **kwargs):
                rets = [getattr(obj, item)(*args, **kwargs) for obj in self.objects]
                if rets.count(rets[0]) != len(rets):
                    raise ValueError("Methods on grouped objects had different results.")

                return rets[0]

            return grouped_function

        # These attributes are used during object import and should return
        # the concatenation results of all objects in self.objects
        if item in ["visual_mesh_to_material", "link_name_to_vm", "body_ids", "is_fixed"]:
            return list(itertools.chain.from_iterable(attrs))

        # Otherwise, check that it's the same for everyone and then just return the value.
        if attrs.count(attrs[0]) != len(attrs):
            raise ValueError("Grouped objects had different values for this attribute.")

        return attrs[0]

    def _load(self, simulator):
        body_ids = []
        for obj in self.objects:
            body_ids += obj.load(simulator)
        return body_ids

    def get_body_ids(self):
        ids = []
        for obj in self.objects:
            if obj.get_body_ids() is not None:
                ids.extend(obj.get_body_ids())
        if len(ids) == 0:
            ids = None
        return ids

    def get_position(self):
        raise ValueError("Cannot get_position on ObjectGrouper")

    def get_orientation(self):
        raise ValueError("Cannot get_orientation on ObjectGrouper")

    def get_position_orientation(self):
        raise ValueError("Cannot get_position_orientation on ObjectGrouper")

    def set_position(self, pos):
        raise ValueError("Cannot set_position on ObjectGrouper")

    def set_orientation(self, orn):
        raise ValueError("Cannot set_orientation on ObjectGrouper")

    def set_position_orientation(self, pos, orn):
        raise ValueError("Cannot set_position_orientation on ObjectGrouper")

    def set_base_link_position_orientation(self, pos, orn):
        for obj, (part_pos, part_orn) in zip(self.objects, self.pose_offsets):
            new_pos, new_orn = p.multiplyTransforms(pos, orn, part_pos, part_orn)
            obj.set_base_link_position_orientation(new_pos, new_orn)

    def dump_state(self):
        return {
            get_state_name(state_type): state_instance.dump()
            for state_type, state_instance in self.states.items()
            if issubclass(state_type, AbsoluteObjectState)
        }

    def load_state(self, dump):
        for state_type, state_instance in self.states.items():
            if issubclass(state_type, AbsoluteObjectState):
                state_instance.load(dump[get_state_name(state_type)])


class ObjectMultiplexer(BaseObject):
    """A multi-object wrapper that acts as a proxy for the selected one between the set of objects it contains."""

    def __init__(self, name, multiplexed_objects, current_index):
        assert multiplexed_objects and all(isinstance(obj, BaseObject) for obj in multiplexed_objects)
        assert 0 <= current_index < len(multiplexed_objects)

        for obj in multiplexed_objects:
            obj.multiplexer = self

        self._multiplexed_objects = multiplexed_objects

        self.current_index = current_index
        self.name = name
        categories = set(x.category for x in multiplexed_objects)
        assert len(categories) == 1, "All multiplexed objects should be the same category."
        self.category = categories.pop()

    def set_selection(self, idx):
        assert 0 <= idx < len(self._multiplexed_objects)
        self.current_index = idx

    def current_selection(self):
        return self._multiplexed_objects[self.current_index]

    def __getattr__(self, item):
        # This attribute is used during object import and should return
        # the concatenation results of all objects in self._multiplexed_objects
        if item in ["visual_mesh_to_material", "link_name_to_vm"]:
            attrs = [getattr(obj, item) for obj in self._multiplexed_objects]
            return list(itertools.chain.from_iterable(attrs))

        return getattr(self.current_selection(), item)

    def _load(self, simulator):
        body_ids = []
        for obj in self._multiplexed_objects:
            body_ids += obj.load(simulator)
        return body_ids

    def get_body_ids(self):
        return self.current_selection().get_body_ids()

    def get_position(self):
        return self.current_selection().get_position()

    def get_orientation(self):
        return self.current_selection().get_orientation()

    def get_position_orientation(self):
        return self.current_selection().get_position_orientation()

    def set_position(self, pos):
        return self.current_selection().set_position(pos)

    def set_orientation(self, orn):
        return self.current_selection().set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        return self.current_selection().set_position_orientation(pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        return self.current_selection().set_base_link_position_orientation(pos, orn)

    def dump_state(self):
        return {
            "current_index": self.current_index,
            "sub_states": [obj.dump_state() for obj in self._multiplexed_objects],
        }

    def load_state(self, dump):
        self.current_index = dump["current_index"]

        assert len(dump) == len(self._multiplexed_objects)
        for obj, obj_dump in zip(self._multiplexed_objects, dump["sub_states"]):
            obj.load_state(obj_dump)
