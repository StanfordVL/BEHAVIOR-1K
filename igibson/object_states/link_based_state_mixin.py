

# from igibson.external.pybullet_tools.utils import get_link_state, link_from_name


class LinkBasedStateMixin(object):
    def __init__(self):
        super(LinkBasedStateMixin, self).__init__()

        # This cannot be decided yet due to the MRO. Corrected in initialize_link_mixin.
        self.is_point_link = None

        self.body_id = None

        # This is only used if we're looking for a point link.
        self.inertial_transforms = None

        # This is only used if we're looking for a geometry-based (URDF) link.
        self.link_id = None

    @staticmethod
    def get_state_link_name():
        raise ValueError("LinkBasedState child should specify link name by overriding get_state_link_name.")

    def initialize_link_mixin(self):
        assert not self._initialized

        self.is_point_link = hasattr(self.obj, "meta_links") and self.get_state_link_name() in self.obj.meta_links

        # Get the body id. Note that this annotation only works with single-body objects.
        body_ids = self.obj.get_body_ids()
        assert len(body_ids) == 1, "LinkBasedStateMixin expects object to be single-body."
        self.body_id = body_ids[0]

        if self.is_point_link:
            # For point links, compute the necessary transforms in advance.
            dynamics_info = p.getDynamicsInfo(self.body_id, -1)
            inertial_pos = dynamics_info[3]
            inertial_orn = dynamics_info[4]
            self.inertial_transforms = p.invertTransform(inertial_pos, inertial_orn)
        else:
            # For URDF links, get the link ID too.
            try:
                self.link_id = link_from_name(self.body_id, self.get_state_link_name())
            except ValueError:
                return False

        return True

    def get_link_position(self):
        if self.is_point_link:
            # Get the point link offset and transform it based on the object info.
            point_link_offset = self.obj.meta_links[self.get_state_link_name()] * self.obj.scale

            obj_pos, obj_orn = self.obj.get_position_orientation()
            com_pos, com_orn = p.multiplyTransforms(obj_pos, obj_orn, *self.inertial_transforms)
            return p.multiplyTransforms(com_pos, com_orn, point_link_offset, [0, 0, 0, 1])[0]
        else:
            # The necessary link is not found
            if self.link_id is None:
                return

            return get_link_state(self.body_id, self.link_id).linkWorldPosition
