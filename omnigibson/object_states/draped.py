import torch as th

import omnigibson as og
from omnigibson.object_states.cloth_mixin import ClothStateMixin
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import sample_cloth_on_rigid


class Draped(RelativeObjectState, KinematicsMixin, BooleanStateMixin, ClothStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContactBodies)
        return deps

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("DrapedOver does not support set_value(False)")

        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("DrapedOver state requires obj1 is cloth and obj2 is rigid.")

        state = og.sim.dump_state(serialized=False)

        if sample_cloth_on_rigid(self.obj, other, randomize_xy=True) and self.get_value(other):
            return True
        else:
            og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        """
        Check whether the (cloth) object is draped on the other (rigid) object.
        The cloth object should touch the rigid object and its CoM should be below the average position of the contact points.
        """
        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Draped state requires obj1 is cloth and obj2 is rigid.")

        # Find the links of @other that are in contact with @self.obj
        contact_links = self.obj.states[ContactBodies].get_value() & set(other.links.values())
        if len(contact_links) == 0:
            return False
        contact_link_prim_paths = {contact_link.prim_path for contact_link in contact_links}

        # Filter the contact points to only include the ones that are on the contact links
        contact_positions = []
        for contact in self.obj.contact_list():
            if len({contact.body0, contact.body1} & contact_link_prim_paths) > 0:
                contact_positions.append(contact.position)

        # The center of mass of the cloth needs to be below the average position of the contact points
        mean_contact_position = th.mean(th.stack(contact_positions), dim=0)
        center_of_mass = th.mean(self.obj.root_link.keypoint_particle_positions, dim=0)
        return center_of_mass[2] < mean_contact_position[2]
