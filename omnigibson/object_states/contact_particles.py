from collections import defaultdict
import inspect

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import RelativeObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.systems import PhysicalParticleSystem
from omni.physx import get_physx_scene_query_interface

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.POSITIONAL_VALIDATION_EPSILON = 1e-10


class ContactParticles(RelativeObjectState, KinematicsMixin):
    """
    Object state that handles contact checking between rigid bodies and individual particles.
    """

    def __init__(self, obj):
        super().__init__(obj=obj)

    def _get_value(self, system, link=None):
        # Make sure system is valid
        assert issubclass(system, PhysicalParticleSystem), "Can only get ContactParticles for a PhysicalParticleSystem!"

        # Create contacts dictionary, mapping instancer to set of particle IDs in contact
        contacts = defaultdict(set)

        # Variables to update mid-iteration
        inst = None
        idx = 0

        # Define callback function to use for omni's overlap_sphere() call
        def report_hit(hit):
            nonlocal link, inst, idx
            link_name = None if link is None else link.prim_path.split("/")[-1]
            base, body = "/".join(hit.rigid_body.split("/")[:-1]), hit.rigid_body.split("/")[-1]
            continue_traversal = True
            # If no links are specified, then we assume checking contact with any link owned by this object
            # Otherwise, we check for exact match of link name
            if (link is None and base == self.obj.prim_path) or (link is not None and link_name == body):
                # Add to contacts and terminate early
                contacts[inst].add(idx)
                continue_traversal = False
            return continue_traversal

        # Grab the relaxed AABB of this object or its link for coarse filtering of particles to ignore checking
        lower, upper = self.obj.states[AABB].get_value() if link is None else link.aabb

        # Add margin for filtering inbound
        lower = lower - (system.particle_contact_offset + 0.001)
        upper = upper + (system.particle_contact_offset + 0.001)

        # Iterate over all instancers and aggregate contacts
        for inst in system.particle_instancers.values():
            positions = inst.particle_positions
            # Only check positions that are within the relaxed AABB of this object
            inbound_idxs = ((lower < positions) & (positions < upper)).all(axis=-1).nonzero()[0]
            for idx in inbound_idxs:
                get_physx_scene_query_interface().overlap_sphere(system.particle_contact_offset, positions[idx], report_hit, False)

        # Return contacts
        return contacts

    def _set_value(self, system, new_value):
        raise NotImplementedError("ContactParticles state currently does not support setting.")

    # TODO: investigate whether this caching actually makes things faster because we hypothesize that it will be very
    # rare for all the particles to be still.
    # def cache_info(self, get_value_args):
    #     # Run super first
    #     info = super().cache_info(get_value_args=get_value_args)
    #
    #     # Store the system's particle positions for each instancer
    #     for arg in get_value_args:
    #         if inspect.isclass(arg) and issubclass(arg, PhysicalParticleSystem):
    #             info[arg] = {instancer: instancer.particle_positions for instancer in arg.particle_instancers.values()}
    #
    #     return info
    #
    # def _cache_is_valid(self, get_value_args):
    #     # Run super first
    #     is_valid = super()._cache_is_valid(get_value_args=get_value_args)
    #
    #     if not is_valid:
    #         return False
    #
    #     for arg, info in self._cache[get_value_args]["info"].items():
    #         if inspect.isclass(arg) and issubclass(arg, PhysicalParticleSystem):
    #             # TODO: adopt the has_changed mechanism in object_state_base
    #             # Check if the particle positions have changed
    #
    #             # If the instancers don't match, return False
    #             if list(arg.particle_instancers.values()) != list(info.keys()):
    #                 return False
    #
    #             # If there are no instancers, skip
    #             if len(info.keys()) == 0:
    #                 continue
    #
    #             arg_pos= np.vstack([instancer.particle_positions for instancer in arg.particle_instancers.values()])
    #             info_pos = np.vstack([particle_positions for particle_positions in info.values()])
    #
    #             # If any of the particles moved, return False
    #             if np.any(np.linalg.norm(arg_pos - info_pos, axis=1) >= m.POSITIONAL_VALIDATION_EPSILON):
    #                 return False
    #
    #     return True
