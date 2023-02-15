from collections import defaultdict
import inspect

import omnigibson as og
from omnigibson.object_states.object_state_base import RelativeObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.systems import FluidSystem
from omni.physx import get_physx_scene_query_interface


class ContactFluids(RelativeObjectState, KinematicsMixin):
    """
    Object state that handles contact checking between rigid bodies and individual fluid particles.
    """

    def __init__(self, obj):
        super().__init__(obj=obj)

    def _get_value(self, system, link=None):
        # Make sure system is valid
        assert issubclass(system, FluidSystem), "Can only get ContactFluids for a FluidSystem!"

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

        # Add margin, which is a slightly relaxed radius value
        lower -= system.particle_contact_offset * 1.01
        upper += system.particle_contact_offset * 1.01

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
        raise NotImplementedError("ContactFluids state currently does not support setting.")

    def cache_info(self, get_value_args):
        # Run super first
        info = super().cache_info(get_value_args=get_value_args)

        # Store the system's number of particles for each instancer
        for arg in get_value_args:
            if inspect.isclass(arg) and issubclass(arg, FluidSystem):
                info[arg] = arg.n_particles

        return info

    def _cache_is_valid(self, get_value_args):
        # Run super first
        is_valid = super()._cache_is_valid(get_value_args=get_value_args)

        # If it's valid, do final check with system
        if is_valid:
            for arg, info in self._cache[get_value_args]["info"].items():
                if inspect.isclass(arg) and issubclass(arg, FluidSystem):
                    # Make sure the number of particles in the system is the same; otherwise
                    # something has changed, so we need to update the cache internally
                    is_valid = arg.n_particles == info

        return is_valid
