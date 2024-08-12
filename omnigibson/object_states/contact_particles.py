import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import RelativeObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Distance tolerance for detecting contact
m.CONTACT_AABB_TOLERANCE = 2.5e-2
m.CONTACT_TOLERANCE = 5e-3


class ContactParticles(RelativeObjectState, KinematicsMixin):
    """
    Object state that handles contact checking between rigid bodies and individual particles.
    """

    def _get_value(self, system, link=None):
        """
        Args:
            system (PhysicalParticleSystem): System whose contact particle info should be aggregated
            link (None or RigidPrim): If specified, the specific link to check for particles' contact

        Returns:
            set of int: Set of particle IDs in contact
        """
        # Make sure system is valid
        assert self.obj.scene.is_physical_particle_system(
            system_name=system.name
        ), "Can only get ContactParticles for a PhysicalParticleSystem!"

        # Variables to update mid-iteration
        contacts = set()
        idx = 0

        # Define callback function to use for omni's overlap_sphere() call
        def report_hit(hit):
            nonlocal link, idx
            link_name = None if link is None else link.prim_path.split("/")[-1]
            base, body = "/".join(hit.rigid_body.split("/")[:-1]), hit.rigid_body.split("/")[-1]
            continue_traversal = True
            # If no links are specified, then we assume checking contact with any link owned by this object
            # Otherwise, we check for exact match of link name
            if (link is None and base == self.obj.prim_path) or (link is not None and link_name == body):
                # Add to contacts and terminate early
                contacts.add(idx)
                continue_traversal = False
            return continue_traversal

        # Grab the relaxed AABB of this object or its link for coarse filtering of particles to ignore checking
        lower, upper = self.obj.states[AABB].get_value() if link is None else link.visual_aabb

        # Add margin for filtering inbound
        lower = lower - (system.particle_radius + m.CONTACT_AABB_TOLERANCE)
        upper = upper + (system.particle_radius + m.CONTACT_AABB_TOLERANCE)

        # Iterate over all particles and aggregate contacts
        positions = system.get_particles_position_orientation()[0]
        # Only check positions that are within the relaxed AABB of this object
        inbound_idxs = ((lower < positions) & (positions < upper)).all(dim=-1).nonzero()
        dist = system.particle_contact_radius + m.CONTACT_TOLERANCE
        for idx in inbound_idxs:
            og.sim.psqi.overlap_sphere(dist, positions[idx.item()].cpu().numpy(), report_hit, False)

        # Return contacts
        return contacts

    def _set_value(self, system, new_value):
        raise NotImplementedError("ContactParticles state currently does not support setting.")

    def _cache_is_valid(self, get_value_args):
        # Cache is never valid since particles always change poses
        return False
