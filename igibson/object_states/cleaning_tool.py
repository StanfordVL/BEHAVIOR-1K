# from igibson.external.pybullet_tools.utils import aabb_contains_point, get_aabb
from igibson.object_states.aabb import AABB
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.dirty import Dusty, Stained
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.soaked import Soaked
from igibson.object_states.toggle import ToggledOn
from igibson.objects.particles import Dust, Stain
from igibson.utils.usd_utils import BoundingBoxAPI

_LINK_NAME = "cleaning_tool_area"


class CleaningTool(AbsoluteObjectState, LinkBasedStateMixin):
    def __init__(self, obj):
        super(CleaningTool, self).__init__(obj)

    @staticmethod
    def get_state_link_name():
        return _LINK_NAME

    def _initialize(self):
        self.initialize_link_mixin()

    def _update(self):
        # TODO!
        return
        # # Check if this tool interacts with any dirt particles.
        # for particle_system in self.simulator.particle_systems:
        #     # We don't check for inheritance, just the leaf types.
        #     # Faster and doesn't need access to the private type _Dirt.
        #     particle_type = type(particle_system)
        #     if particle_type is not Dust and particle_type is not Stain:
        #         continue
        #
        #     # Check if the surface has any particles.
        #     if not particle_system.get_num_active():
        #         continue
        #
        #     # We need to be soaked to clean stains.
        #     if isinstance(particle_system, Stain):
        #         if Soaked not in self.obj.states or not self.obj.states[Soaked].get_value():
        #             continue
        #
        #     # Check if we're touching the parent of the particle system through our
        #     # cleaning link.
        #     contact_bodies = self.obj.states[ContactBodies].get_value()
        #     touching_body = [
        #         cb for cb in contact_bodies if cb.body1 in particle_system.parent_obj.link_prim_paths
        #     ]
        #     touching_link = any(self.link is None or cb.body0 == self.link.prim_path for cb in touching_body)
        #     if not touching_link:
        #         continue
        #
        #     # Time to check for colliding particles in our AABB.
        #     if self.link is not None:
        #         # If we have a cleaning link, use it.
        #         aabb = BoundingBoxAPI.compute_aabb(self.link.prim_path())
        #     else:
        #         # Otherwise, use the full-object AABB.
        #         aabb = self.obj.states[AABB].get_value()
        #
        #     # Find particles in the AABB.
        #     for particle in particle_system.get_active_particles():
        #         pos = particle.get_position()
        #         if BoundingBoxAPI.aabb_contains_point(pos, aabb):
        #             particle_system.stash_particle(particle)

    def _set_value(self, new_value):
        raise ValueError("Cannot set valueless state CleaningTool.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [AABB]

    @staticmethod
    def get_optional_dependencies():
        return AbsoluteObjectState.get_optional_dependencies() + [Dusty, Stained, Soaked, ToggledOn, ContactBodies]
