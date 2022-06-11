from igibson.object_states.aabb import AABB
from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.dirty import Dusty, Stained
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.soaked import Soaked
from igibson.object_states.toggle import ToggledOn
from igibson.utils.usd_utils import BoundingBoxAPI
from igibson.systems.macro_particle_system import MacroParticleSystem, StainSystem

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
        for system in self._simulator.scene.systems:
            if not issubclass(system, MacroParticleSystem):
                continue

            # Check if the system has any particles.
            if system.n_particles == 0:
                continue

            # We need to be soaked to clean stains.
            if system == StainSystem:
                if Soaked not in self.obj.states or not self.obj.states[Soaked].get_value():
                    continue

            # Time to check for colliding particles in our AABB.
            if self.link is not None:
                # If we have a cleaning link, use it.
                aabb = BoundingBoxAPI.compute_aabb(self.link.prim_path())
            else:
                # Otherwise, use the full-object AABB.
                aabb = self.obj.states[AABB].get_value()

            # A very lenient relaxed AABB is used
            lower, upper = aabb
            lower -= 0.05
            upper += 0.05
            relaxed_aabb = aabb

            # Find particles in the relaxed AABB.
            for particle_name in list(system.particles.keys()):
                particle = system.particles[particle_name]
                pos = particle.get_position()
                if BoundingBoxAPI.aabb_contains_point(pos, relaxed_aabb):
                    system.remove_particle(particle_name)

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
