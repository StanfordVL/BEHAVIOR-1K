from omnigibson.object_states.object_state_base import BaseObjectRequirement
from omnigibson.macros import gm


class ParticleRequirement(BaseObjectRequirement):
    """
    Class for sanity checking objects that requires particle systems
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        if not gm.USE_GPU_DYNAMICS:
            return False, "Particle systems are not enabled when GPU dynamics is off."

        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # No actual requirement on the asset side.
        return True, None
