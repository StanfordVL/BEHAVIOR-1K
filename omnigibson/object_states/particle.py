import torch as th

from omnigibson.object_states.object_state_base import BaseObjectRequirement


class ParticleRequirement(BaseObjectRequirement):
    """
    Class for sanity checking objects that requires particle systems
    """

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        from omnigibson.macros import gm

        if not gm.USE_GPU_DYNAMICS:
            return False, f"Particle systems are not enabled when GPU dynamics is off."

        return True, None
