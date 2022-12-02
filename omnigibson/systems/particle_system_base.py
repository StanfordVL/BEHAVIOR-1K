from omnigibson.utils.python_utils import classproperty
from omnigibson.systems.system_base import BaseSystem


class BaseParticleSystem(BaseSystem):
    """
    Global system for modeling particles, e.g.: dirt, water, etc.
    """

    @classproperty
    def n_particles(cls):
        """
        Returns:
            int: Number of active particles in this system
        """
        raise NotImplementedError()

    @classproperty
    def color(cls):
        """
        Returns:
            3-array: (R,G,B) color of the particles generated from this system
        """
        raise NotImplementedError()
