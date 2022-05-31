from igibson.utils.python_utils import classproperty
from igibson.systems.system_base import BaseSystem


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
