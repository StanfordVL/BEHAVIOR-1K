import igibson.macros as m
from igibson.systems.system_base import SYSTEMS_REGISTRY

# Only import omni-related particle systems if we're enabling them
if m.ENABLE_OMNI_PARTICLES:
    from igibson.systems.micro_particle_system import *

from igibson.systems.macro_particle_system import *
