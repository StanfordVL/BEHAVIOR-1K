"""
Default settings used throughout iGibson. These are generally magic numbers that were tuned heuristically.

Use get_settings() to return all the settings used in iGibson.
"""
from addict import Dict

# Initialize settings
settings = Dict()

# Now we can start filling it!
settings.object_states.dirty.CLEAN_THRESHOLD = 0.5
settings.object_states.dirty.FLOOR_CLEAN_THRESHOLD = 0.75
settings.object_states.dirty.MIN_PARTICLES_FOR_SAMPLING_SUCCESS = 5

# TODO: Fill this in from the rest of the states

