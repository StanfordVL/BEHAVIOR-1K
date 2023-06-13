import numpy as np
from omnigibson.object_states.aabb import AABB

def get_center_extent(obj_states):
    assert AABB in obj_states
    aabb = obj_states[AABB].get_value()
    center, extent = get_aabb_center(aabb), get_aabb_extent(aabb)
    return center, extent

def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.

def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)