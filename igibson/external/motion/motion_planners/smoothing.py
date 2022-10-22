"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from random import randint
import numpy as np


def smooth_path(path, extend, collision, iterations=50):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[:i + 1] + \
                shortcut + smoothed_path[j + 1:]
    return smoothed_path


def optimize_path(path, extend, collision, iterations=50):
    def cost_fn(l):
        s = 0
        for i in range(len(l)-1):
            s += np.sqrt((l[i][0] - l[i+1][0])**2 + (l[i][1] - l[i+1][1])**2)
        return s
    # smoothed_paths = []
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = randint(0, len(smoothed_path) - 1)
        j = randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        # print('short cut cost', cost_fn(shortcut),
        #       'original cost:', cost_fn(smoothed_path[i:j]))
        if (cost_fn(shortcut) < cost_fn(smoothed_path[i:j])) and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[:i + 1] + \
                shortcut + smoothed_path[j + 1:]
            # smoothed_paths.append(np.copy(smoothed_path))
    # return smoothed_paths
    return smoothed_path

# TODO: sparsify path to just waypoints
