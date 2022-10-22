"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from random import random
import numpy as np
from .utils import irange, argmin, RRT_ITERATIONS


class TreeNode(object):

    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    #def retrace(self):
    #    if self.parent is None:
    #        return [self]
    #    return self.parent.retrace() + [self]

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env, color=(1, 0, 0, .5)):
        from manipulation.primitives.display import draw_node, draw_edge
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return 'TreeNode(' + str(self.config) + ')'
    __repr__ = __str__


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


def rrt(start, goal_sample, distance, sample, extend, collision, goal_test=lambda q: False, iterations=RRT_ITERATIONS, goal_probability=.2):
    #goal_test = lambda q: np.linalg.norm(q - goal_sample) < 0.5
    if collision(start):
        return None
    if not callable(goal_sample):
        g = goal_sample
        goal_sample = lambda: g
    nodes = [TreeNode(start)]
    for i in irange(iterations):
        goal = random() < goal_probability or i == 0
        s = goal_sample() if goal else sample()

        last = argmin(lambda n: distance(n.config, s), nodes)
        for q in extend(last.config, s):
            if collision(q):
                break
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if np.linalg.norm(np.array(last.config) - goal_sample()) < 0.5:#goal_test(last.config):
                return configs(last.retrace())
        else:
            if goal:
                return configs(last.retrace())
    return None
