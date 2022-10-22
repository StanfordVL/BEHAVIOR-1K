"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from collections import Mapping
from random import random

from .rrt import TreeNode, configs
from .utils import irange, argmin, pairs, randomize, take, enum

ts = enum('ALL', 'SUCCESS', 'PATH', 'NONE')

# TODO - resample and use nearest neighbors when the tree is large
# TODO - possible bug if a node is already in the tree


class MultiTree(Mapping, object):

    def __init__(self, start, distance, sample, extend, collision):
        self.nodes = {}
        self.distance = distance
        self.sample = sample
        self.extend = extend
        self.collision = collision
        self.add(TreeNode(start))

    def add(self, *nodes):
        for n in nodes:
            self.nodes[n.config] = n

    def __getitem__(self, q):
        return self.nodes[q]
    #  return first(lambda v: self.distance(v.config, q) < 1e-6, self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        for n in self.nodes.values():
            yield n

    def __call__(self, q1, q2=None, iterations=50):
        if q1 in self:
            path1 = self[q1].retrace()
        else:
            path1 = self.grow(q1, iterations=iterations)
        if q2 is None:
            return configs(path1)
        if q2 in self:
            path2 = self[q2].retrace()
        else:
            path2 = self.grow(q2, iterations=iterations)
        if path1 is None or path2 is None:
            return None

        for i in range(min(len(path1), len(path2))):
            if path1[i] != path2[i]:
                break
        else:
            i += 1
        return configs(path1[:i - 1:-1] + path2[i - 1:])

    def clear(self):
        for n in self:
            n.clear()

    def draw(self, env):
        for n in self:
            n.draw(env)


class MultiRRT(MultiTree):

    def grow(self, goal_sample, iterations=50, goal_probability=.2, store=ts.PATH, max_tree_size=500):
        if not callable(goal_sample):
            goal_sample = lambda: goal_sample
        nodes, new_nodes = list(
            take(randomize(self.nodes.values()), max_tree_size)), []
        for i in irange(iterations):
            goal = random() < goal_probability or i == 0
            s = goal_sample() if goal else self.sample()

            last = argmin(lambda n: self.distance(
                n.config, s), nodes + new_nodes)
            for q in self.extend(last.config, s):
                if self.collision(q):
                    break
                last = TreeNode(q, parent=last)
                new_nodes.append(last)
            else:
                if goal:
                    path = last.retrace()
                    if store in [ts.ALL, ts.SUCCESS]:
                        self.add(*new_nodes)
                    elif store == ts.PATH:
                        new_nodes_set = set(new_nodes)
                        self.add(*[n for n in path if n in new_nodes_set])
                    return path
        if store == ts.ALL:
            self.add(*new_nodes)
        return None


class MultiBiRRT(MultiTree):

    def grow(self, goal, iterations=50, store=ts.PATH, max_tree_size=500):
        if goal in self:
            return self[goal].retrace()
        if self.collision(goal):
            return None
        nodes1, new_nodes1 = list(
            take(randomize(self.nodes.values()), max_tree_size)), []
        nodes2, new_nodes2 = [], [TreeNode(goal)]
        for _ in irange(iterations):
            if len(nodes1) + len(new_nodes1) > len(nodes2) + len(new_nodes2):
                nodes1, nodes2 = nodes2, nodes1
                new_nodes1, new_nodes2 = new_nodes2, new_nodes1

            s = self.sample()
            last1 = argmin(lambda n: self.distance(
                n.config, s), nodes1 + new_nodes1)
            for q in self.extend(last1.config, s):
                if self.collision(q):
                    break
                last1 = TreeNode(q, parent=last1)
                new_nodes1.append(last1)

            last2 = argmin(lambda n: self.distance(
                n.config, last1.config), nodes2 + new_nodes2)
            for q in self.extend(last2.config, last1.config):
                if self.collision(q):
                    break
                last2 = TreeNode(q, parent=last2)
                new_nodes2.append(last2)
            else:
                if len(nodes1) == 0:
                    nodes1, nodes2 = nodes2, nodes1
                    new_nodes1, new_nodes2 = new_nodes2, new_nodes1
                    last1, last2 = last2, last1

                path1, path2 = last1.retrace(), last2.retrace()[:-1][::-1]
                for p, n in pairs(path2):
                    n.parent = p
                if len(path2) == 0:  # TODO - still some kind of circular error
                    for n in new_nodes2:
                        if n.parent == last2:
                            n.parent = path1[-1]
                else:
                    path2[0].parent = path1[-1]
                path = path1 + path2

                if store in [ts.ALL, ts.SUCCESS]:
                    self.add(*(new_nodes1 + new_nodes2[:-1]))
                elif store == ts.PATH:
                    new_nodes_set = set(new_nodes1 + new_nodes2[:-1])
                    self.add(*[n for n in path if n in new_nodes_set])
                return path
        if store == ts.ALL:
            self.add(*new_nodes1)
        return None
