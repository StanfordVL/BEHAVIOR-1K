"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from collections import Mapping

#class StarRoadmap(Mapping, object):
class StarRoadmap(Mapping, object):

    def __init__(self, center, planner):
        self.center = center
        self.planner = planner
        self.roadmap = {}

    """
    def __getitem__(self, q):
        return self.roadmap[q]

    def __len__(self):
        return len(self.roadmap)

    def __iter__(self):
        return iter(self.roadmap)
    """

    def grow(self, goal):
        if goal not in self.roadmap:
            self.roadmap[goal] = self.planner(self.center, goal)
        return self.roadmap[goal]

    def __call__(self, start, goal):
        start_traj = self.grow(start)
        if start_traj is None:
            return None
        goal_traj = self.grow(goal)
        if goal_traj is None:
            return None
        return start_traj.reverse(), goal_traj
