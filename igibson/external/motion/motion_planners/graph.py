"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from collections import namedtuple, Mapping
from heapq import heappush, heappop


class Vertex(object):

    def __init__(self, value):
        self.value = value
        self.edges = []

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.value) + ')'


class Edge(object):

    def __init__(self, v1, v2, value, cost):
        self.v1, self.v2 = v1, v2
        self.v1.edges.append(self)
        self.value = value
        self.cost = cost

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.v1.value) + ' -> ' + str(self.v2.value) + ')'

SearchNode = namedtuple('SearchNode', ['cost', 'edge'])


class Graph(Mapping, object):

    def __init__(self):
        self.vertices = {}
        self.edges = []

    def __getitem__(self, value):
        return self.vertices[value]

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __call__(self, value1, value2):  # TODO - goal_test
        if value1 not in self or value2 not in self:
            return None

        start, goal = self[value1], self[value2]
        queue = [(0, start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            edge = nodes[v].edge
            if edge is None:
                return [v.value], []
            vertices, edges = retrace(edge.v1)
            return vertices + [v.value], edges + [edge.value]

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for edge in cv.edges:
                cost = nodes[cv].cost + edge.cost
                if edge.v2 not in nodes or cost < nodes[edge.v2].cost:
                    nodes[edge.v2] = SearchNode(cost, edge)
                    heappush(queue, (cost, edge.v2))
        return None

    def add(self, value):
        if value not in self:
            self.vertices[value] = Vertex(value)
        return self.vertices[value]

    def connect(self, value1, value2, edge_value=None, edge_cost=1):
        v1, v2 = self.add(value1), self.add(value2)
        edge = Edge(v1, v2, edge_value, edge_cost)
        self.edges.append(edge)
        return edge
