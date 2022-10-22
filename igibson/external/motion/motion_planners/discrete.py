"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from collections import deque
from heapq import heappop, heappush

from recordclass import recordclass
import numpy as np

from .utils import INF

Node = recordclass('Node', ['g', 'parent'])


def retrace(visited, q):
    if q is None:
        return []
    return retrace(visited, visited[tuple(q)].parent) + [q]


def bfs(start, goal, neighbors, collision, max_iterations=INF):
    if collision(start) or collision(goal):
        return None
    iterations = 0
    visited = {tuple(start): Node(0, None)}
    expanded = []
    queue = deque([start])
    while len(queue) != 0 and iterations < max_iterations:
        current = queue.popleft()
        iterations += 1
        expanded.append(current)
        if goal is not None and tuple(current) == tuple(goal):
            return retrace(visited, current)
        for next in neighbors(current):
            # TODO - make edges for real (and store bad edges)
            if tuple(next) not in visited and not collision(next):
                visited[tuple(next)] = Node(
                    next, visited[tuple(current)].g + 1, current)
                queue.append(next)
    return None


def astar(start, goal, distance, neighbors, collision,
          max_iterations=INF, cost=lambda g, h: g + h):  # TODO - put start and goal in neighbors
    if collision(start) or collision(goal):
        return None
    queue = [(cost(0, distance(start, goal)), 0, start)]
    visited = {tuple(start): Node(0, None)}
    iterations = 0
    while len(queue) != 0 and iterations < max_iterations:
        _, current_g, current = heappop(queue)
        current = np.array(current)
        if visited[tuple(current)].g != current_g:
            continue
        iterations += 1
        if tuple(current) == tuple(goal):
            return retrace(visited, current)
        for next in neighbors(current):
            next_g = current_g + distance(current, next)
            if (tuple(next) not in visited or next_g < visited[tuple(next)].g) and not collision(next):
                visited[tuple(next)] = Node(next_g, current)
                # ValueError: The truth value of an array with more than one
                # element is ambiguous.
                heappush(queue, (cost(next_g, distance(next, goal)), next_g, next))
    return None
