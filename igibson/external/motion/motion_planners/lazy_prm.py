"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
from scipy.spatial.kdtree import KDTree
from heapq import heappush, heappop
from collections import namedtuple
from omnigibson.external.motion.motion_planners.utils import INF, elapsed_time
from omnigibson.external.motion.motion_planners.rrt_connect import direct_path
from omnigibson.external.motion.motion_planners.smoothing import smooth_path

import random
import time
import numpy as np

Node = namedtuple('Node', ['g', 'parent'])
def unit_cost_fn(v1, v2): return 1.
def zero_heuristic_fn(v): return 0


def retrace_path(visited, vertex):
    if vertex is None:
        return []
    return retrace_path(visited, visited[vertex].parent) + [vertex]


def dijkstra(start_v, neighbors_fn, cost_fn=unit_cost_fn):
    # Update the heuristic over time
    start_g = 0
    visited = {start_v: Node(start_g, None)}
    queue = [(start_g, start_v)]
    while queue:
        current_g, current_v = heappop(queue)
        if visited[current_v].g < current_g:
            continue
        for next_v in neighbors_fn(current_v):
            next_g = current_g + cost_fn(current_v, next_v)
            if (next_v not in visited) or (next_g < visited[next_v].g):
                visited[next_v] = Node(next_g, current_v)
                heappush(queue, (next_g, next_v))
    return visited


def wastar_search(start_v, end_v, neighbors_fn, cost_fn=unit_cost_fn,
                  heuristic_fn=zero_heuristic_fn, w=1, max_cost=INF, max_time=INF):
    # TODO: lazy wastar to get different paths
    #heuristic_fn = lambda v: cost_fn(v, end_v)
    def priority_fn(g, h): return g + w*h
    def goal_test(v): return v == end_v

    start_time = time.time()
    start_g, start_h = 0, heuristic_fn(start_v)
    visited = {start_v: Node(start_g, None)}
    queue = [(priority_fn(start_g, start_h), start_g, start_v)]
    while queue and (elapsed_time(start_time) < max_time):
        _, current_g, current_v = heappop(queue)
        if visited[current_v].g < current_g:
            continue
        if goal_test(current_v):
            return retrace_path(visited, current_v)
        for next_v in neighbors_fn(current_v):
            next_g = current_g + cost_fn(current_v, next_v)
            if (next_v not in visited) or (next_g < visited[next_v].g):
                visited[next_v] = Node(next_g, current_v)
                next_h = heuristic_fn(next_v)
                if priority_fn(next_g, next_h) < max_cost:
                    heappush(queue, (priority_fn(next_g, next_h), next_g, next_v))
    return None


def check_path(path, colliding_vertices, colliding_edges, samples, extend_fn, collision_fn):
    # TODO: bisect order
    vertices = list(path)
    random.shuffle(vertices)
    for v in vertices:
        if v not in colliding_vertices:
            colliding_vertices[v] = collision_fn(samples[v])
        if colliding_vertices[v]:
            return False

    edges = list(zip(path, path[1:]))
    random.shuffle(edges)
    for v1, v2 in edges:
        if (v1, v2) not in colliding_edges:
            segment = list(extend_fn(samples[v1], samples[v2]))
            random.shuffle(segment)
            colliding_edges[v1, v2] = any(map(collision_fn, segment))
            colliding_edges[v2, v1] = colliding_edges[v1, v2]
        if colliding_edges[v1, v2]:
            return False
    return True


def lazy_prm(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, num_samples=100, max_degree=10,
             weights=None, p_norm=2, max_distance=INF, approximate_eps=0.0,
             max_cost=INF, max_time=INF, max_paths=INF):
    # TODO: multi-query motion planning
    start_time = time.time()
    # TODO: can embed pose and/or points on the robot for other distances
    if weights is None:
        weights = np.ones(len(start_conf))

    def embed_fn(q): return weights * q
    # def distance_fn(q1, q2): return np.linalg.norm(
    #     embed_fn(q2) - embed_fn(q1), ord=p_norm)

    def cost_fn(v1, v2): return distance_fn(samples[v1], samples[v2])
    # TODO: can compute cost between waypoints from extend_fn

    samples = []
    while len(samples) < num_samples:
        conf = sample_fn()
        if (distance_fn(start_conf, conf) + distance_fn(conf, end_conf)) < max_cost:
            samples.append(conf)

    start_index, end_index = 0, 1
    samples[start_index] = start_conf
    samples[end_index] = end_conf

    embedded = list(map(embed_fn, samples))
    kd_tree = KDTree(embedded)
    vertices = list(range(len(samples)))
    edges = set()
    for v1 in vertices:
        # TODO: could dynamically compute distances
        distances, neighbors = kd_tree.query(embedded[v1], k=max_degree+1, eps=approximate_eps,
                                             p=p_norm, distance_upper_bound=max_distance)
        for d, v2 in zip(distances, neighbors):
            if (d < max_distance) and (v1 != v2):
                edges.update([(v1, v2), (v2, v1)])
    neighbors_from_index = {v: set() for v in vertices}
    for v1, v2 in edges:
        neighbors_from_index[v1].add(v2)
    #print(time.time() - start_time, len(edges), float(len(edges))/len(samples))

    colliding_vertices, colliding_edges = {}, {}

    def neighbors_fn(v1):
        for v2 in neighbors_from_index[v1]:
            if not (colliding_vertices.get(v2, False) or
                    colliding_edges.get((v1, v2), False)):
                yield v2

    visited = dijkstra(end_index, neighbors_fn, cost_fn)
    def heuristic_fn(v): return visited[v].g if v in visited else INF
    while elapsed_time(start_time) < max_time:
        # TODO: extra cost to prioritize reusing checked edges
        path = wastar_search(start_index, end_index, neighbors_fn=neighbors_fn,
                             cost_fn=cost_fn, heuristic_fn=heuristic_fn,
                             max_cost=max_cost, max_time=max_time-elapsed_time(start_time))
        if path is None:
            return None, samples, edges, colliding_vertices, colliding_edges
        cost = sum(cost_fn(v1, v2) for v1, v2 in zip(path, path[1:]))
        # print('Length: {} | Cost: {:.3f} | Vertices: {} | Edges: {} | Time: {:.3f}'.format(
        #     len(path), cost, len(colliding_vertices), len(colliding_edges), elapsed_time(start_time)))
        if check_path(path, colliding_vertices, colliding_edges, samples, extend_fn, collision_fn):
            break

    solution = [start_conf]
    for q1, q2 in zip(path, path[1:]):
        solution.extend(extend_fn(samples[q1], samples[q2]))
    return solution, samples, edges, colliding_vertices, colliding_edges


def lazy_prm_replan_loop(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, params_list, smooth=0, **kwargs):
    if collision_fn(start_conf) or collision_fn(end_conf):
        return None
    path = direct_path(start_conf, end_conf, extend_fn, collision_fn)
    if path is not None:
        return path
    for num_samples in params_list:
        res = lazy_prm(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                       num_samples=num_samples, **kwargs)
        # print(res)
        path, samples, edges, colliding_vertices, colliding_edges = res
        if path is not None:
            return smooth_path(path, extend_fn, collision_fn, iterations=smooth)
    return None
