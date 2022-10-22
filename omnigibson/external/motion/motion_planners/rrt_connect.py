"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by OmniGibson team.
"""
import cv2

from .smoothing import smooth_path
from .rrt import TreeNode, configs
from .utils import irange, argmin, RRT_ITERATIONS, RRT_RESTARTS, RRT_SMOOTHING


def asymmetric_extend(q1, q2, extend_fn, backward=False):
    if backward:
        return reversed(list(extend_fn(q2, q1)))
    return extend_fn(q1, q2)


def rrt_connect(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, iterations=RRT_ITERATIONS, draw_path=None, draw_point=None):
    # TODO: collision(q1, q2)
    debugging_prints = False
    if debugging_prints:
        print("rrt_connect: check whether src and dst are collision free")
    if collision_fn(q1) or collision_fn(q2):
        if debugging_prints:
            print("rrt_connect: src or dst are in collision! End")
        return None
    if debugging_prints:
        print("rrt_connect: src and dst are collision free. Continue")
    nodes1, nodes2 = [TreeNode(q1)], [TreeNode(q2)]
    for iteration in irange(iterations):
        swap = len(nodes1) > len(nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        s = sample_fn()
        if debugging_prints:
            print("rrt_connect: new sampled point {}".format(s))

        if draw_point is not None:
            draw_point(s, (0, 0, 255), not_in_image=True)
            cv2.waitKey(1)

        last1 = argmin(lambda n: distance_fn(n.config, s), tree1)
        for q in asymmetric_extend(last1.config, s, extend_fn, swap):
            if collision_fn(q):
                if debugging_prints:
                    print("rrt_connect: collision in the point {} along the direct path from sample to closest point of tree1".format(q))
                break
            if debugging_prints:
                print("rrt_connect: collision-free point {} along the direct path from sample to closest point. Adding it to the tree1".format(q))
            if draw_path is not None:
                draw_path(last1.config, q, (0, 255, 0))
            last1 = TreeNode(q, parent=last1)
            tree1.append(last1)

        last2 = argmin(lambda n: distance_fn(n.config, last1.config), tree2)
        for q in asymmetric_extend(last2.config, last1.config, extend_fn, not swap):
            if collision_fn(q):
                if debugging_prints:
                    print("rrt_connect: collision the point {} along the direct path from last point of tree1 and to closest point of tree2".format(q))
                break
            if debugging_prints:
                print("rrt_connect: collision-free point {} along the direct path from last point of tree1 and to closest point of tree2. Adding it to the tree2".format(q))
            if draw_path is not None:
                draw_path(last2.config, q, (255, 255, 0))
            last2 = TreeNode(q, parent=last2)
            tree2.append(last2)
        else:
            if debugging_prints:
                print("rrt_connect: full collision-free path between points of tree1 and tree2. Connecting path found! END")
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            # print('{} iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            return configs(path1[:-1] + path2[::-1])
    return None


# TODO: version which checks whether the segment is valid

def direct_path(q1, q2, extend_fn, collision_fn):
    debugging_prints = False
    if collision_fn(q1) or collision_fn(q2):
        return None
    path = [q1]
    for q in extend_fn(q1, q2):
        if debugging_prints:
            print("direct_path: extending to {}".format(q))
        if collision_fn(q):
            if debugging_prints:
                print("direct_path: in collision")
            return None
        path.append(q)
    return path


def birrt(q1, q2, distance, sample, extend, collision, draw_path=None, draw_point=None,
          restarts=RRT_RESTARTS, iterations=RRT_ITERATIONS, smooth=RRT_SMOOTHING):
    debugging_prints = False
    # If the initial or final configurations are in collision, there is no collision free path
    if collision(q1) or collision(q2):
        return None
    # Test if there is a direct path between initial and final configurations
    if debugging_prints:
        print("birrt: Check direct path")
    path = direct_path(q1, q2, extend, collision)
    if path is not None:
        if debugging_prints:
            print("birrt: There is a direct path! End")
        return path
    if debugging_prints:
        print("birrt: No direct path")
    for attempt in irange(restarts + 1):
        path = rrt_connect(q1, q2, distance, sample, extend, collision, iterations=iterations, draw_path=draw_path, draw_point=draw_point)
        if path is not None:
            if debugging_prints:
                print("birrt: {} RRT connect attempts".format(attempt))
            if smooth is None:
                return path
            return smooth_path(path, extend, collision, iterations=smooth)
    return None
