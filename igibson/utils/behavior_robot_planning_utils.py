import os
import time

import numpy as np


import igibson
# from igibson.external.pybullet_tools.utils import (
    CIRCULAR_LIMITS,
    MAX_DISTANCE,
    PI,
    birrt,
    circular_difference,
    direct_path,
    get_base_difference_fn,
    get_base_distance_fn,
    pairwise_collision,
)
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


def plan_base_motion_br(
    robot: BehaviorRobot,
    end_conf,
    base_limits,
    obstacles=[],
    direct=False,
    weights=1 * np.ones(3),
    resolutions=0.05 * np.ones(3),
    max_distance=MAX_DISTANCE,
    override_sample_fn=None,
    **kwargs
):
    def sample_fn():
        x, y = np.random.uniform(*base_limits)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    if override_sample_fn is not None:
        sample_fn = override_sample_fn

    difference_fn = get_base_difference_fn()
    distance_fn = get_base_distance_fn(weights=weights)

    # TODO(behaviorrobot): Unify
    body_ids = []
    for part in ["body", "left_hand", "right_hand"]:
        body_ids.append(robot.links[part].body_id)

    def extend_fn(q1, q2):
        target_theta = np.arctan2(q2[1] - q1[1], q2[0] - q1[0])

        n1 = int(np.abs(circular_difference(target_theta, q1[2]) / resolutions[2])) + 1
        n3 = int(np.abs(circular_difference(q2[2], target_theta) / resolutions[2])) + 1
        steps2 = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n2 = int(np.max(steps2)) + 1

        for i in range(n1):
            q = (i / (n1)) * np.array(difference_fn((q1[0], q1[1], target_theta), q1)) + np.array(q1)
            q = tuple(q)
            yield q

        for i in range(n2):
            q = (i / (n2)) * np.array(
                difference_fn((q2[0], q2[1], target_theta), (q1[0], q1[1], target_theta))
            ) + np.array((q1[0], q1[1], target_theta))
            q = tuple(q)
            yield q

        for i in range(n3):
            q = (i / (n3)) * np.array(difference_fn(q2, (q2[0], q2[1], target_theta))) + np.array(
                (q2[0], q2[1], target_theta)
            )
            q = tuple(q)
            yield q

    def collision_fn(q):
        # TODO: update this function
        # set_base_values(body, q)
        robot.set_position_orientation([q[0], q[1], robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, q[2]]))
        return any(
            pairwise_collision(body_id, obs, max_distance=max_distance) for obs in obstacles for body_id in body_ids
        )

    pos = robot.get_position()
    yaw = p.getEulerFromQuaternion(robot.get_orientation())[2]
    start_conf = [pos[0], pos[1], yaw]
    if collision_fn(start_conf):
        # print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        # print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)


def dry_run_base_plan(robot: BehaviorRobot, plan):
    for (x, y, yaw) in plan:
        robot.set_position_orientation([x, y, robot.initial_z_offset], p.getQuaternionFromEuler([0, 0, yaw]))
        time.sleep(0.01)


def dry_run_arm_plan(robot: BehaviorRobot, plan):
    for (x, y, z, roll, pitch, yaw) in plan:
        robot.links["right_hand"].set_position_orientation([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
        time.sleep(0.01)


def get_hand_difference_fn():
    def fn(q2, q1):
        dx, dy, dz = np.array(q2[:3]) - np.array(q1[:3])
        droll = circular_difference(q2[3], q1[3])
        dpitch = circular_difference(q2[4], q1[4])
        dyaw = circular_difference(q2[5], q1[5])
        return (dx, dy, dz, droll, dpitch, dyaw)

    return fn


def get_hand_distance_fn(weights=1 * np.ones(6)):
    difference_fn = get_hand_difference_fn()

    def fn(q1, q2):
        difference = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, difference * difference))

    return fn


def plan_hand_motion_br(
    robot: BehaviorRobot,
    obj_in_hand,
    end_conf,
    hand_limits,
    obstacles=[],
    direct=False,
    weights=(1, 1, 1, 5, 5, 5),
    resolutions=0.02 * np.ones(6),
    max_distance=MAX_DISTANCE,
    **kwargs
):
    def sample_fn():
        x, y, z = np.random.uniform(*hand_limits)
        r, p, yaw = np.random.uniform((-PI, -PI, -PI), (PI, PI, PI))
        return (x, y, z, r, p, yaw)

    difference_fn = get_hand_difference_fn()
    distance_fn = get_hand_distance_fn(weights=weights)

    pos = robot.links["right_hand"].get_position()
    orn = robot.links["right_hand"].get_orientation()
    rpy = p.getEulerFromQuaternion(orn)
    start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

    if obj_in_hand is not None:
        obj_pos = obj_in_hand.get_position()
        obj_orn = obj_in_hand.get_orientation()
        local_pos, local_orn = p.multiplyTransforms(*p.invertTransform(pos, orn), obj_pos, obj_orn)

    def extend_fn(q1, q2):
        steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n = int(np.max(steps)) + 1

        for i in range(n):
            q = (i / float(n)) * np.array(difference_fn(q2, q1)) + np.array(q1)
            q = tuple(q)
            yield q

    def collision_fn(q):
        # TODO: update this function
        # set_base_values(body, q)
        robot.links["right_hand"].set_position_orientation(
            [q[0], q[1], q[2]], p.getQuaternionFromEuler([q[3], q[4], q[5]])
        )
        if obj_in_hand is not None:
            obj_in_hand.set_position_orientation(
                *p.multiplyTransforms(
                    [q[0], q[1], q[2]], p.getQuaternionFromEuler([q[3], q[4], q[5]]), local_pos, local_orn
                )
            )

        # TODO(behaviorrobot): Unify
        collision = any(
            pairwise_collision(robot.links["right_hand"].body_id, obs, max_distance=max_distance) for obs in obstacles
        )

        if obj_in_hand is not None:
            collision = collision or any(
                pairwise_collision(bid, obs, max_distance=max_distance)
                for obs in obstacles
                for bid in obj_in_hand.get_body_ids()
            )

        return collision

    if collision_fn(start_conf):
        # print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        # print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)


if __name__ == "__main__":
    config = parse_config(os.path.join(igibson.example_config_path, "behavior.yaml"))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(
        mode="gui_interactive", use_pb_gui=True, image_width=256, image_height=256, rendering_settings=settings
    )

    scene = EmptyScene()
    scene.objects_by_id = {}
    s.import_scene(scene)

    agent = BehaviorRobot(show_visual_head=True, use_ghost_hands=False)
    s.import_object(agent)
    s.register_main_vr_robot(agent)
    initial_pos_z_offset = 0.7

    agent.initial_z_offset = initial_pos_z_offset
    agent.set_position_orientation([0, 0, initial_pos_z_offset], [0, 0, 0, 1])
    # plan = plan_base_motion_br(agent, [3,3,1], [(-5,-5), (5,5)])
    plan = plan_hand_motion_br(agent, [3, 3, 3, 0, 0, 0], ((-5, -5, -5), (5, 5, 5)))
    print(plan)
    for q in plan:
        agent.links["right_hand"].set_position_orientation(
            [q[0], q[1], q[2]], p.getQuaternionFromEuler([q[3], q[4], q[5]])
        )
        time.sleep(0.05)

    for i in range(10000):
        action = np.zeros((28,))
        if i < 2:
            action[19] = 1
            action[27] = 1
        agent.apply_action(action)
        s.step()
