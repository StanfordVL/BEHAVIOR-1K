import copy

import numpy as np


from igibson.metrics.metric_base import MetricBase


# TODO(behaviorrobot): Unify these metrics.
class BehaviorRobotMetric(MetricBase):
    def __init__(self):
        self.initialized = False

        self.state_cache = {}
        self.next_state_cache = {}

        self.agent_pos = {part: [] for part in ["left_hand", "right_hand", "body"]}
        self.agent_grasping = {part: [] for part in ["left_hand", "right_hand"]}

        self.agent_local_pos = {part: [] for part in ["left_hand", "right_hand"]}

        self.agent_reset = {part: [] for part in ["left_hand", "right_hand", "body"]}

        self.delta_agent_work = {part: [] for part in ["left_hand", "right_hand", "body"]}
        self.delta_agent_distance = {part: [] for part in ["left_hand", "right_hand", "body"]}
        self.delta_agent_grasp_distance = {part: [] for part in ["left_hand", "right_hand"]}

        self.clip = 0.2

    def step_callback(self, env, _):
        robot = env.robots[0]
        agent_work = {part: 0 for part in ["left_hand", "right_hand", "body"]}
        agent_distance = {part: 0 for part in ["left_hand", "right_hand", "body"]}

        for part in ["left_hand", "right_hand", "body"]:
            self.next_state_cache[part] = {
                "position": np.array(p.getBasePositionAndOrientation(robot.links[part].body_id)[0]),
            }

        if not self.initialized:
            self.state_cache = copy.deepcopy(self.next_state_cache)
            self.initialized = True

        if robot.action[19] > 0 and robot.action[27] > 0:
            self.agent_reset["left_hand"].append(True)
            self.agent_reset["right_hand"].append(True)
            self.agent_reset["body"].append(True)
        if robot.action[19] > 0:
            self.agent_reset["left_hand"].append(True)
            self.agent_reset["right_hand"].append(False)
            self.agent_reset["body"].append(True)
        elif robot.action[27] > 0:
            self.agent_reset["left_hand"].append(False)
            self.agent_reset["right_hand"].append(True)
            self.agent_reset["body"].append(True)
        else:
            self.agent_reset["left_hand"].append(False)
            self.agent_reset["right_hand"].append(False)
            self.agent_reset["body"].append(False)

        for part in self.state_cache:
            delta_pos = np.linalg.norm(self.next_state_cache[part]["position"] - self.state_cache[part]["position"])
            self.agent_pos[part].append(list(self.state_cache[part]["position"]))
            # Exclude agent teleports
            delta_pos = np.clip(delta_pos, -self.clip, self.clip)
            if robot.links[part].movement_cid is None:
                force = 0
                work = 0
            else:
                force = p.getConstraintState(robot.links[part].movement_cid)
                work = np.abs((delta_pos * np.linalg.norm(force)))

            distance = np.abs(delta_pos)
            if part in ["left_hand", "right_hand"]:
                self.agent_local_pos[part].append(list(robot.links[part].get_local_position_orientation()[0]))
            if part in ["left_hand", "right_hand"] and (
                len(p.getContactPoints(robot.links[part].body_id)) > 0 or robot.links[part].object_in_hand is not None
            ):
                self.delta_agent_grasp_distance[part].append(distance)
                self.agent_grasping[part].append(True)
            elif part in ["left_hand", "right_hand"]:
                self.delta_agent_grasp_distance[part].append(0)
                self.agent_grasping[part].append(False)

            agent_work[part] = work
            agent_distance[part] = distance

            self.delta_agent_work[part].append(work)
            self.delta_agent_distance[part].append(distance)

        self.state_cache = copy.deepcopy(self.next_state_cache)

    def gather_results(self):
        return {
            "agent_distance": {
                "timestep": self.delta_agent_distance,
            },
            "grasp_distance": {
                "timestep": self.delta_agent_grasp_distance,
            },
            "work": {
                "timestep": self.delta_agent_work,
            },
            "pos": {
                "timestep": self.agent_pos,
            },
            "local_pos": {
                "timestep": self.agent_local_pos,
            },
            "grasping": {
                "timestep": self.agent_grasping,
            },
            "reset": {
                "timestep": self.agent_reset,
            },
        }


class FetchRobotMetric(MetricBase):
    def __init__(self):
        self.initialized = False

        self.state_cache = {}
        self.next_state_cache = {}

        self.agent_pos = {part: [] for part in ["gripper", "body"]}
        self.agent_grasping = {part: [] for part in ["gripper"]}

        self.agent_local_pos = {part: [] for part in ["gripper"]}

        self.delta_agent_distance = {part: [] for part in ["gripper", "body"]}
        self.delta_agent_grasp_distance = {part: [] for part in ["gripper"]}

        self.clip = 0.2

    def step_callback(self, env, _):
        robot = env.robots[0]
        agent_distance = {part: 0 for part in self.agent_pos}

        self.next_state_cache = {
            "gripper": {"position": robot.get_end_effector_position()},
            "body": {"position": robot.get_position()},
        }

        if not self.initialized:
            self.state_cache = copy.deepcopy(self.next_state_cache)
            self.initialized = True

        self.agent_pos["body"].append(list(self.state_cache["body"]["position"]))
        delta_pos = np.linalg.norm(
            np.array(self.next_state_cache["body"]["position"]) - self.state_cache["body"]["position"]
        )
        distance = np.abs(delta_pos)
        self.delta_agent_distance["body"].append(distance)

        self.agent_pos["gripper"].append(list(self.state_cache["gripper"]["position"]))
        delta_pos = np.linalg.norm(
            self.next_state_cache["gripper"]["position"] - self.state_cache["gripper"]["position"]
        )
        gripper_distance = np.abs(delta_pos)
        self.delta_agent_distance["gripper"].append(gripper_distance)

        self.agent_local_pos["gripper"].append(robot.get_relative_eef_position().tolist())

        eef = robot._links[robot.eef_link_name]
        contacts = p.getContactPoints(bodyA=eef.body_id, linkIndexA=eef.link_id)
        if len(contacts) > 0:
            self.delta_agent_grasp_distance["gripper"].append(gripper_distance)
            self.agent_grasping["gripper"].append(True)
        else:
            self.delta_agent_grasp_distance["gripper"].append(0)
            self.agent_grasping["gripper"].append(False)

        self.state_cache = copy.deepcopy(self.next_state_cache)

    def gather_results(self):
        return {
            "agent_distance": {
                "timestep": self.delta_agent_distance,
            },
            "grasp_distance": {
                "timestep": self.delta_agent_grasp_distance,
            },
            "pos": {
                "timestep": self.agent_pos,
            },
            "local_pos": {
                "timestep": self.agent_local_pos,
            },
            "grasping": {
                "timestep": self.agent_grasping,
            },
        }
