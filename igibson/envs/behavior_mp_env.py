import argparse
import time
from enum import IntEnum

import gym.spaces
import numpy as np


from igibson import object_states
from igibson.envs.behavior_env import BehaviorEnv
from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS
from igibson.object_states.on_floor import RoomFloor
from igibson.object_states.utils import sample_kinematics
from igibson.objects.usd_object import URDFObject
from igibson.robots.behavior_robot import BRBody, BREye, BRHand
from igibson.utils.behavior_robot_planning_utils import dry_run_base_plan, plan_base_motion_br, plan_hand_motion_br
from igibson.utils.utils import restoreState


class ActionPrimitives(IntEnum):
    NAVIGATE_TO = 0
    LEFT_GRASP = 1
    RIGHT_GRASP = 2
    LEFT_PLACE_ONTOP = 3
    RIGHT_PLACE_ONTOP = 4
    LEFT_PLACE_INSIDE = 5
    RIGHT_PLACE_INSIDE = 6
    OPEN = 7
    CLOSE = 8


def get_aabb_volume(lo, hi):
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def detect_collision(bodyA, object_in_hand=None):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA or body_id == object_in_hand:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot):
    left_object_in_hand = robot.links["left_hand"].object_in_hand
    right_object_in_hand = robot.links["right_hand"].object_in_hand
    return (
        detect_collision(robot.links["body"].get_body_id())
        or detect_collision(robot.links["left_hand"].get_body_id(), left_object_in_hand)
        or detect_collision(robot.links["right_hand"].get_body_id(), right_object_in_hand)
    )


class BehaviorMPEnv(BehaviorEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        automatic_reset=False,
        seed=0,
        action_filter="mobile_manipulation",
        use_motion_planning=False,
        activity_relevant_objects_only=True,
    ):
        """
        @param config_file: config_file path
        @param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive
        @param action_timestep: environment executes action per action_timestep second
        @param physics_timestep: physics timestep for pybullet
        @param device_idx: which GPU to run the simulation and rendering on
        @param automatic_reset: whether to automatic reset after an episode finishes
        @param seed: RNG seed for sampling
        @param action_filter: see BehaviorEnv
        @param use_motion_planning: Whether motion-planned primitives or magic primitives should be used
        @param activity_relevant_objects_only: Whether the actions should be parameterized by AROs or all scene objs.
        """
        super(BehaviorMPEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            action_filter=action_filter,
            seed=seed,
            automatic_reset=automatic_reset,
        )

        self.use_motion_planning = use_motion_planning
        self.activity_relevant_objects_only = activity_relevant_objects_only
        self.robots[0].initial_z_offset = 0.7

    def load_action_space(self):
        if self.activity_relevant_objects_only:
            self.addressable_objects = [
                item
                for item in self.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
            ]
        else:
            self.addressable_objects = list(
                set(self.task.simulator.scene.objects_by_name.values()) | set(self.task.object_scope.values())
            )

        self.num_objects = len(self.addressable_objects)
        self.action_space = gym.spaces.Discrete(self.num_objects * len(ActionPrimitives))

    def get_body_ids(self, include_self=False):
        ids = []
        for object in self.scene.get_objects():
            if isinstance(object, URDFObject):
                ids.extend(object.body_ids)

        if include_self:
            ids.append(self.robots[0].links["left_hand"].get_body_id())
            ids.append(self.robots[0].links["body"].get_body_id())

        return ids

    def reset_and_release_hand(self, hand):
        self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())
        for _ in range(100):
            self.robots[0].links[hand].set_close_fraction(0)
            self.robots[0].links[hand].trigger_fraction = 0
            p.stepSimulation()

    def step(self, action):
        obj_list_id = int(action) % self.num_objects
        action_primitive = int(action) // self.num_objects

        obj = self.addressable_objects[obj_list_id]
        if not (isinstance(obj, BRBody) or isinstance(obj, BRHand) or isinstance(obj, BREye)):
            if action_primitive == ActionPrimitives.NAVIGATE_TO:
                if self.navigate_to_obj(obj):
                    print("PRIMITIVE: navigate to {} success".format(obj.name))
                else:
                    print("PRIMITIVE: navigate to {} fail".format(obj.name))

            elif action_primitive == ActionPrimitives.RIGHT_GRASP or action_primitive == ActionPrimitives.LEFT_GRASP:
                hand = "right_hand" if action_primitive == ActionPrimitives.RIGHT_GRASP else "left_hand"
                obj_in_hand_id = self.robots[0].links[hand].object_in_hand
                obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
                if obj_in_hand is None:
                    if isinstance(obj, URDFObject) and hasattr(obj, "states") and object_states.AABB in obj.states:
                        lo, hi = obj.states[object_states.AABB].get_value()
                        volume = get_aabb_volume(lo, hi)
                        if volume < 0.2 * 0.2 * 0.2 and not obj.fixed_base:  # we can only grasp small objects
                            self.navigate_if_needed(obj)
                            self.grasp_obj(obj, hand)
                            obj_in_hand_id = self.robots[0].links[hand].object_in_hand
                            obj_in_hand = (
                                self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
                            )
                            print("PRIMITIVE: grasp {} success, obj in hand {}".format(obj.name, obj_in_hand))
                        else:
                            print("PRIMITIVE: grasp {} fail, too big or fixed".format(obj.name))
            elif (
                action_primitive == ActionPrimitives.LEFT_PLACE_ONTOP
                or action_primitive == ActionPrimitives.RIGHT_PLACE_ONTOP
            ):
                hand = "right_hand" if action_primitive == ActionPrimitives.RIGHT_PLACE_ONTOP else "left_hand"
                obj_in_hand_id = self.robots[0].links[hand].object_in_hand
                obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
                if obj_in_hand is not None and obj_in_hand != obj:
                    print("PRIMITIVE:attempt to place {} ontop {}".format(obj_in_hand.name, obj.name))

                    if isinstance(obj, URDFObject):
                        self.navigate_if_needed(obj)

                        state = p.saveState()
                        result = sample_kinematics(
                            "onTop",
                            obj_in_hand,
                            obj,
                            True,
                            use_ray_casting_method=True,
                            max_trials=20,
                        )

                        if result:
                            pos = obj_in_hand.get_position()
                            orn = obj_in_hand.get_orientation()
                            self.place_obj(state, pos, orn, hand)
                            print("PRIMITIVE: place {} ontop {} success".format(obj_in_hand.name, obj.name))
                        else:
                            p.removeState(state)
                            print("PRIMITIVE: place {} ontop {} fail, sampling fail".format(obj_in_hand.name, obj.name))
                    else:
                        state = p.saveState()
                        result = sample_kinematics(
                            "onFloor", obj_in_hand, obj, True, use_ray_casting_method=True, max_trials=20
                        )
                        if result:
                            print("PRIMITIVE: place {} ontop {} success".format(obj_in_hand.name, obj.name))
                            pos = obj_in_hand.get_position()
                            orn = obj_in_hand.get_orientation()
                            self.place_obj(state, pos, orn, hand)
                        else:
                            print("PRIMITIVE: place {} ontop {} fail, sampling fail".format(obj_in_hand.name, obj.name))
                            p.removeState(state)

            elif (
                action_primitive == ActionPrimitives.LEFT_PLACE_INSIDE
                or action_primitive == ActionPrimitives.RIGHT_PLACE_INSIDE
            ):
                hand = "right_hand" if action_primitive == ActionPrimitives.RIGHT_PLACE_INSIDE else "left_hand"
                obj_in_hand_id = self.robots[0].links[hand].object_in_hand
                obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
                if obj_in_hand is not None and obj_in_hand != obj and isinstance(obj, URDFObject):
                    print("PRIMITIVE:attempt to place {} inside {}".format(obj_in_hand.name, obj.name))
                    if (
                        hasattr(obj, "states")
                        and object_states.Open in obj.states
                        and obj.states[object_states.Open].get_value()
                    ) or (hasattr(obj, "states") and not object_states.Open in obj.states):
                        self.navigate_if_needed(obj)

                        state = p.saveState()
                        result = sample_kinematics(
                            "inside",
                            obj_in_hand,
                            obj,
                            True,
                            use_ray_casting_method=True,
                            max_trials=20,
                        )

                        if result:
                            pos = obj_in_hand.get_position()
                            orn = obj_in_hand.get_orientation()
                            self.place_obj(state, pos, orn, hand)
                            print("PRIMITIVE: place {} inside {} success".format(obj_in_hand.name, obj.name))
                        else:
                            print(
                                "PRIMITIVE: place {} inside {} fail, sampling fail".format(obj_in_hand.name, obj.name)
                            )
                            p.removeState(state)
                    else:
                        print(
                            "PRIMITIVE: place {} inside {} fail, need open not open".format(obj_in_hand.name, obj.name)
                        )
            elif action_primitive == ActionPrimitives.OPEN:
                self.navigate_if_needed(obj)

                if hasattr(obj, "states") and object_states.Open in obj.states:
                    obj.states[object_states.Open].set_value(True, fully=True)
                else:
                    print("PRIMITIVE open failed, cannot be opened")

            elif action_primitive == ActionPrimitives.CLOSE:
                self.navigate_if_needed(obj)

                if hasattr(obj, "states") and object_states.Open in obj.states:
                    obj.states[object_states.Open].set_value(False)
                else:
                    print("PRIMITIVE close failed, cannot be opened")

        state, reward, done, info = super(BehaviorMPEnv, self).step(np.zeros(17))
        print("PRIMITIVE satisfied predicates:", info["satisfied_predicates"])
        return state, reward, done, info

    def grasp_obj(self, obj, hand):
        if self.use_motion_planning:
            x, y, _ = obj.get_position()
            z = obj.states[object_states.AABB].get_value()[1][2]
            hand_x, hand_y, hand_z = self.robots[0].links[hand].get_position()

            # add a little randomness to avoid getting stuck
            x += np.random.uniform(-0.025, 0.025)
            y += np.random.uniform(-0.025, 0.025)
            z += np.random.uniform(-0.025, 0.025)

            minx = min(x, hand_x) - 0.5
            miny = min(y, hand_y) - 0.5
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 0.5
            maxy = max(y, hand_y) + 0.5
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            # plan a motion to above the object
            plan = plan_hand_motion_br(
                robot=self.robots[0],
                obj_in_hand=None,
                end_conf=[x, y, z + 0.05, 0, np.pi * 5 / 6.0, np.random.uniform(-np.pi, np.pi)],
                hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                obstacles=self.get_body_ids(include_self=True),
            )
            restoreState(state)
            p.removeState(state)

            if plan is not None:
                grasp_success = self.execute_grasp_plan(plan, obj, hand)
                print("grasp success", grasp_success)
                if not grasp_success:
                    print("grasp failed")
                    self.reset_and_release_hand(hand)
            else:
                print("plan is None")
                self.reset_and_release_hand(hand)
                # reset hand
        else:
            obj.set_position(np.array(self.robots[0].links[hand].get_position()))  #  - np.array([0,0,0.05])
            self.robots[0].links[hand].set_close_fraction(1)
            self.robots[0].links[hand].trigger_fraction = 1
            p.stepSimulation()
            obj.set_position(np.array(self.robots[0].links[hand].get_position()))
            self.robots[0].links[hand].handle_assisted_grasping(
                np.zeros(
                    28,
                ),
                override_ag_data=(obj.get_body_id(), -1),
            )

    def execute_grasp_plan(self, plan, obj, hand):
        for x, y, z, roll, pitch, yaw in plan:
            self.robots[0].links[hand].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        x, y, z, roll, pitch, yaw = plan[-1]

        for i in range(25):
            self.robots[0].links[hand].move([x, y, z - i * 0.005], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        for _ in range(50):
            self.robots[0].links[hand].set_close_fraction(1)
            self.robots[0].links[hand].trigger_fraction = 1
            p.stepSimulation()

        grasp_success = (
            self.robots[0]
            .links[hand]
            .handle_assisted_grasping(
                np.zeros(
                    28,
                ),
                override_ag_data=(obj.get_body_id(), -1),
            )
        )

        for x, y, z, roll, pitch, yaw in plan[::-1]:
            self.robots[0].links[hand].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
            p.stepSimulation()

        return grasp_success

    def place_obj(self, original_state, target_pos, target_orn, hand):
        obj_in_hand_id = self.robots[0].links[hand].object_in_hand
        obj_in_hand = self.scene.objects_by_id[obj_in_hand_id]

        pos = obj_in_hand.get_position()
        restoreState(original_state)
        p.removeState(original_state)
        if not self.use_motion_planning:
            self.reset_and_release_hand(hand)

            self.robots[0].links[hand].force_release_obj()
            obj_in_hand.set_position_orientation(target_pos, target_orn)

        else:
            x, y, z = target_pos
            hand_x, hand_y, hand_z = self.robots[0].links[hand].get_position()

            minx = min(x, hand_x) - 1
            miny = min(y, hand_y) - 1
            minz = min(z, hand_z) - 0.5
            maxx = max(x, hand_x) + 1
            maxy = max(y, hand_y) + 1
            maxz = max(z, hand_z) + 0.5

            state = p.saveState()
            obstacles = self.get_body_ids()
            obstacles.remove(self.obj_in_hand.get_body_id())
            plan = plan_hand_motion_br(
                robot=self.robots[0],
                obj_in_hand=obj_in_hand,
                end_conf=[x, y, z + 0.1, 0, np.pi * 5 / 6.0, 0],
                hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                obstacles=obstacles,
            )  #
            restoreState(state)
            p.removeState(state)

            if plan:
                for x, y, z, roll, pitch, yaw in plan:
                    self.robots[0].links[hand].move([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))
                    p.stepSimulation()

                self.reset_and_release_hand(hand)

                # force release object to avoid dealing with stateful AG release mechanism
                self.robots[0].links[hand].force_release_obj()
                self.robots[0].set_position_orientation(self.robots[0].get_position(), self.robots[0].get_orientation())

                # reset the released object to zero velocity
                p.resetBaseVelocity(obj_in_hand.get_body_id(), linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

                # let object fall
                for _ in range(100):
                    p.stepSimulation()

    def sample_fn(self):
        random_point = self.scene.get_random_point()
        x, y = random_point[1][:2]
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    def navigate_to_obj(self, obj):
        # test agent positions around an obj
        # try to place the agent near the object, and rotate it to the object
        valid_position = None  # ((x,y,z),(roll, pitch, yaw))
        original_position = self.robots[0].get_position()
        original_orientation = self.robots[0].get_orientation()
        if isinstance(obj, URDFObject):
            distance_to_try = [0.6, 1.2, 1.8, 2.4]
            obj_pos = obj.get_position()
            for distance in distance_to_try:
                for _ in range(20):
                    yaw = np.random.uniform(-np.pi, np.pi)
                    pos = [obj_pos[0] + distance * np.cos(yaw), obj_pos[1] + distance * np.sin(yaw), 0.7]
                    orn = [0, 0, yaw - np.pi]
                    self.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
                    eye_pos = self.robots[0].links["eye"].get_position()
                    obj_pos = obj.get_position()
                    ray_test_res = p.rayTest(eye_pos, obj_pos)
                    blocked = False
                    if len(ray_test_res) > 0 and ray_test_res[0][0] != obj.get_body_id():
                        blocked = True

                    if not detect_robot_collision(self.robots[0]) and not blocked:
                        valid_position = (pos, orn)
                        break
                if valid_position is not None:
                    break
        else:
            for _ in range(60):
                _, pos = obj.scene.get_random_point_by_room_instance(obj.room_instance)
                yaw = np.random.uniform(-np.pi, np.pi)
                orn = [0, 0, yaw]
                self.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
                if not detect_robot_collision(self.robots[0]):
                    valid_position = (pos, orn)
                    break

        if valid_position is not None:
            target_x = valid_position[0][0]
            target_y = valid_position[0][1]
            x = original_position[0]
            y = original_position[1]
            minx = min(x, target_x) - 1
            miny = min(y, target_y) - 1
            maxx = max(x, target_x) + 1
            maxy = max(y, target_y) + 1

            if self.use_motion_planning:
                self.robots[0].set_position_orientation(original_position, original_orientation)
                plan = plan_base_motion_br(
                    robot=self.robots[0],
                    end_conf=[valid_position[0][0], valid_position[0][1], valid_position[1][2]],
                    base_limits=[(minx, miny), (maxx, maxy)],
                    obstacles=self.get_body_ids(),
                    override_sample_fn=self.sample_fn,
                )

                if plan is not None:
                    if self.mode != "headless":
                        dry_run_base_plan(robot=self.robots[0], plan=plan)
                    else:
                        # TODO: Still execute the plan in headless mode instead of just teleporting.
                        self.robots[0].set_position_orientation(
                            valid_position[0], p.getQuaternionFromEuler(valid_position[1])
                        )
                    return True
                else:
                    self.robots[0].set_position_orientation(original_position, original_orientation)
                    return False
            else:
                self.robots[0].set_position_orientation(valid_position[0], p.getQuaternionFromEuler(valid_position[1]))
                return True
        else:
            self.robots[0].set_position_orientation(original_position, original_orientation)
            return False

    def navigate_if_needed(self, obj):
        if obj.states[object_states.InReachOfRobot].get_value():
            return

        for _ in range(10):
            if self.navigate_to_obj(obj):
                return

    def reset(self, resample_objects=False):
        obs = super(BehaviorMPEnv, self).reset()
        for hand in ["left_hand", "right_hand"]:
            self.robots[0].links[hand].set_close_fraction(0)
            self.robots[0].links[hand].trigger_fraction = 0
            self.robots[0].links[hand].force_release_obj()
        return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="igibson/examples/configs/behavior.yaml",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="gui_interactive",
        help="which mode for simulation (default: gui_interactive)",
    )
    args = parser.parse_args()

    env = BehaviorMPEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 300.0,
        physics_timestep=1.0 / 300.0,
        use_motion_planning=True,
    )
    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()

        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(reward, info)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
