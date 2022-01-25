import numpy as np


from igibson.robots.turtlebot import Turtlebot
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.utils.utils import restoreState


class DynamicNavRandomTask(PointNavRandomTask):
    """
    Dynamic Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of dynamic objects
    """

    def __init__(self, env):
        super(DynamicNavRandomTask, self).__init__(env)
        self.num_dynamic_objects = self.config.get("num_dynamic_objects", 1)
        # dynamic objects will repeat their actions for 10 action timesteps
        self.dynamic_objects_action_repeat = self.config.get("dynamic_objects_action_repeat", 10)

        self.dynamic_objects = self.load_dynamic_objects(env)
        self.dynamic_objects_last_actions = [robot.action_space.sample() for robot in self.dynamic_objects]

    def load_dynamic_objects(self, env):
        """
        Load dynamic objects (Turtlebots)

        :param env: environment instance
        :return: a list of interactive objects
        """
        dynamic_objects = []
        for _ in range(self.num_dynamic_objects):
            robot = Turtlebot(self.config)
            env.simulator.import_object(robot)
            dynamic_objects.append(robot)
        return dynamic_objects

    def reset_dynamic_objects(self, env):
        """
        Reset the poses of dynamic objects to have no collisions with the scene or the robot

        :param env: environment instance
        """
        max_trials = 100
        for robot in self.dynamic_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                reset_success = env.test_valid_position(robot, pos, orn)
                restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                print("WARNING: Failed to reset dynamic obj without collision")

            env.land(robot, pos, orn)

            p.removeState(state_id)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the dynamic objects after scene and agent reset

        :param env: environment instance
        """
        super(DynamicNavRandomTask, self).reset_scene(env)
        self.reset_dynamic_objects(env)

    def step(self, env):
        """
        Perform task-specific step: move the dynamic objects with action repeat

        :param env: environment instance
        """
        super(DynamicNavRandomTask, self).step(env)
        if env.current_step % self.dynamic_objects_action_repeat == 0:
            self.dynamic_objects_last_actions = [robot.action_space.sample() for robot in self.dynamic_objects]
        for robot, action in zip(self.dynamic_objects, self.dynamic_objects_last_actions):
            robot.apply_action(action)
