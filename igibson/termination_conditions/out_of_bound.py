from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition


class OutOfBound(BaseTerminationCondition):
    """
    OutOfBound used for navigation tasks in InteractiveIndoorScene
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config):
        super(OutOfBound, self).__init__(config)
        self.fall_off_thresh = self.config.get("fall_off_thresh", 0.03)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """

        done = False
        # fall off the cliff of valid region
        if isinstance(env.scene, InteractiveIndoorScene):
            robot_z = env.robots[0].get_position()[2]
            if robot_z < (env.scene.get_floor_height() - self.fall_off_thresh):
                done = True
        success = False
        return done, success
