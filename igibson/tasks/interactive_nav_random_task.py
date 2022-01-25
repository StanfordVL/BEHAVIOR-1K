import numpy as np


from igibson.objects.ycb_object import YCBObject
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.utils.utils import restoreState


class InteractiveNavRandomTask(PointNavRandomTask):
    """
    Interactive Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of interactive objects that are small and light
    """

    def __init__(self, env):
        super(InteractiveNavRandomTask, self).__init__(env)
        self.interactive_objects = self.load_interactive_objects(env)
        env.collision_ignore_body_b_ids |= set([bid for obj in self.interactive_objects for bid in obj.get_body_ids()])

    def load_interactive_objects(self, env):
        """
        Load interactive objects (YCB objects)

        :param env: environment instance
        :return: a list of interactive objects
        """
        interactive_objects = []
        object_paths = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
        ]

        for object_path in object_paths:
            obj = YCBObject(object_path)
            env.simulator.import_object(obj)
            interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """
        max_trials = 100

        for obj in self.interactive_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                reset_success = env.test_valid_position(obj, pos, orn)
                restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            env.land(obj, pos, orn)

            p.removeState(state_id)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        super(InteractiveNavRandomTask, self).reset_scene(env)
        self.reset_interactive_objects(env)
