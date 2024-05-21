import omnigibson.utils.transform_utils as T
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class GraspGoal(SuccessCondition):
    """
    GraspGoal (success condition)
    """

    def __init__(self, obj_name):
        self.obj_name = obj_name
        self.obj = None

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        self.obj = env.scene.object_registry("name", self.obj_name) if self.obj is None else self.obj
        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        return obj_in_hand == self.obj
