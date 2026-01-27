import logging

from .robot_context import RobotContext, MODE_MANIPULATION

logger = logging.getLogger(__name__)


class GripperController:
    def __init__(self, ctx: RobotContext):
        self.ctx = ctx

    def close(self, max_steps: int = 80) -> tuple[bool, list, list]:
        """Close gripper fully. Continues closing after grasp detection to ensure full grip."""
        observations = []
        actions = []
        grasped = False
        grasp_step = None
        MIN_STEPS_AFTER_GRASP = 3

        logger.debug("Closing gripper (grasping_mode=%s, max %d steps)",
                     self.ctx.robot.grasping_mode, max_steps)

        for step in range(max_steps):
            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            obj_in_hand = self.ctx.robot._ag_obj_in_hand.get(self.ctx.arm)
            if obj_in_hand is not None and not grasped:
                logger.debug("Grasped %s at step %d", obj_in_hand.name, step)
                grasped = True
                grasp_step = step

            if grasped and grasp_step is not None and (step - grasp_step) >= MIN_STEPS_AFTER_GRASP:
                break

            if terminated or truncated:
                break

        return grasped, observations, actions

    def open(self, num_steps: int = 60) -> tuple[list, list]:
        """Open gripper completely to release object."""
        observations = []
        actions = []

        for step in range(num_steps):
            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.gripper_idx] = 1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions
