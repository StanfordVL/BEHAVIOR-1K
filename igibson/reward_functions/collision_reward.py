from igibson.reward_functions.reward_function_base import BaseRewardFunction


class CollisionReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.

    Args:
        r_collision (float): Penalty value (>0) to penalize collisions
    """

    def __init__(self, r_collision=0.1):
        # Store internal vars
        assert r_collision > 0, f"r_collision must be positive, got: {r_collision}!"
        self._r_collision = r_collision

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        # Penalty is Reward is -self._r_collision if there were any collisions in the last timestep
        reward = float(len(env.current_collisions) > 0) * -self._r_collision
        return reward, {}
