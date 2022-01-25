import gym
import numpy as np


from igibson.robots.robot_locomotor import LocomotionRobot


class Quadrotor(LocomotionRobot):
    """
    Quadrotor robot
    Reference: https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations
    Uses joint torque control
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.torque = config.get("torque", 0.02)
        LocomotionRobot.__init__(
            self,
            "quadrotor/quadrotor.urdf",
            action_dim=6,
            torque_coef=2.5,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
            **kwargs
        )

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [
            [self.torque, 0, 0, 0, 0, 0],
            [-self.torque, 0, 0, 0, 0, 0],
            [0, self.torque, 0, 0, 0, 0],
            [0, -self.torque, 0, 0, 0, 0],
            [0, 0, self.torque, 0, 0, 0],
            [0, 0, -self.torque, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def apply_action(self, action):
        """
        Apply policy action. Zero gravity.
        """
        real_action = self.policy_action_to_robot_action(action)
        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(self.get_body_ids()[0], real_action[:3], real_action[3:])

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord("w"),): 0,  # +x
            (ord("s"),): 1,  # -x
            (ord("d"),): 2,  # +y
            (ord("a"),): 3,  # -y
            (ord("z"),): 4,  # +z
            (ord("x"),): 5,  # -z
            (): 6,
        }
