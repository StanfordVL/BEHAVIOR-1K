from omnigibson.envs.env_wrapper import EnvironmentWrapper
import omnigibson as og

class RLEnv(EnvironmentWrapper):
    def __init__(self, env_config):
        cfg = env_config['cfg']
        self.env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
        self.reset_positions = env_config['reset_positions']
        super().__init__(self.env)

    def reset(self, seed, options):
        for name, position in enumerate(self.reset_positions):
            self.env.scene.object_registry("name", name).set_position_orientation(*position)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info
