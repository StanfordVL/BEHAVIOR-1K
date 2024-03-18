import omnigibson as og

class VectorEnvironment():
    
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.envs = [og.Environment(configs=config, num_env=i) for i in range(num_envs)]

    def step(self, actions):
        try:
            observations, rewards, dones, infos = [], [], [], []
            for i, action in enumerate(actions):
                self.envs[i]._pre_step(action)
            # Run simulation step
            og.sim.step()
            for i, action in enumerate(actions):
                obs, reward, done, info = self.envs[i]._post_step(action)
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            return self._post_step()
        except:
            raise ValueError(f"Failed to execute environment step {self._current_step} in episode {self._current_episode}")
    
    def reset(self):
        for env in self.envs:
            env.reset()

    def close(self):
        for env in self.envs:
            env.close()

    def __len__(self):
        return self.num_envs
    