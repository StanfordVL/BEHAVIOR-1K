import copy
import omnigibson as og

# TODO(rl): Figure out if there is a good interface to implement
class VectorEnvironment:

    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.envs = [og.Environment(configs=copy.deepcopy(config), num_env=i) for i in range(num_envs)]

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
            return observations, rewards, dones, infos
        except Exception as e:
            print(e)

    def reset(self):
        for env in self.envs:
            env.reset()

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
