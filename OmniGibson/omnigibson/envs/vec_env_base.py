import copy
import omnigibson as og
from omnigibson.sensors import TiledVisionSensor
from tqdm import trange
import time


class VectorEnvironment:
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()

        # We check that every robot & object has a name specified in the config, so we have one-to-one mapping between envs
        if "robots" in config:
            for i, robot_cfg in enumerate(config["robots"]):
                assert "name" in robot_cfg, f"Robot at index {i} must specify a name for vector environment!"
        if "objects" in config:
            for i, obj_cfg in enumerate(config["objects"]):
                assert "name" in obj_cfg, f"Object at index {i} must specify a name for vector environment!"
        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = [
            og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]
        self.tiled_sensor = TiledVisionSensor(envs=self.envs)

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def step(self, actions):
        observations, rewards, terminates, truncates, infos = [], [], [], [], []
        for i, action in enumerate(actions):
            self.envs[i]._pre_step(action)
        a = time.time()
        og.sim.step()
        print("Sim step time:", time.time() - a)

        # tiled_buffer = self.tiled_sensor.get_obs()

        for i, action in enumerate(actions):
            # TODO: ignore camera observation here
            # TODO: potentially, we could get the tiled image first, segment it, and then replace all the normal camera observations with the segmented tiled image
            obs, reward, terminated, truncated, info = self.envs[i]._post_step(action)
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminated)
            truncates.append(truncated)
            infos.append(info)
        return observations, rewards, terminates, truncates, infos

    def reset(self, get_obs=True, **kwargs):
        # TODO: reset tiled rendering camera
        if get_obs:
            observations, infos = [], []
            for env in self.envs:
                obs, info = env.reset(get_obs=get_obs, **kwargs)
                observations.append(obs)
                infos.append(info)
            return observations, infos
        else:
            for env in self.envs:
                env.reset(get_obs=get_obs, **kwargs)

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
