from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from learner_worker import GRPCVecEnv

import sys

n_envs = int(sys.argv[1])
env = GRPCVecEnv("localhost:50051", n_envs)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")