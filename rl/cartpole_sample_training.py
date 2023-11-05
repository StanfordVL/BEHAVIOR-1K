from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.offline.estimators import (
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
import gymnasium as gym

config = (
    DQNConfig()
    .environment(env="CartPole-v1")
    .framework("torch")
    .offline_data(input_="./cartpole_sample_training")
    .evaluation(
        evaluation_interval=100,
        evaluation_duration=10,
        evaluation_num_workers=1,
        evaluation_duration_unit="episodes",
        evaluation_config={"input": "./cartpole_sample_training_eval"},
        off_policy_estimation_methods={
            "is": {"type": ImportanceSampling},
            "wis": {"type": WeightedImportanceSampling},
            "dm_fqe": {
                "type": DirectMethod,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
            "dr_fqe": {
                "type": DoublyRobust,
                "q_model_config": {"type": FQETorchModel, "polyak_coef": 0.05},
            },
        },
    )
)

# config = (
#     DQNConfig()
#     .environment(env="CartPole-v0")
#     .framework("torch")
#     .rollouts(num_rollout_workers=1)
#     .evaluation(
#         evaluation_interval=100,
#         evaluation_duration=10,
#         evaluation_num_workers=1,
#         evaluation_duration_unit="episodes",
#         evaluation_config={"input": "sampler"},
#     )
# )

algo = config.build()
for i in range(1000):
    result = algo.train()
    if i % 100 == 99:
        print(pretty_print(result['info']['learner']))
        from IPython import embed; embed()
        # print(pretty_print(result['evaluation']['sampler_results']))
        print(i)
        print('----------------------------------')

algo.save('./cartpole_sample_training_algo')
algo_loaded = Algorithm.from_checkpoint("./cartpole_sample_training_algo")
# algo.save('./cartpole_sample_training_algo_ddpg')
# algo_loaded = Algorithm.from_checkpoint("./cartpole_sample_training_algo_ddpg")
# algo.save('./cartpole_sample_training_algo_dqn')
# algo_loaded = Algorithm.from_checkpoint("./cartpole_sample_training_algo_dqn")
# algo_loaded = algo


env = gym.make("CartPole-v1", render_mode="human")
terminated = False
truncated = False
obs, info = env.reset()
for i in range(10):
    while not terminated and not truncated:
        action = algo_loaded.compute_single_action(obs, explore=False)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)  # take a random action
        if terminated or truncated:
            obs, info = env.reset()
    terminated = False
    truncated = False

env.close()
