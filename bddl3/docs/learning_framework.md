# Learning Frameworks

### Overview

iGibson can be used with any learning framework that accommodates OpenAI gym interface. Feel free to use your favorite ones.

### Examples

#### TF-Agents

In this example, we show an environment wrapper of [TF-Agents](https://github.com/tensorflow/agents) for iGibson and an example training code for [SAC agent](https://arxiv.org/abs/1801.01290). The code can be found in [our fork of TF-Agents](https://github.com/StanfordVL/agents/): [agents/blob/igibson/tf_agents/environments/suite_gibson.py](https://github.com/StanfordVL/agents/blob/igibson/tf_agents/environments/suite_gibson.py) and [agents/blob/igibson/tf_agents/agents/sac/examples/v1/train_single_env.sh](https://github.com/StanfordVL/agents/blob/igibson/tf_agents/agents/sac/examples/v1/train_single_env.sh).

```python
def load(config_file,
         model_id=None,
         env_mode='headless',
         action_timestep=1.0 / 10.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    config_file = os.path.join(os.path.dirname(igibson.__file__), config_file)
    env = iGibsonEnv(config_file=config_file,
                     scene_id=model_id,
                     mode=env_mode,
                     action_timestep=action_timestep,
                     physics_timestep=physics_timestep,
                     device_idx=device_idx)

    discount = env.config.get('discount_factor', 0.99)
    max_episode_steps = env.config.get('max_step', 500)

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
    )
```
