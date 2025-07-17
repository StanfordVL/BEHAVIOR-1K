# :material-earth: **Environment**

## Description

**`OmniGibson`**'s Environment class is an [OpenAI gym-compatible](https://gymnasium.farama.org/content/gym_compatibility/) interface and is the main entry point for interacting with the underlying simulation. A single environment loads a user-specified scene, object(s), robot(s), and task combination, and steps the resulting simulation while deploying user-specified actions to the loaded robot(s) and returning observations and task information from the simulator.

## Usage

### Creating

Creating a minimal environment requires the definition of a config dictionary. This dictionary should contain details about the [scene](./scenes.md), [objects](./objects.md), [robots](./robots.md), and specific characteristics of the environment:

??? code "env_simple.py"
    ``` python linenums="1"
    import omnigibson as og
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "Scene",
        },
        "objects": [],
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": 'all',
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                    },
                },
            }
        ]
    }
    
    env = og.Environment(configs=cfg)
    action = ...
    obs, reward, terminated, truncated, info = env.step(action)
    ```

### Runtime

Once created, the environment can be interfaced roughly in the same way as an OpenAI gym environment, and include common methods such as `step`, `reset`, `render`, as well as properties such as `observation_space` and `action_space`. Stepping the environment is done via `obs, reward, terminated, truncated, info = env.step(action)`, and resetting can manually be executed via `obs = env.reset()`. Robots are tracked explicitly via `env.robots`, and the underlying scene and all corresponding objects within the scene can be accessed via `env.scene`.


## Types

**`OmniGibson`** provides the main [`Environment`](../reference/envs/env_base.md) class, which should offer most of the essential functionality necessary for running robot experiments and interacting with the underlying simulator.

However, for more niche use-caches (such as demonstration collection, or batched environments), **`OmniGibson`** offers the [`EnvironmentWrapper`](../reference/envs/env_wrapper.md) class to easily extend the core environment functionality.
