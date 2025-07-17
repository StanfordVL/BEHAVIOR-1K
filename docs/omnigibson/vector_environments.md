# :octicons-stack-16: **Vector Environments**

## Description

To support large-scale parallelization, we now support vector environments. Each environment is similar to our regular [environment](./environments.md), but our simulator now can keep track of multiple environments simultaneously. We have implemented many vectorized operations to optimize the performance of these parallel environments. We are also actively working on further enhancements to make them faster. Some use cases for this include reinforcement learning, parallelized training with domain randomization, and parallelized policy evaluation.

## Usage

Creating a minimal vector environment requires the definition of a config dictionary. This dictionary is copied across all environments:

??? code "vec_env_simple.py"
    ``` python linenums="1"
    import omnigibson as og
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "objects": [],
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
            }
        ]
    }
    
    vec_env = og.VectorEnvironment(num_envs=3, config=cfg)
    actions = [...]
    observations, rewards, terminates, truncates, infos = vec_env.step(actions)
    ```
