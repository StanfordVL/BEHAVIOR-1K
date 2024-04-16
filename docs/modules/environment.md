---
icon: material/earth
---

# 🌎 **Environment**

The OpenAI Gym Environment serves as a top-level simulation object, offering a suite of common interfaces. These include methods such as `step`, `reset`, `render`, and properties like `observation_space` and `action_space`. The OmniGibson Environment builds upon this foundation by also supporting the loading of scenes, robots, and tasks. Following the OpenAI Gym interface, the OmniGibson environment further provides access to both the action space and observation space of the robots and external sensors.

Creating a minimal environment requires the definition of a config dictionary. This dictionary should contain details about the scene, objects, robots, and specific characteristics of the environment:

<details>
<summary>Click to see code!</summary>
<pre><code>
import omnigibson as og

cfg = {
    "env": {
        "action_frequency": 10,
        "physics_frequency": 120,
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
</code></pre>
</details>

