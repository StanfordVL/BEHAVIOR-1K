# Evaluation

---

## Running Evaluations

We provide a unified entry point for running evaluation:
```
python OmniGibson/omnigibson/eval.py policy=websocket task.name=$TASK_NAME env_wrapper._target_=$WRAPPER_MODULE
```
Here is a brief explanation of the arguments:

- `$TASK_NAME` is the name of the task, a full list of tasks can be found in the demo gallery, as well as `TASK_TO_NAME_INDICES` under `OmniGibson/omnigibson/learning/utils/eval_utils.py`

- `WRAPPER_MODULE` is the full module path of the environment wrapper that will be used. For standard track, you MUST use the following command to start the evaluator:
    ```
    python OmniGibson/omnigibson/eval.py policy=websocket task.name=$TASK_NAME
    ```
Here, it will use the default `omnigibson.envs.EnvironmentWrapper`, which is a barebone wrapper that does not provide anything beyond our standard. The evaluator will load the task and spawn a server listening on `0.0.0.0:80` Feel free to use `omnigibson.learning.utils.network_utils.WebsocketPolicyServer` to serve your policy and communicate with the Evaluator. 

For privileged information track, you are allowed to design your own environment wrapper, within which you can arbitrarily query the environment instance for privileged information. We provided an example wrapper at `omnigibson.learning.wrappers.RichObservationWrapper`, which added `normal` and `flow` as additional visual observation modalities, as well as query for the pose of task relavant objects at every frame. The custom wrapper you wrote needs to submitted for inspection to make sure you have not abused the environment by any way (e.g. teleporting the robot, or changing object states directly). 


As a starter, we provided a codebase of common imitation learning algorithms for you to get started. Please refer to the baselines section for more information.


## Metrics and Results

We will calculate the following metric during policy rollout:

### Primary Metric (Ranking)
- **Task success score rate:** Averaged across 50 tasks.
- **Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).

### Secondary Metrics (Efficiency)
- **Simulated time:** Total simulation time (hardware-independent).
- **Distance navigated:** Accumulated distance traveled by the agent’s base body.
- **Displacement of end effectors/hands:** Accumulated displacement of the agent’s end effectors/hands.

*Secondary metrics will be normalized using human averages from 200 demonstrations per task.*

When running the eval script, an json file will be outputed after ech rollout episode containing the results. Here is a sample output json file for one episode of evaluation:

```
{
    "agent_distance": {
        "base": 9.703554042062024e-06, 
        "left": 0.019627160858362913, 
        "right": 0.015415858360938728
    }, 
    "normalized_agent_distance": {
        "base": 4.93031697036899e-06, 
        "left": 0.006022007241065448, 
        "right": 0.0037894888066205374
    }, 
    "q_score": {
        "final": 0.0
    }, 
    "time": {
        "simulator_steps": 6, 
        "simulator_time": 0.2, 
        "normalized_time": 0.002791165032284476
    }
}
```

** YOU ARE NOT ALLOWED TO MODIFY THE OUTPUT JSON IN ANY WAY **. Since each tasks will be evaluated on 20 instances and 5 rollout each, there should be 5k json files after the full evaluation. Zip them all and upload through google form. For privileged information track participants, zip your wrapper code and submit together with the result json files.