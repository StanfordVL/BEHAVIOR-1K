# Evaluation and Rules

---

## Challenge Tracks

For the 1st BEHAVIOR-1K Challenge, We will have the following two tracks for the 1st challenge:

- **Standard track:** Participants are restricted to using the state observations we provided in the demonstration dataset for their policy models.
    - RGB + depth + segmentation + proprioception information
    - No object state

- **Privileged information track:** Participants are allowed to query the simulator for any privileged information, such as target object poses, scene point cloud, etc, and use such information for the policy models.

We will select the top three winning teams from each track, they will share the challenge prizes, and will be invited to present their approaches at the challenge workshop!
 🏆 Prizes for each track: 🥇 $1,000 🥈 $500 🥉 $300


## Running Evaluations

We provide a unified entry point for running evaluation:
```
python OmniGibson/omnigibson/eval.py policy=websocket log_path=$LOG_PATH task.name=$TASK_NAME env_wrapper._target_=$WRAPPER_MODULE
```
Here is a brief explanation of the arguments:

- `$LOG_PATH` is the path to where the evaluator will store the logs (metrics json file and recorded rollout videos)

- `$TASK_NAME` is the name of the task, a full list of tasks can be found in the demo gallery, as well as `TASK_TO_NAME_INDICES` under `OmniGibson/omnigibson/learning/utils/eval_utils.py`

- `$WRAPPER_MODULE` is the full module path of the environment wrapper that will be used. By default, running the following command will use `omnigibson.learning.wrappers.RGBLowResWrapper`:
    ```
    python OmniGibson/omnigibson/eval.py policy=websocket log_path=$LOG_PATH task.name=$TASK_NAME
    ```
which is a barebone wrapper that does not provide anything beyond low resolution rgb and proprioception info. There are three example wrappers under `omnigibson.learning.wrappers`:

    - `RGBLowResWrapper`: only use rgb as visual observation and camera resolutions of 224 * 224. Only using low-res RGB can help speed up the simulator and thus reduce evaluation time compared to the two other example wrappers. This wrapper is ok to use in standard track. 
    - `DefaultWrapper`: wrapper with the default observation config used during data collection (rgb + depth + segmentation, 720p for head camera and 480p for wrist camera). This wrapper is ok to use in standard track. 
    - `RichObservationWrapper`: this will load additional observation modalities, such as normal and flow, as well as privileged task information. This wrapper can only be used in privileged information track. 

After launching, the evaluator will load the task and spawn a server listening on `0.0.0.0:80`. The IP and port can be changed in `omnigibson/learning.configs/policy/websocket.yaml`. See `omnigibson/learning/configs/base_config.yaml` for more available arguments that you can overwrite. Feel free to use `omnigibson.learning.utils.network_utils.WebsocketPolicyServer` (adapted from [openpi](https://github.com/Physical-Intelligence/openpi)) to serve your policy and communicate with the Evaluator. 

You are welcome to use the wrappers we provided, or implement custom wrappers for your own use case. For privileged information track, you can arbitrarily query the environment instance for privileged information within the wrapper, as shown in the example `RichObservationWrapper`, which added `normal` and `flow` as additional visual observation modalities, as well as query for the pose of task relevant objects at every frame. We ask that you also include the wrapper code when submitting your result. The wrapper code will be manually inspected by our team to make sure the submission is on the right track, and you have not abused the environment by any means (e.g. teleporting the robot, or changing object states directly). 


As a starter, we provided a codebase of common imitation learning algorithms for you to get started. Please refer to the baselines section for more information.


## Metrics and Results

We will calculate the following metric during policy rollout:

### Primary Metric (Ranking)
- **Task success score:** Averaged across 50 tasks.
- **Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).

### Secondary Metrics (Efficiency)
- **Simulated time:** Total simulation time (hardware-independent).
- **Distance navigated:** Accumulated distance traveled by the agent’s base body. This metric evaluates the efficiency of the agent in navigating the environment.
- **Displacement of end effectors/hands:** Accumulated displacement of the agent’s end effectors/hands. This metric evaluates the efficiency of the agent in its interaction with the environment.

*Secondary metrics will be normalized using human averages from 200 demonstrations per task.*


## Evaluation Protocol and Logistics

**Evaluation protocol:**

- **Training:** The training instances and human demonstrations (200 per task) are released to the public.

- **Self-evaluation and report:** We have prepared 20 additional instances for validation. Participants should report their performance on the validation instances and submit their scores using our Google Form below. You should evaluate your policy 5 times (with time-outs, provided by our evaluation script) on each instance. We will update the leaderboard once we sanity-check the performance.

- **Final evaluation:** We will hold out 20 more instances for final evaluation. After we freeze the leaderboard on November 15th, 2025, we will evaluate the top-5 solutions on the leaderboard using these instances.

- Each instance differs in terms of:
    - Initial object states
    - Initial robot poses

<iframe 
  src="https://player.vimeo.com/video/1115082804?badge=0&autopause=0&autoplay=1&muted=1&loop=1&title=0&byline=0&portrait=0&controls=0" 
  width="640" 
  height="320" 
  frameborder="0" 
  allow="autoplay; fullscreen" 
  allowfullscreen>
</iframe>

**Submission details**

When running the eval script, an json file will be outputed after each rollout episode containing the results. Here is a sample output json file for one episode of evaluation:

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

- Submit your results and models at [Google Form](https://forms.gle/54tVqi5zs3ANGutn7).
    - You can view the leaderboard [here](./leaderboard.md).
    - We encourage you to submit intermediate results and models to be showcased on our leaderboard.

- Final model submission and evaluation:
    - Submitted models and our compute specs
        - The model should run on a single 24GB VRAM GPU. We will use the following GPUs to perform the final evaluation: RTX 3090, A5000, TitanRTX
    - IP address-based evaluation: You can serve your models and provide us with corresponding IP addresses that allow us to query your models for evaluation. We recommend common model serving libraries, such as [TorchServe](https://docs.pytorch.org/serve/), [LitServe](https://lightning.ai/docs/litserve/home), [vLLM](https://docs.vllm.ai/en/latest/index.html), [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), etc.
    - The same model with different checkpoints from the same team will be considered as a single entry.


** YOU ARE NOT ALLOWED TO MODIFY THE OUTPUT JSON IN ANY WAY **. Since each tasks will be evaluated on 20 instances and 5 rollout each, there should be 5k json files after the full evaluation. Zip them all and upload through google form. For privileged information track participants, zip your wrapper code and submit together with the result json files.


**Challenge office hours**

- Every Monday and Thursday, 4:30pm-6:00pm, PST, over [Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1).
