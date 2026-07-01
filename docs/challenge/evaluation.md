# Evaluation and Rules

## Challenge Track

For the 2026 BEHAVIOR Challenge, there is a single evaluation track:

- **Challenge track:** Participants are restricted to robot onboard observations for policy inputs:
    - RGB + depth + proprioception
    - No ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only privileged information during evaluation.

You are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using it during challenge-track evaluation. BDDL task definitions can be used and are identical during evaluation. You may also collect additional data yourself via teleoperation, RL, scripted policies, or other approaches.

There are no restrictions on the type of policy used. Methods such as IL, RL, or TAMP are all allowed. Additional components like SLAM or LLM-based querying are also permitted, provided the policy follows the challenge-track observation restrictions during evaluation. If a submission depends on external model-query APIs, participants must provide the credentials, quota, and serving configuration needed for evaluation; the organizers will not cover external API usage costs.

## Running Evaluations

We provide [OmniGibson/omnigibson/eval/eval.py](https://github.com/StanfordVL/BEHAVIOR-1K/blob/my/eval/OmniGibson/omnigibson/eval/eval.py) as the command-line entry point for running websocket-based evaluations. Start your policy server first, then run the evaluator from the repository root:

```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 \
  --port 8000 \
  --instance-indices 0 \
  --num-rollouts 1 \
  --output-dir outputs/b1k_eval \
  --write-video
```

The evaluator connects to the policy server at `--host` and `--port`; the policy server is responsible for receiving observations and returning robot actions. The websocket interface is implemented by the evaluation utilities adapted from [openpi](https://github.com/Physical-Intelligence/openpi), and baseline servers such as OpenPI or GR00T can expose compatible endpoints.

Key arguments:

- `--task-name`: BEHAVIOR task id, e.g. `turning_on_radio`. The 2026 task list is available in the [Demo Gallery](./tasks/).
- `--host` and `--port`: address of the websocket policy server. The default port is `8000`. The evaluator waits for the server health check at `/healthz`, then opens the websocket connection.
- `--instance-indices`: indices into the task's test instance list. Indices `0-19` are public test instances; indices `20-39` are hidden instances reserved for final evaluation.
- `--num-rollouts`: number of rollouts to run for each selected instance.
- `--max-steps`: optional episode timeout in simulator steps. If omitted, the evaluator uses the default timeout based on human demonstration length.
- `--env-wrapper`: full target path of the evaluation wrapper. The default is `omnigibson.eval.wrappers.DefaultWrapper`; use `omnigibson.eval.wrappers.RGBDFullResWrapper` for official RGB + depth challenge-track evaluation.
- `--output-dir`: directory where rollout results are written. JSON metrics are written under `<output-dir>/json/`.
- `--write-video`: save rollout MP4 videos under `<output-dir>/videos/`.
- `--video-fps`: frame rate for saved rollout videos.
- `--headless` / `--no-headless`: run OmniGibson headless or with rendering UI.

The evaluator sends flattened observations to the policy server. The server should return a msgpack-encoded response containing an `action` array with the robot action for the current step. The helper server implementation is [WebsocketPolicyServer](https://github.com/StanfordVL/BEHAVIOR-1K/blob/my/eval/OmniGibson/omnigibson/eval/utils/network_utils.py), and the evaluator-side client is `omnigibson.eval.policies.WebsocketPolicy`.

Each successful rollout produces a JSON result containing `q_score`, `time`, `agent_distance`, and normalized efficiency metrics. If `--write-video` is enabled, the evaluator also records head and wrist camera videos for the rollout.

Example wrappers live under `omnigibson.eval.wrappers`:

- `DefaultWrapper`: low-resolution RGB observations at `224 x 224`, plus proprioception. This is useful for faster debugging but does not include depth.
- `RGBDFullResWrapper`: official RGB + depth challenge observations, with a `720 x 720` head camera and `480 x 480` wrist cameras.

You are welcome to use the provided wrappers or implement a custom wrapper for your own policy. Submitted evaluation wrappers must expose only RGB, depth, and proprioception to the policy. Include the wrapper code in your submission; the organizers will manually inspect it to ensure the challenge-track observation restrictions are followed and that the environment is not manipulated directly, e.g. by teleporting the robot or changing object states.

## Configure Robot Action Space

By default, the evaluator will take in absolute joint angles for all the robot joints (23-dim). Participants are allowed to modify the `controllers` section in the robot config yaml file [OmniGibson/omnigibson/learning/configs/robot/r1pro.yaml](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/configs/robot/r1pro.yaml) to suit their needs. By default the configuration is empty:

```
controllers:
```

Which is equivalant to absolute base velocity, absolute torso joint angles, absolute arm joint angles, 1-dim continuous gripper actions, as specified in [R1_CONTROLLER_CONFIG](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/joylo/gello/robots/sim_robot/og_teleop_cfg.py#L180-L232):


```
controllers:
  base:
    name: HolonomicBaseJointController
    motor_type: velocity
    vel_kp: 150
    command_input_limits: [[-1, -1, -1], [1, 1, 1]]
    command_output_limits: [[-0.75, -0.75, -1], [1, 1, 1]]
    use_impedances: false
  trunk:
    name: JointController
    motor_type: position
    pos_kp: 150
    command_input_limits: null
    command_output_limits: null
    use_impedances: false
    use_delta_commands: false
  arm_left:
    name: JointController
    motor_type: position
    pos_kp: 150
    command_input_limits: null
    command_output_limits: null
    use_impedances: false
    use_delta_commands: false
  arm_right:
    name: JointController
    motor_type: position
    pos_kp: 150
    command_input_limits: null
    command_output_limits: null
    use_impedances: false
    use_delta_commands: false
  gripper_left:
    name: MultiFingerGripperController
    mode: smooth
    command_input_limits: default
    command_output_limits: default
  gripper_right:
    name: MultiFingerGripperController
    mode: smooth
    command_input_limits: default
    command_output_limits: default
```

For more information regarding how to set robot controllers, please take a look at our [robot controller documentation](https://behavior.stanford.edu/omnigibson/controllers.html). The robot configuration yaml file (`r1pro.yaml`) needs to be included in your final submission. Below we provide some examples on modifying this config:
	
1. delta arm joint angles:

    ```
    arm_left:
      name: JointController
      motor_type: position
      pos_kp: 150
      command_input_limits: null
      command_output_limits: null
      use_impedances: false
      use_delta_commands: true
    ```

    Notice the change in `use_delta_commands`.

2. absolute EEF poses (in robot base frame) with IK Controller:

    ```
    arm_left:
      name: InverseKinematicsController
      command_input_limits: null
      command_output_limits: null
      mode: absolute_pose
    ```

3. delta EEF poses (in robot base frame) with IK Controller:
  
    ```
    arm_left:
      name: InverseKinematicsController
      command_input_limits: null
      mode: pose_delta_ori
    ```

    Notice the change in `mode`.

4.  absolute normalized gripper joint angles

    ```
    gripper_left:
      name: JointController
      motor_type: position
      command_input_limits: default
      command_output_limits: default
      use_impedances: false
      use_delta_commands: false
    ```


## Metrics and Results

We will calculate the following metric during policy rollout:

### Primary Metric (Ranking)
- **Task success score:** Averaged across 100 tasks.
- **Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).

### Secondary Metrics (Efficiency)
- **Simulated time:** Total simulation time (hardware-independent).
- **Distance navigated:** Accumulated distance traveled by the agent’s base body. This metric evaluates the efficiency of the agent in navigating the environment.
- **Displacement of end effectors/hands:** Accumulated displacement of the agent’s end effectors/hands. This metric evaluates the efficiency of the agent in its interaction with the environment.

*Secondary metrics will be normalized using human averages from 200 demonstrations per task.*

The success score (**Q**) is the metric used for ranking submissions. If two submissions achieve the same score, secondary metrics will be used to break ties. 

## Evaluation Protocol and Logistics

**Evaluation protocol:**

- **Training:** The training instances and human demonstrations (200 per task) are released to the public.

- **Self-evaluation and report:** In addition to the 200 human-collected demonstrations, we provide 20 extra configuration instances for each task. Use the **first 10** instances for evaluation results. Participants should report their performance on these 10 instances through the process described on the [submission page](./submission.md). You should evaluate your policy 1 time on each instance, using the default time-outs provided by our evaluation script. We will update the leaderboard once we sanity-check the performance. The **remaining 10** instances are not used for leaderboard reporting and may serve as a test set before evaluating your final policy.


- **Final evaluation:** We will hold out 10 more instances for final evaluation. After we freeze the leaderboard upon submission deadline, we will evaluate the top-5 solutions on the leaderboard using these instances.

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

## Performance Benchmarks

### System Spec

The following benchmarks were measured on:

- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **CPU:** AMD Ryzen 9 7950X 16-Core Processor (32 threads)
- **RAM:** 128GB
- **OS:** Ubuntu 22.04.5 LTS

**Scene Load Time:** Approximately 150-300 seconds (one-time cost per trial, varies by scene complexity)

### Evaluation Frame Rate with Random Actions

The following table records the approximate frames per second (FPS) performance when running evaluation with random actions across different settings:

| Sensor Modality | Resolution (Head, Wrist)| FPS |
|---------|------------|-----|
| RGB | 224x224, 224x224 | 24.55 |
| RGB | 720x720, 480x480 | 20.62 |
| RGB + depth | 224x224, 224x224 | 16.55 |
| RGB + depth | 720x720, 480x480 | 13.52 |
