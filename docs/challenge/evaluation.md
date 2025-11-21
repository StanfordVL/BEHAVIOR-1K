# Evaluation and Rules

---

## Challenge Tracks

For the 1st BEHAVIOR-1K Challenge, We will have the following two tracks for the 1st challenge:

- **Standard track:** Participants are restricted to using the state observations we provided in the demonstration dataset for their policy models.
    - RGB + depth + segmentation (semantic, instance, instance id) + proprioception information (Note that robot global poses are NOT allowed)
    - No object state

- **Privileged information track:** Participants are allowed to query the simulator for any privileged information, such as target object poses, full-scene point cloud (Note: point cloud obtained from on-board vision sensor (e.g. estimated through rgbd) is fine for stanford track), robot global poses, etc, and use such information for the policy models.

For both tracks, you are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using them during standard track evaluation. BDDL task definitions can be used for both tracks and are identical during evaluation. You may also collect additional data yourself (via teleoperation, RL, scripted policies, etc.) for both tracks. However, you may **not** collect data on evaluation instances, as these are reserved for testing the generalization capability of your submitted policy.

There are no restrictions on the type of policy used for either track. Methods such as IL, RL, or TAMP are all allowed. Additional components like SLAM or LLM-based querying are also permitted.

We will select the top three winning teams from each track, they will share the challenge prizes, and will be invited to present their approaches at the challenge workshop!

 🏆 Prizes for each track: 

  1. 🥇 $1,000 + GeForce 5080
  2. 🥈 $500 + (Jetson Orin Nano Super or $1,000 Brev Credits)
  3. 🥉 $300 + $500 Brev Credits


## Running Evaluations

We provide a [evaluation script](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/eval.py) as a unified entry point for running evaluation:
```
python OmniGibson/omnigibson/learning/eval.py policy=websocket log_path=$LOG_PATH task.name=$TASK_NAME env_wrapper._target_=$WRAPPER_MODULE
```
Here is a brief explanation of the arguments:

- `$LOG_PATH` is the path to where the evaluator will store the logs (metrics json file and recorded rollout videos)

- `$TASK_NAME` is the name of the task, a full list of tasks can be found in the demo gallery, as well as `TASK_TO_NAME_INDICES` under `OmniGibson/omnigibson/learning/utils/eval_utils.py`

- `$WRAPPER_MODULE` is the full module path of the environment wrapper that will be used. By default, running the following command will use `omnigibson.learning.wrappers.RGBLowResWrapper`:
    ```
    python OmniGibson/omnigibson/learning/eval.py policy=websocket log_path=$LOG_PATH task.name=$TASK_NAME
    ```
which is a barebone wrapper that does not provide anything beyond low resolution rgb and proprioception info. There are three example wrappers under `omnigibson.learning.wrappers`:

    - `RGBLowResWrapper`: only use rgb as visual observation and camera resolutions of 224 * 224. Only using low-res RGB can help speed up the simulator and thus reduce evaluation time compared to the two other example wrappers. This wrapper is ok to use in both standard track and priveleged track.
    - `DefaultWrapper`: wrapper with the default observation config used during data collection (rgb + depth + segmentation, 720p for head camera and 480p for wrist camera). This wrapper is ok to use in standard track, but evaluation will be considerably slower compared to `RGBLowResWrapper`. 
    - `HeavyRobotWrapper`: wrapper with the RGB low resolution observation config, and heavy robot base mass (250kg) used during data collection. This wrapper is ok to use in both standard track and privileged track. 
    - `RichObservationWrapper`: this will load additional observation modalities, such as normal and flow, as well as privileged task information. This wrapper can only be used in privileged information track. 

There are some more optional arguments, see [base_config.yaml](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/configs/base_config.yaml). 

After launching, the evaluator will try to listen to `0.0.0.0:80`. The IP and port can be changed in `omnigibson/learning.configs/policy/websocket.yaml`. You can then use `omnigibson.learning.utils.network_utils.WebsocketPolicyServer` (adapted from [openpi](https://github.com/Physical-Intelligence/openpi)) to serve your policy and communicate with the Evaluator. 

There are other arguments that you can overwrite in CLI. For example, `partial_scene_load` will only load objects in the task-relevant rooms, which will help speed up loading and evaluation time. `testing_on_train_instances` will load the training instances instead of testing instances, which is helpful for debugging model overfitting. `max_steps` can be set to enable early stopping for rollouts (setting it to null will use the default timeout, which is 2 * average human task completion time within the demo dataset). See `omnigibson/learning/configs/base_config.yaml` for more available arguments that you can overwrite. 

You are welcome to use the wrappers we provided, or implement custom wrappers for your own use case. For privileged information track, you can arbitrarily query the environment instance for privileged information within the wrapper, as shown in the example `RichObservationWrapper`, which added `normal` and `flow` as additional visual observation modalities, as well as query for the pose of task relevant objects at every frame. We ask that you also include the wrapper code when submitting your result. The wrapper code will be manually inspected by our team to make sure the submission is on the right track, and you have not abused the environment by any means (e.g. teleporting the robot, or changing object states directly). 

As a starter, we provided a codebase of common imitation learning algorithms for you to get started. Please refer to the [baselines section](./baselines.md) for more information.


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
- **Task success score:** Averaged across 50 tasks.
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

- **Self-evaluation and report:** In addition to the 200 human-collected demonstrations, we provide 20 extra configuration instances for each task. Use the **first 10** instances for evaluation results. Participants should report their performance on theese 10 instances and submit their scores using our Google Form located at the [submission page](./submission.md). You should evaluate your policy 1 time (with time-outs = 2 * average task completion time within the dataset, provided by our evaluation script) on each instance. We will update the leaderboard once we sanity-check the performance. The **remaining 10** instances are not used for evaluation and may serve as a test set before evaluating your final policy.


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


**Challenge office hours**

- Every Monday and Thursday, 4:30pm-5:30pm, PST, over [Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1).

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
| RGB + depth + segmentation | 224x224, 224x224 | 16.55 |
| RGB + depth + segmentation | 720x720, 480x480 | 13.52 |
