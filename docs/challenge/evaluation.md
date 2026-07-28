# Evaluation and Rules

## Challenge Track

For the 2026 BEHAVIOR Challenge, there is a single evaluation track:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Challenge track</td>
      <td>Participants are restricted to robot onboard observations for policy inputs.</td>
    </tr>
    <tr>
      <td>Allowed policy inputs</td>
      <td>RGB + depth + proprioception</td>
    </tr>
    <tr>
      <td>Robot embodiment</td>
      <td>Not fixed. Participants may use the default R1Pro robot or provide their own OmniGibson-supported robot through a custom robot configuration file.</td>
    </tr>
    <tr>
      <td>Not allowed during evaluation</td>
      <td>No ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only privileged information during evaluation.</td>
    </tr>
  </tbody>
</table>

You are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using it during challenge-track evaluation. BDDL task definitions can be used and are identical during evaluation. You may also collect additional data yourself via teleoperation, RL, scripted policies, or other approaches.

There are no restrictions on the type of policy used. Methods such as IL, RL, or TAMP are all allowed. Additional components like SLAM or LLM-based querying are also permitted, provided the policy follows the challenge-track observation restrictions during evaluation. If a submission depends on external model-query APIs, participants must provide the credentials, quota, and serving configuration needed for evaluation; the organizers will not cover external API usage costs.

## Running Evaluations

!!! warning "Use the latest `main` branch"
    Run challenge evaluation from the latest `main` branch of `BEHAVIOR-1K`, not the older `v3.9.0` tag. The challenge evaluator and dataset interfaces may receive compatibility fixes on `main`, and the organizers will use the latest `main` branch for evaluation.

We provide `OmniGibson/omnigibson/eval/eval.py` as the command-line entry point for running websocket-based evaluations. Start your policy server first, then run the evaluator from the repository root:

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

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td><code>--task-name</code></td>
      <td>BEHAVIOR task id, e.g. <code>turning_on_radio</code>. The 2026 task list is available in the <a href="./tasks/index.html">Demo Gallery</a>.</td>
    </tr>
    <tr>
      <td><code>--host</code> and <code>--port</code></td>
      <td>Address of the websocket policy server. The default port is <code>8000</code>. The evaluator waits for the server health check at <code>/healthz</code>, then opens the websocket connection.</td>
    </tr>
    <tr>
      <td><code>--instance-indices</code></td>
      <td>Indices into the task's test instance list. Use indices <code>0 1 2 3 4 5 6 7 8 9</code> for reported evaluation results. Indices <code>0-19</code> are public test instances; indices <code>20-39</code> are hidden instances reserved for final evaluation.</td>
    </tr>
    <tr>
      <td><code>--num-rollouts</code></td>
      <td>Number of rollouts to run for each selected instance.</td>
    </tr>
    <tr>
      <td><code>--max-steps</code></td>
      <td>Optional episode timeout in simulator steps. If omitted, the evaluator uses a task-specific default timeout equal to <code>1.5x</code> the mean human demonstration length.</td>
    </tr>
    <tr>
      <td><code>--env-wrapper</code></td>
      <td>Full target path of the evaluation wrapper. The default is <code>omnigibson.eval.wrappers.DefaultWrapper</code>; use <code>omnigibson.eval.wrappers.RGBDFullResWrapper</code> for official RGB + depth challenge-track evaluation.</td>
    </tr>
    <tr>
      <td><code>--output-dir</code></td>
      <td>Directory where rollout results are written. JSON metrics are written under <code>&lt;output-dir&gt;/json/</code>.</td>
    </tr>
    <tr>
      <td><code>--write-video</code></td>
      <td>Save rollout MP4 videos under <code>&lt;output-dir&gt;/videos/</code>. This is required for challenge evaluation outputs because rollout videos are part of the submission.</td>
    </tr>
    <tr>
      <td><code>--video-fps</code></td>
      <td>Frame rate for saved rollout videos.</td>
    </tr>
    <tr>
      <td><code>--headless</code> / <code>--no-headless</code></td>
      <td>Run OmniGibson headless or with rendering UI.</td>
    </tr>
  </tbody>
</table>

The evaluator sends flattened observations to the policy server. The server should return a msgpack-encoded response containing an `action` array with the robot action for the current step. The helper server implementation is `WebsocketPolicyServer` in `OmniGibson/omnigibson/eval/utils/network_utils.py`, and the evaluator-side client is `omnigibson.eval.policies.WebsocketPolicy`.

Each successful rollout produces a JSON result containing `q_score`, `time`, `agent_distance`, and normalized efficiency metrics. For challenge submissions, run evaluation with `--write-video`; this records the head and wrist camera videos that must be submitted with the rollout metrics.

Example wrappers live under `omnigibson.eval.wrappers`:

<div class="challenge-submission-grid">
  <section>
    <h3><code>DefaultWrapper</code></h3>
    <p>Low-resolution RGB observations at <code>224 x 224</code>, plus proprioception. This is useful for faster debugging but does not include depth.</p>
  </section>
  <section>
    <h3><code>RGBDFullResWrapper</code></h3>
    <p>Official RGB + depth challenge observations, with a <code>720 x 720</code> head camera and <code>480 x 480</code> wrist cameras.</p>
  </section>
</div>

You are welcome to use the provided wrappers or implement a custom wrapper for your own policy. Submitted evaluation wrappers must expose only RGB, depth, and proprioception to the policy. Include the wrapper code in your submission; the organizers will manually inspect it to ensure the challenge-track observation restrictions are followed and that the environment is not manipulated directly, e.g. by teleporting the robot or changing object states.

## Custom Robot Configuration

By default, the evaluator loads the bundled R1Pro robot config at `OmniGibson/omnigibson/eval/r1pro.yaml`. The 2026 challenge does not restrict participants to a fixed robot embodiment: you may instead pass a custom OmniGibson robot configuration file with `--robot-config`, as long as the robot and controllers are supported by OmniGibson and the policy still follows the challenge-track observation restrictions.

```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config path/to/my_robot.yaml \
  --host 127.0.0.1 \
  --port 8000
```

The config file should contain one complete robot dictionary using canonical OmniGibson fields. In particular:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td><code>model</code></td>
      <td>Required and should be the lowercase OmniGibson robot model id, e.g. <code>r1pro</code>. Use <code>model</code>, not the deprecated <code>type</code> key.</td>
    </tr>
    <tr>
      <td><code>name</code></td>
      <td>Required. The default name is <code>robot_r1</code>.</td>
    </tr>
    <tr>
      <td><code>controller_config</code></td>
      <td>Controls the action space. You may use any robot controllers supported by OmniGibson, such as joint, IK, base, or gripper controllers, as long as the resulting action array returned by your policy matches <code>robot.action_dim</code>.</td>
    </tr>
    <tr>
      <td>Standard robot fields</td>
      <td><code>obs_modalities</code>, <code>proprio_obs</code>, <code>sensor_config</code>, <code>action_normalize</code>, <code>grasping_mode</code>, and other standard robot config fields may be customized as needed, subject to the challenge-track observation restrictions.</td>
    </tr>
    <tr>
      <td>Runtime start pose</td>
      <td>The evaluator overwrites the robot <code>position</code> and <code>orientation</code> at runtime with the task instance's prescribed start pose.</td>
    </tr>
    <tr>
      <td>Submission requirement</td>
      <td>Include the exact robot config file in your final submission.</td>
    </tr>
  </tbody>
</table>

If your robot is not already available in OmniGibson, see the [custom robot import tutorial](../tutorials/custom_robot_import.md) for how to import a new robot model into BEHAVIOR / OmniGibson before referencing it from a custom robot config.

Minimal structure:

```yaml
model: r1pro
name: robot_r1
eval:
  camera_sensor_names:
    left_wrist: robot_r1:left_realsense_link:Camera:0
    right_wrist: robot_r1:right_realsense_link:Camera:0
    head: robot_r1:zed_link:Camera:0
obs_modalities:
  - proprio
  - rgb
controller_config:
  base:
    name: HolonomicBaseJointController
    motor_type: velocity
  arm_left:
    name: JointController
    motor_type: position
  arm_right:
    name: JointController
    motor_type: position
```

The optional `eval.camera_sensor_names` block maps evaluation camera roles to robot sensor names. It is used by the provided wrappers and video writer to identify the head and wrist cameras. `--write-video` requires the `head`, `left_wrist`, and `right_wrist` roles. The official `RGBDFullResWrapper` uses these roles to set the head camera to `720 x 720` and wrist cameras to `480 x 480`; all other vision sensors are treated as wrist-resolution sensors.

For controller syntax and supported controller types, see the [OmniGibson controller documentation](https://behavior.stanford.edu/omnigibson/controllers.html). For the recommended R1Pro baseline config, start from the bundled `OmniGibson/omnigibson/eval/r1pro.yaml` and modify `controller_config` or other robot fields as needed.


## Metrics and Results

We will calculate the following metric during policy rollout:

<div class="challenge-submission-grid">
  <section>
    <h3>Primary Metric (Ranking)</h3>
    <p><strong>Task success score:</strong> Averaged across 100 tasks.</p>
    <p><strong>Calculation:</strong> Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).</p>
  </section>
  <section>
    <h3>Secondary Metrics (Efficiency)</h3>
    <p><strong>Simulated time:</strong> Total simulation time (hardware-independent).</p>
    <p><strong>Distance navigated:</strong> Accumulated distance traveled by the agent’s base body. This metric evaluates the efficiency of the agent in navigating the environment.</p>
    <p><strong>Displacement of end effectors/hands:</strong> Accumulated displacement of the agent’s end effectors/hands. This metric evaluates the efficiency of the agent in its interaction with the environment.</p>
  </section>
</div>

*Secondary metrics will be normalized using human averages from 200 demonstrations per task.*

The success score (**Q**) is the metric used for ranking submissions. If two submissions achieve the same score, secondary metrics will be used to break ties. 

## Prizes

The 2026 BEHAVIOR Challenge has an $11,000 prize pool:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>1st place</td>
      <td>$5,000</td>
    </tr>
    <tr>
      <td>2nd place</td>
      <td>$3,000</td>
    </tr>
    <tr>
      <td>3rd place</td>
      <td>$2,000</td>
    </tr>
    <tr>
      <td>Outstanding open-source solution</td>
      <td>$1,000</td>
    </tr>
  </tbody>
</table>

## Evaluation Protocol and Logistics

**Evaluation protocol:**

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Training</td>
      <td>The training instances and human demonstrations (200 per task) are released to the public.</td>
    </tr>
    <tr>
      <td>Self-evaluation and report</td>
      <td>In addition to the 200 human-collected demonstrations, we provide 20 extra configuration instances for each task. Use the <strong>first 10 public instances</strong>, corresponding to instance indices <code>0-9</code> (<code>--instance-indices 0 1 2 3 4 5 6 7 8 9</code>), for evaluation results. Participants should report their performance on these 10 instances through the process described on the <a href="./submission.html">submission page</a>. You should evaluate your policy 1 time on each instance, using the default <code>1.5x</code> mean-human-length timeouts provided by our evaluation script. We will update the leaderboard once we sanity-check the performance. The <strong>remaining 10 public instances</strong>, indices <code>10-19</code>, are not used for leaderboard reporting and may serve as a test set before evaluating your final policy.</td>
    </tr>
    <tr>
      <td>Simulation nondeterminism</td>
      <td>Because the simulator can be nondeterministic, different rollouts of the same policy may produce different results for a given instance. This is expected. Participants should not cherry-pick rollout results for individual instances or assemble the best outcomes across runs, instances, or tasks to improve the reported success rate. The challenge uses many tasks and multiple instances per task to reduce the effect of rollout-level nondeterminism.</td>
    </tr>
    <tr>
      <td>Final evaluation</td>
      <td>We will hold out 10 more instances for final evaluation. After we freeze the leaderboard upon submission deadline, we will evaluate the top-5 solutions on the leaderboard using these instances.</td>
    </tr>
    <tr>
      <td>Instance variation</td>
      <td>Each instance differs in terms of initial object states and initial robot poses.</td>
    </tr>
  </tbody>
</table>

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

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>GPU</td>
      <td>NVIDIA RTX 4090 (24GB VRAM)</td>
    </tr>
    <tr>
      <td>CPU</td>
      <td>AMD Ryzen 9 7950X 16-Core Processor (32 threads)</td>
    </tr>
    <tr>
      <td>RAM</td>
      <td>128GB</td>
    </tr>
    <tr>
      <td>OS</td>
      <td>Ubuntu 22.04.5 LTS</td>
    </tr>
  </tbody>
</table>

**Scene Load Time:** Approximately 150-300 seconds (one-time cost per trial, varies by scene complexity)

### Evaluation Frame Rate with Random Actions

The following table records the approximate frames per second (FPS) performance when running evaluation with random actions across different settings:

<table class="challenge-data-table">
  <thead>
    <tr>
      <th>Sensor Modality</th>
      <th>Resolution (Head, Wrist)</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RGB</td>
      <td>224x224, 224x224</td>
      <td>24.55</td>
    </tr>
    <tr>
      <td>RGB</td>
      <td>720x720, 480x480</td>
      <td>20.62</td>
    </tr>
    <tr>
      <td>RGB + depth</td>
      <td>224x224, 224x224</td>
      <td>16.55</td>
    </tr>
    <tr>
      <td>RGB + depth</td>
      <td>720x720, 480x480</td>
      <td>13.52</td>
    </tr>
  </tbody>
</table>
