# Submission Guidelines

<div class="challenge-portal-cta">
  <a href="https://behavior-1k-2026-challenge-leaderboard.hf.space/submit">Open Submission Portal</a>
</div>

## Submission Overview

- **No formal registration is required** to participate in the challenge.
- **Full evaluation size:** 100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs.
- **Partial submissions are allowed.** Missing rollout instances count as zero in the final score.
- **Multiple checkpoints** from the same team and model family are considered one entry.
- **No cherry-picking rollout results.** Simulation nondeterminism is expected, and different rollouts of the same policy may produce different results for a given instance. Submissions should report the prescribed rollout for each instance rather than selecting the best outcomes across repeated runs, instances, or tasks.
- **Robot configuration must be reproducible.** If you use a custom robot config with `--robot-config`, include the exact YAML/JSON file in your submission package.

## Evaluation Outputs

After running the evaluation script at `OmniGibson/omnigibson/eval/eval.py`, each rollout produces:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Metrics JSON</td>
      <td>Episode metrics, including task success score, normalized movement, and simulator time.</td>
    </tr>
    <tr>
      <td>Rollout MP4</td>
      <td>Video recording of the evaluated trajectory. Run the evaluator with <code>--write-video</code>; rollout videos are required for challenge submissions.</td>
    </tr>
  </tbody>
</table>

See [Evaluation and Rules](./evaluation.md) for the evaluation protocol, wrappers, metrics, and command-line options.

## Robot and Wrapper Configuration

Final evaluation must be reproducible from the files you submit. Include the exact evaluation wrapper and robot configuration used to generate your reported results.

- If you use the default robot setup, include the bundled `omnigibson/eval/r1pro.yaml` or clearly state that you used it unchanged.
- If you use a custom robot setup, include the complete robot config passed through `--robot-config`. The file must contain canonical OmniGibson robot fields such as `model`, `name`, `controller_config`, observation settings, sensor settings, and any `eval.camera_sensor_names` needed by the evaluation wrappers or video writer.
- The returned action array from your policy server must match the action space induced by the submitted robot config.
- Custom wrappers must still expose only RGB, depth, and proprioception to the policy for challenge-track evaluation.

??? example "Sample output JSON"

    ```json
    {
        "task": "turning_on_radio",
        "instance_id": 0,
        "rollout_id": 0,
        "steps": 500,
        "success": false,
        "agent_distance": {
            "base": 2.0,
            "left": 1.5,
            "right": 1.2
        },
        "normalized_agent_distance": {
            "base": 1.5,
            "left": 1.2,
            "right": 1.1
        },
        "q_score": {
            "final": 0.4
        },
        "time": {
            "simulator_steps": 500,
            "simulator_time": 16.6666666667,
            "normalized_time": 1.6
        }
    }
    ```

!!! warning "Do not edit evaluation outputs"

    Do not modify the output JSON files or rollout videos in any way.

## Final Model Evaluation

There are two supported ways to submit your model for final evaluation.

<div class="challenge-submission-grid">
  <section>
    <h3>Docker-based evaluation</h3>
    <p><strong>Recommended.</strong> Submit a Docker image that serves your policy. We run OmniGibson outside the container and connect to your policy through the WebSocket policy client.</p>
    <p>The submitted model should run on a single 24GB VRAM GPU. Final evaluation will use GPUs such as RTX 3090, A5000, and TitanRTX.</p>
  </section>
  <section>
    <h3>IP address-based evaluation</h3>
    <p>Serve your policy yourself and provide an IP address that allows us to query it for evaluation. We only accept IP address-based submissions that expose at least 64 ports for parallel evaluation.</p>
    <p>Common serving options include TorchServe, LitServe, vLLM, NVIDIA Triton, or an equivalent model-serving stack.</p>
  </section>
</div>


!!! note "Submission confidentiality and open source"

    Submitted solutions will remain confidential unless participants explicitly grant permission for disclosure. We strongly encourage open-source submissions, as they help advance reproducible research and accelerate progress in embodied AI.

### Docker Test Command

We provide a sample Dockerfile that starts a dummy local policy with zero actions: `OmniGibson/docker/submission.Dockerfile`.

1. Start an evaluation instance in another terminal:

    ```bash
    python -m omnigibson.eval.eval \
      --task-name turning_on_radio \
      --robot-config OmniGibson/omnigibson/eval/r1pro.yaml \
      --host 127.0.0.1 \
      --port 8000 \
      --instance-indices 0 \
      --output-dir outputs/b1k_eval
    ```

2. Build the sample image:

    ```bash
    docker build -f OmniGibson/docker/submission.Dockerfile -t b1k-challenge-example .
    ```

3. Run the container:

    ```bash
    docker run -p 8000:8000 b1k-challenge-example
    ```

!!! warning "Do not launch OmniGibson inside the submitted container"

    The Docker image may include OmniGibson as a utility library for policy code, but Isaac Sim is not installed in the container. For evaluation, OmniGibson runs outside the container and communicates with the policy server.

## Final Submission Package

Your final zip file should contain:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Metrics JSON files</td>
      <td>One JSON file for each rollout performed, up to 1,000 files.</td>
    </tr>
    <tr>
      <td>Wrapper code</td>
      <td>The <code>.py</code> wrapper used during evaluation.</td>
    </tr>
    <tr>
      <td>Robot config</td>
      <td>The exact robot <code>.yaml</code> or <code>.json</code> config used during evaluation, including any custom <code>controller_config</code>, sensor settings, observation settings, and <code>eval.camera_sensor_names</code>.</td>
    </tr>
    <tr>
      <td>README</td>
      <td>Instructions for evaluating your policy, including the full evaluator command, wrapper path, robot config path, Docker image details or IP address information. For IP address-based submissions, include at least 64 available ports.</td>
    </tr>
  </tbody>
</table>

In addition, submit a link through the [submission portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit) to all rollout MP4 videos, one for each rollout performed, up to 1,000 videos.
