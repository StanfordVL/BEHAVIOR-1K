# Submission Guidelines

<div class="challenge-portal-cta">
  <a href="https://behavior-1k-2026-challenge-leaderboard.hf.space/submit">Open Submission Portal</a>
</div>

## Submission Overview

- **No formal registration is required** to participate in the challenge.
- **Full evaluation size:** 100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs.
- **Partial submissions are allowed.** Missing rollout instances count as zero in the final score.
- **Multiple checkpoints** from the same team and model family are considered one entry.

## Evaluation Outputs

After running the [evaluation script](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/eval/eval.py), each rollout produces:

<table class="challenge-data-table">
  <tbody>
    <tr>
      <td>Metrics JSON</td>
      <td>Episode metrics, including task success score, normalized movement, and simulator time.</td>
    </tr>
    <tr>
      <td>Rollout MP4</td>
      <td>Video recording of the evaluated trajectory.</td>
    </tr>
  </tbody>
</table>

See [Evaluation and Rules](./evaluation.md) for the evaluation protocol, wrappers, metrics, and command-line options.

??? example "Sample output JSON"

    ```json
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

We provide a sample Dockerfile that starts a dummy local policy with zero actions: [OmniGibson/docker/submission.Dockerfile](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/docker/submission.Dockerfile).

1. Start an evaluation instance in another terminal:

    ```bash
    python OmniGibson/omnigibson/learning/eval.py log_path=$LOG_PATH policy=websocket task.name=turning_on_radio
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
      <td>The R1Pro <code>.yaml</code> config used during evaluation.</td>
    </tr>
    <tr>
      <td>README</td>
      <td>Instructions for evaluating your policy, including Docker image details or IP address information. For IP address-based submissions, include at least 64 available ports.</td>
    </tr>
  </tbody>
</table>

In addition, submit a link through the [submission portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit) to all rollout MP4 videos, one for each rollout performed, up to 1,000 videos.
