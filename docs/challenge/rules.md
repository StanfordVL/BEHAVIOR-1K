# The 1st BEHAVIOR Challenge Rules

## Challenge Tracks

We will have the following two tracks for the 1st challenge:

- **Standard track:** Participants are restricted to using the state observations we provided in the demonstration dataset for their policy models.
    - RGB + depth + instance segmentation + proprioception information
    - No object state

- **Privileged information track:** Participants are allowed to query the simulator for any privileged information, such as target object poses, scene point cloud, etc, and use such information for the policy models.

We will select the top three winning teams from each track, they will share the challenge prizes, and will be invited to present their approaches at the challenge workshop!
 🏆 Prizes for each track: 🥇 $1,000 🥈 $500 🥉 $300

## Performance Metrics

**Primary metric (used for ranking): task success rate**, averaged across 50 tasks. Partial successes are counted as the number of goal BDDL predicates satisfied at the end of the episode, divided by the total number of goal predicates.

**Secondary metrics (measuring solution efficiency)** include the following:

- **Simulated time**: Total simulation steps, independent of the computer used.
- **Distance navigated**: Accumulated distance traveled by the agent’s base body. This metric evaluates the efficiency of the agent in navigating the environment.
- **Displacement of hands**: Accumulated displacement of the agent’s hands. This metric evaluates the efficiency of the agent in its interaction with the environment.

In order to compare secondary metrics across different tasks, we will normalize these using the **human averages** obtained from 200 human demonstrations for each task.

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

- Submit your results and models at [Google Form](https://forms.gle/54tVqi5zs3ANGutn7).
    - To self-report your performance, check our [evaluation guide](./evaluation.md). 
    - You can view the leaderboard [here](./leaderboard.md).
    - We encourage you to submit intermediate results and models to be showcased on our leaderboard.

- Final model submission and evaluation:
    - Submitted models and our compute specs
        - The model should run on a single 24GB VRAM GPU. We will use the following GPUs to perform the final evaluation: RTX 3090, A5000, TitanRTX
    - IP address-based evaluation: You can serve your models and provide us with corresponding IP addresses that allow us to query your models for evaluation. We recommend common model serving libraries, such as [TorchServe](https://docs.pytorch.org/serve/), [LitServe](https://lightning.ai/docs/litserve/home), [vLLM](https://docs.vllm.ai/en/latest/index.html), [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), etc.
    - The same model with different checkpoints from the same team will be considered as a single entry.


**Challenge office hours**

- Every Monday and Thursday, 4:30pm-6:00pm, PST, over [Zoom](https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1https://stanford.zoom.us/j/92909660940?pwd=RgFrdC8XeB3nVxABqb1gxrK96BCRBa.1).
