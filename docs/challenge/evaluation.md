# Evaluation and Challenge Rules

## Challenge Tracks

We will have the following two tracks for the 1st challenge:

### 1. Standard Track
- Participants are restricted to using the state observations provided in the demonstration dataset for their policy models.
- **Allowed observations:** RGB, depth, instance segmentation, proprioception information
- **Not allowed:** Object state

### 2. Privileged Information Track
- Participants may query the simulator for any privileged information (e.g., target object poses, scene point cloud) and use it for their policy models.

**Awards:**  
Top three teams from each track will share the challenge prizes and be invited to present at the workshop!

| Place | Prize |
|-------|-------|
| 🥇 1st | $1,000 |
| 🥈 2nd | $500   |
| 🥉 3rd | $300   |

---

## Performance Metrics

### Primary Metric (Ranking)
- **Task success score rate:** Averaged across 50 tasks.
- **Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates).

### Secondary Metrics (Efficiency)
- **Simulated time:** Total simulation time (hardware-independent).
- **Distance navigated:** Accumulated distance traveled by the agent’s base body.
- **Displacement of end effectors/hands:** Accumulated displacement of the agent’s end effectors/hands.

*Secondary metrics are normalized using human averages from 200 demonstrations per task.*

---

## Evaluation Protocol and Logistics

### Evaluation Protocol

1. **Training:**  
    - Training instances and 200 human demonstrations per task are released publicly.

2. **Self-Evaluation & Report:**  
    - 20 additional validation instances provided.
    - Evaluate your policy 5 times (with time-outs, via our evaluation script) per instance.
    - Submit scores using our Google Form (see below).
    - Leaderboard updated after sanity-check.

3. **Final Evaluation:**  
    - 20 held-out instances for final evaluation.
    - Leaderboard freezes on **November 15th, 2025**.
    - Top-5 solutions evaluated on these instances.



---


### Final Model Submission & Evaluation

- **Hardware:** Model should run on a single 24GB VRAM GPU (RTX 3090, A5000, TitanRTX).
- **Model Serving:** You may serve your models and provide IP addresses for evaluation. Recommended libraries: TorchServe, LitServe, vLLM, NVIDIA Triton, etc.
- **Entry Policy:** Multiple checkpoints from the same team = single entry.

---

## Challenge Office Hours

- **Every Monday and Thursday**
- **4:30pm–6:00pm PST**
- **Over Zoom:** [TBD]

