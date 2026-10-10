# BEHAVIOR Challenge Updates

On this page, we provide updates regarding the **2026 BEHAVIOR Challenge**, including important bug fixes, new feature announcements, and clarifications about challenge rules.

---

### 10/09/2026 {#10092026}

**Final Submission Reminder & Evaluation Rules**

**Challenge rule clarifications:**

1. **Submission deadline:** October 16, 2026, at **11:59 PM Anywhere on Earth (AoE)**. We will use each team's **latest valid submission** received before the deadline, so please ensure it contains your complete final solution and results.

2. **Evaluation time budget:** Each task has a budget of `max_steps × 1 second` per rollout, excluding scene startup. When N rollouts share one GPU, each rollout is charged for its own policy queries plus 1/N of each shared simulator step. Waiting for another rollout's policy response is not charged to it. For the longest task, the per-rollout budget is approximately **10.85 hours** of accounted time.

3. **Per-step timeout:** Each action-query round trip must complete within **600 seconds** and within that rollout's remaining time budget.

4. **Reconnections:** If the policy-server connection is lost, we will attempt to reconnect up to **three times** before stopping the affected rollout and counting it as a failure. Reconnection time counts toward that rollout's time budget.

5. **IP-based submissions:** You are welcome to DM the organizers to schedule a connection and inference-speed test before your final submission.

6. **Leaderboard and final rankings:** To discourage leaderboard farming, we will hide leaderboard scores during the final week of submissions and restore their visibility after the deadline. For policies evaluated on hidden test instances, **hidden-test scores will replace public-instance scores** for final ranking.

7. **No cherry-picking results:** Evaluating the same task instances repeatedly and assembling a submission from the best result of each rollout is **not allowed**. Submit results from a single evaluation run of your final policy, with one rollout per submitted task instance.

---

### 10/07/2026 {#10072026}

**Challenge rule clarifications:**

1. If your submission uses a custom robot, please update to `v3.9.3-post2`. Otherwise, no update is required; you can continue using the previously specified release.

**Bug fixes:**

1. Released `v3.9.3-post2` with a single bug fix for custom robot evaluation.

**New features:**

1. Introduced the [Inference Speed Form](https://forms.gle/5GNqc2eNhFTx4UhN8) to help plan evaluation timeouts. Please provide your policy type and expected average inference speed, including time spent on external API calls if applicable, with units such as seconds per inference or FPS. Your responses will help us choose a wall-clock timeout for each evaluation episode that accommodates different policy runtimes while keeping evaluation manageable.

---

### 09/28/2026 {#09282026}

**Challenge rule clarifications:**

1. The submission deadline is **October 16, 2026, at 11:59 PM Anywhere on Earth (AoE)**. Please review these updates before submitting your final solution through the [submission portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit).

2. Please use `v3.9.3-post1`, which includes an additional light-toggle state fix over `v3.9.3`. We accept final results evaluated on either `v3.9.2` or the `v3.9.3` series, and will use `v3.9.3-post1` for final hidden-test evaluations. Extensive testing has confirmed consistent physics, visual behavior, and policy performance between v3.9.2 and v3.9.3.

3. **Tiled rendering is not recommended.** Although it can speed up evaluation, it introduces visual artifacts that may be out of distribution for your policy. We will **not** use tiled rendering for final evaluations.

4. Ensure that all links in your submission have the necessary viewing permissions so we can inspect your solution and results.

5. For each task ID and instance ID, we will use the same policy-server IP address and port throughout the rollout to support history-dependent policies. For IP-based submissions, we will attempt to reconnect up to three times after a network disconnection. If reconnection fails, the rollout will count as a failure.

6. Due to limited compute, we will evaluate the top few submissions on hidden test instances before the November announcement. These hidden-test results will determine the final leaderboard ranking.

7. Partial submissions are allowed, including results for a subset of tasks or instances. Hidden-test evaluation will follow the tasks and instances covered by your submission; unsubmitted tasks and instances will count as zero.

8. We will introduce a global episode timer and a per-action response timer to ensure policies remain responsive. Details will be announced as soon as they are available.

9. This year's challenge will be hosted at **CoRL 2026 in Austin, Texas, US, November 9–12, 2026**. Top submissions will have the opportunity to present at a workshop and booth. Details about the presentation format and invitations to attend will be sent after the submission deadline.

**Bug fixes:**

1. Fixed the light-toggle state in `v3.9.3-post1`.

2. Improved multi-environment correctness for task initialization, rewards, metrics, resets, state restoration, and demonstration replay.

3. Made Isaac Sim wheel downloads more robust, with clearer failures and retry behavior.

**New features:**

1. Added vectorized environments to run multiple scenes and task instances concurrently through the standard `Environment` API.

2. Added parallel policy evaluation to evaluate multiple challenge instances in batches for substantially higher throughput. Set `--num-envs` to match the number of instance indices evaluated together. Existing single-environment workflows remain supported with `--num-envs=1`.

3. Added support for batched policy transport and optional action-chunk replay.

---

### 08/24/2026 {#08242026}

**Challenge rule clarifications:**

1. Please use the `v3.9.2` tag of the `BEHAVIOR-1K` repository for challenge evaluation. It includes the fixes below.

**Bug fixes:**

1. Corrected the arm, gripper, and trunk velocity observations in the 2026 challenge demonstration dataset. These fields now use the raw simulator joint velocities from the original HDF5 demonstrations, and `meta/stats.json` has been recomputed accordingly. The affected fields are:

    - `state[10:17]`: `arm_left_qvel`
    - `state[26:28]`: `gripper_left_qvel`
    - `state[35:42]`: `arm_right_qvel`
    - `state[51:53]`: `gripper_right_qvel`
    - `state[57:61]`: `trunk_qvel`

    Actions and all other dataset fields are unchanged.

2. Updated partial-scene evaluation to load the exact room instances specified for each task in `B100_task_misc.csv`. This keeps the evaluation scene consistent with the challenge task metadata.

3. Fixed observation loading with `RGBDFullResWrapper` by refreshing simulator handles after changing camera resolutions and before rebuilding the observation space.

4. Fixed bugs affecting the challenge leaderboard and submission form.

**New features:**

1. Introduced a [participant registration form](https://forms.gle/Kf4ABLmDKbuK5Yhj6) for the 2026 BEHAVIOR Challenge.

---

### 07/27/2026 {#07272026}

**Challenge rule clarifications:**

1. Please use the `v3.9.2` tag of the `BEHAVIOR-1K` repository for evaluation and replay workflows, rather than the older `v3.9.0` tag. Since `v3.9.0`, `v3.9.2` includes important challenge updates, including LeRobot v3 / Hugging Face demo download instructions, evaluator Torch thread configuration, sponsor-page content, synchronized BDDL generated data, and synchronized asset synset metadata.

**Bug fixes:**

1. Updated the released demonstration dataset so `observation.state[0:3]` now records the R1Pro base velocity in the robot-local frame. Previously, these dimensions were populated from raw holonomic base joint velocities; the corrected values rotate the base x/y joint velocities by the base yaw and keep the yaw velocity as the third component. This matches the action convention used by the R1Pro base controller.
2. Fixed the released depth videos for the 2026 demonstration dataset. See the Hugging Face discussion for details: [behavior-1k/2026-challenge-demos discussion #2](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/discussions/2).

**New features:**

1. Added `meta/tasks.jsonl` with natural-language task descriptions for all 100 challenge tasks. The first 50 tasks follow the 2025 challenge descriptions with spelling/grammar fixes where needed; the remaining 50 were derived from the 2026 annotations and task definitions.
2. Uploaded per-episode language annotations for all 20,000 demonstrations.
