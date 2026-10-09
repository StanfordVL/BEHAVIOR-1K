# BEHAVIOR Challenge Updates

On this page, we provide updates regarding the **2026 BEHAVIOR Challenge**, including important bug fixes, new feature announcements, and clarifications about challenge rules.

---

### 10/07/2026 {#10072026}

**BEHAVIOR-1K v3.9.3.post2:**

This release contains a single bug fix for custom robot evaluation. If your submission does not use a custom robot, no update is required; you can continue using the previously specified release.

**Help us plan evaluation timeouts:**

Please fill out the [Inference Speed Form](https://forms.gle/5GNqc2eNhFTx4UhN8) with your policy type and expected average inference speed, including time spent on external API calls if applicable. Please include units such as seconds per inference or FPS.

We are planning a wall-clock timeout for each evaluation episode. Your responses will help us choose a limit that accommodates different policy runtimes while keeping evaluation manageable. Thank you!

---

### 09/28/2026 {#09282026}

**Final submission reminder:**

The challenge deadline is approaching: **October 16, 2026, at 11:59 PM Anywhere on Earth (AoE)**. Please review the following updates before submitting your final solution through the [submission portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit).

**BEHAVIOR-1K v3.9.3 release highlights:**

- **Vectorized environments:** Run multiple scenes and task instances concurrently through the standard `Environment` API.
- **Parallel policy evaluation:** Evaluate multiple challenge instances in batches for substantially higher throughput.
- **Improved evaluation tooling:** Support for batched policy transport and optional action-chunk replay.
- **Multi-environment correctness fixes:** More reliable task initialization, rewards, metrics, resets, state restoration, and demonstration replay.
- **Installation improvements:** More robust Isaac Sim wheel downloads, with clearer failures and retry behavior.

**Release correction:** The release to use is **v3.9.3-post1**, which includes an additional fix for the light-toggle state.

We accept final results evaluated on either **v3.9.2** or the **v3.9.3 series**. We will use **v3.9.3-post1** for final hidden-test evaluations. Extensive testing has confirmed consistent physics, visual behavior, and policy performance between v3.9.2 and v3.9.3.

For parallel evaluation, set `--num-envs` to match the number of instance indices evaluated together. Existing single-environment workflows remain supported with `--num-envs=1`.

**Tiled rendering is not recommended.** Although it can speed up evaluation, it introduces visual artifacts that may be out of distribution for your policy. We will **not** use tiled rendering for final evaluations.

**Additional guidelines and requirements:**

1. Ensure that all links in your submission have the necessary viewing permissions so we can inspect your solution and results.

2. For each task ID and instance ID, we will use the same policy-server IP address and port throughout the rollout to support history-dependent policies. For IP-based submissions, we will attempt to reconnect up to three times after a network disconnection. If reconnection fails, the rollout will count as a failure.

3. Due to limited compute, we will evaluate the top few submissions on hidden test instances before the November announcement. These hidden-test results will determine the final leaderboard ranking.

4. Partial submissions are allowed, including results for a subset of tasks or instances. Hidden-test evaluation will follow the tasks and instances covered by your submission; unsubmitted tasks and instances will count as zero.

5. We will introduce a global episode timer and a per-action response timer to ensure policies remain responsive. Details will be announced as soon as they are available.

**Venue:**

This year's challenge will be hosted at **CoRL 2026 in Austin, Texas, US, November 9–12, 2026**. Top submissions will have the opportunity to present at a workshop and booth. Details about the presentation format and invitations to attend will be sent after the submission deadline.

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
