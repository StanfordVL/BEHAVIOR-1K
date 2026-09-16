# BEHAVIOR Challenge Updates

On this page, we provide updates regarding the **2026 BEHAVIOR Challenge**, including important bug fixes, new feature announcements, and clarifications about challenge rules.

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
