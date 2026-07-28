# BEHAVIOR Challenge Updates

On this page, we provide updates regarding the **2026 BEHAVIOR Challenge**, including important bug fixes, new feature announcements, and clarifications about challenge rules.

---

### 07/27/2026 {#07272026}

**Bug fixes:**

1. Updated the released demonstration dataset so `observation.state[0:3]` now records the R1Pro base velocity in the robot-local frame. Previously, these dimensions were populated from raw holonomic base joint velocities; the corrected values rotate the base x/y joint velocities by the base yaw and keep the yaw velocity as the third component. This matches the action convention used by the R1Pro base controller.
2. Fixed the released depth videos for the 2026 demonstration dataset. See the Hugging Face discussion for details: [behavior-1k/2026-challenge-demos discussion #2](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/discussions/2).

**New features:**

1. Released the full 2026 challenge demonstration dataset on Hugging Face: [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos).
2. Added `meta/tasks.jsonl` with natural-language task descriptions for all 100 challenge tasks. The first 50 tasks follow the 2025 challenge descriptions with spelling/grammar fixes where needed; the remaining 50 were derived from the 2026 annotations and task definitions.
3. Uploaded per-episode language annotations for all 20,000 demonstrations.
