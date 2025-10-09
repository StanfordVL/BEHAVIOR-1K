# BEHAVIOR Challenge Weekly Updates

On this page, we provide weekly updates regarding the first **BEHAVIOR Challenge**, including important bug fixes, new feature announcements, and clarifications about challenge rules.

---

### 10/08/2025

**Challenge rule clarifications:**

1. During evaluation, only the task-relevant object poses and the robot’s initial pose will change.  
   The object instances and the poses of background, scene-level objects will remain the same. 
2. For standard track, you are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using them during evaluation.

**Bug fixes:**

1. Fixed a gripper joint range bug in `eval_utils.py`.  
2. Reverted assets from USDZ to USD format to improve loading speed.  
   Please re-download the assets to take advantage of this improvement.  
3. Fixed partial credit assignment during evaluation.
4. Fixed robot initial pose mismatch across evaluation rollouts.

All fixes have been pushed to the `main` branch.

**New features:**

1. We updated the [submission guideline](./submission.md) and added a [sample submission Dockerfile](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/docker/submission.Dockerfile) for reference.  
2. We are excited to announce that [NVIDIA](https://www.nvidia.com/en-us/) will be sponsoring our challenge!  
   The updated prize pool is as follows:
    - **First Place:** $1000 + GeForce 5080  
    - **Second Place:** $500 + (Jetson Orin Nano Super *or* $1000 Brev Credits)  
    - **Third Place:** $300 + $500 Brev Credits  

---

### 09/28/2025

**Challenge rule clarifications:**

1. No formal registration is required to participate in the challenge.  
   Feel free to submit your results directly if you have one!

**Bug fixes:**

1. Fixed multi-worker sharding and action chunk indexing in `BehaviorLeRobotDataset` under chunk streaming mode.  
2. Fixed incorrect robot start pose in the evaluation script.  
3. Provided improved baseline checkpoints.  
   Please refer to [baselines.md](./baselines.md) for details.

All fixes have been pushed to the `main` branch.

**New features:**

1. Added several new CLI arguments for evaluation, including  
   `testing_on_train_instances`, `max_steps`, and `partial_scene_load`.  
   See [base_config.yaml](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/configs/base_config.yaml) for more details.

---

### 09/19/2025

**Challenge rule clarifications:**

1. BDDL task definitions are allowed in the standard track.  
   These definitions are fixed and will remain the same during evaluation.  
2. You may collect additional data yourself (via teleoperation, RL, scripted policies, etc.) for the standard track.  
   However, you may **not** collect data on evaluation instances, as these are reserved for testing the generalization capability of your submitted policy.  
3. There are no restrictions on the type of policy used for either track.  
   Methods such as IL, RL, or TAMP are all allowed.  
   Additional components like SLAM or LLM-based querying are also permitted.  
4. Currently, the success score (**Q**) is the only metric used for ranking submissions.  
   If two submissions achieve the same score, secondary metrics will be used to break ties.  
5. The timeout for each evaluation is set to **2× the mean task completion time** of the 200 human demonstrations and thus varies across tasks.  
6. In addition to the 200 human-collected demonstrations, we provide 20 extra configuration instances for each task.  
   Use the **first 10** instances for evaluation results (see [evaluation.md](./evaluation.md#evaluation-protocol-and-logistics));  
   the **remaining 10** are not used for evaluation and may serve as a validation set before submitting your final policy.

**Bug fixes:**

1. Fixed the Windows installation setup script.  
2. Fixed timestamp type mismatch in `BehaviorLeRobotDataset`.  
3. Improved connection loss handling in `WebsocketClientPolicy`.  
4. Fixed various evaluation-related bugs.

All fixes have been pushed to the `main` branch.

**New features:**

1. Added a new tutorial on configuring the action space during evaluation.  
   See [evaluation.md](./evaluation.md#configure-robot-action-space) for details.
