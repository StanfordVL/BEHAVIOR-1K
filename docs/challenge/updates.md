On this page, we provide weekly updates regarding the first BEHAVIOR Challenge, including important bug fixes, announcements of new features, and clarifications regarding challenge rules.

### 10/06/2025

**Challenge rule clarifications:**

1. During evaluation, only task-relevant object pose and robot initial pose will change, the instance of the objects, pose of background scene-level objects will remain the same. 

**Bug fixes:**

1. Fixed gripper joint range bug in `eval_utils.py`.

2. Reverted assets from USDZ to USD for loading speed improvements. Please re-download the assets to take advantage of the speed improvements. 


All fixes have been pushed to the main branch.

**New features:**

1. We added a sample submission docker for reference. Please take a look at [this docker file](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/docker/submission.Dockerfile)



### 09/28/2025

**Challenge rule clarifications:**

1. There are no formal registration required to participate the challenge, feel free to submit your result directly if you have one!

**Bug fixes:**

1. Fixed multi-worker sharding and action chunk indexing in `BehaviorLeRobotDataset` under chunk streaming mode.

2. Fixed wrong robot start pose in evaluation script.

3. Provided better baselinbe checkpoints. Please refer to [baseline.md](./baselines.md)


All fixes have been pushed to the main branch.

**New features:**

1. We added some more CLI arguments for evaluation, including `testing_on_train_instances`, `early stopping`, `partial scene load`. Please take a look at [base_config.yaml](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/OmniGibson/omnigibson/learning/configs/base_config.yaml)



### 09/19/2025

**Challenge rule clarifications:**

1. BDDL task definitions are allowed in the standard track. These task definitions are fixed and will be the same during evaluation.

2. Collecting more data by yourself (with teleoperation, RL, scripted policies, etc.) is allowed for standard track. Do note, however, that you are not allowed to collect data on the evaluation instances, those are meant to test the generalization capability of the submitted policy. 

3. There are no restrictions on the form of the policy for both tracks. It could be IL, RL, TAMP, etc. Components like SLAM, querying LLMs, are also allowed. 

3. At this point, the success score (Q) is the only metric used for ranking submissions. If two submissions have the same score, secondary metrics will be used to break ties.

4. The timeout for each evaluation is set to be 2 * mean task completion time of the 200 human demos, thus it varies across tasks. 

5. Besides the 200 human collected demos, we provided 20 additional configuration instances for each task. You should use the first 10 instances to get the evaluation results (see [evaluation.md](./evaluation.md#evaluation-protocol-and-logistics)); the latter 10, however, are not used for evaluation. Feel free to use those as a validation set before running your policy against the first 10 evaluation instances.

**Bug fixes:**

1. Fixed Windows installation setup script.

2. Fixed timestamp miscast in `BehaviorLeRobotDataset`.

3. Better handling of connection loss in `WebsocketClientPolicy`.

4. Fixed various evaluation bugs.

All fixes have been pushed to the main branch.

**New features:**

1. We added a new tutorial regarding action space configuration during evaluation, see [evaluation.md](./evaluation.md#configure-robot-action-space)