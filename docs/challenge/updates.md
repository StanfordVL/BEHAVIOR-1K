On this page, we provide weekly updates regarding the first BEHAVIOR Challenge, including important bug fixes, announcements of new features, and clarifications regarding challenge rules.


### 09/19/2025

**Challenge rule clarifications:**

1. BDDL task definitions are allowed in the standard track. These task definitions do not change across evaluation.

2. Collecting more data by yourself (with teleoperation, RL, scripted policies, etc.) is allowed for standard track. Do note, however, that you are recommended against collecting data on the evaluation instances, those are meant to test the generalization capability of the submitted policy. 

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