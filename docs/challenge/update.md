## Weekly Update

At this page, we provide weekely update regarding the first BEHAVIOR challenge, this include important bug fixes, announcement on new features, clarification regarding challenge rules, etc.

### 09/19/2025

- Challenge rule clarifications:
    1. BDDL task definitions are allowed to use in standard track, these task definitions do not change across evaluation.
    2. Collecting more data by yourself (with teleoperation, RL, scripted policies, etc.) is allowed for standard track. Do note, however, that you are recommended against collecting data on the evaluation instances, those are meant to test the generalization capability of the submitted policy. 
    3. At this point, success score (Q) is the only metric we will be ranking teh submissions, if two submission has the same score, secondary metrics will be used to differentiate the two. 
    4. The timeout for each evaluation is set to be 2 * mean task completion time of the 200 human demos, thus it varies across tasks. 
    4. Besides the 200 human collected demos, we provided 20 additional configuration instances for each task. You should be using the first 10 instances to get the evaluation results (see TBD), the latter 10, however, are not used for evaluation. Feel free to use those as validation set before running your policy agains the first 10 evaluation instances. 

- Bug fixes:
    1. Fixed Windows installation script on latest main (65b6725)

- New features:
    1. We added a new tutorial regarding action type specification during evaluation, see [TBD]()