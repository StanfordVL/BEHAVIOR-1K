# :material-list-box: **Tasks**

## Description

`Task`s define the high-level objectives that an agent must complete in a given `Environment`, subject to certain constraints (e.g. not flip over).

`Task`s have two important internal variables:

- `_termination_conditions`: a dict of {`str`: `TerminationCondition`} that define when an episode should be terminated. For each of the termination conditions, `termination_condition.step(...)` returns a tuple of `(done [bool], success [bool])`. If any of the termination conditions returns `done = True`, the episode is terminated. If any returns `success = True`, the episode is cnosidered successful.
- `_reward_functions`: a dict of {`str`: `RewardFunction`} that define how the agent is rewarded. Each reward function has a `reward_function.step(...)` method that returns a tuple of `(reward [float], info [dict])`. The `reward` is a scalar value that is added to the agent's total reward for the current step. The `info` is a dictionary that can contain additional information about the reward.

`Task`s usually specify task-relevant observations (e.g. goal location for a navigation task) via the `_get_obs` method, which returns a tuple of `(low_dim_obs [dict], obs [dict])`, where the first element is a dict of low-dimensional observations that will be automatically flattened into a 1D array, and the second element is everything else that shouldn't be flattened. Different types of tasks should overwrite the `_get_obs` method to return the appropriate observations.

`Task`s also define the reset behavior (in-between episodes) of the environment via the `_reset_scene`, `_reset_agent`, and `_reset_variables` methods. 

- `_reset_scene`: reset the scene for the next episode, default is `scene.reset()`.
- `_reset_agent`: reset the agent for the next episode, default is do nothing.
- `_reset_variables`: reset any internal variables as needed, default is do nothing.

Different types of tasks should overwrite these methods for the appropriate reset behavior, e.g. a navigation task might want to randomize the initial pose of the agent and the goal location.

## Usage

### Specifying
Every `Environment` instance includes a task, defined by its config that is passed to the environment constructor via the `task` key.
This is expected to be a dictionary of relevant keyword arguments, specifying the desired task configuration to be created (e.g. reward type and weights, hyperparameters for reset behavior, etc).
The `type` key is required and specifies the desired task class. Additional keys can be specified and will be passed directly to the specific task class constructor.
An example of a task configuration is shown below in `.yaml` form:

??? code "point_nav_example.yaml"
    ``` yaml linenums="1"
    task:
      type: PointNavigationTask
      robot_idn: 0
      floor: 0
      initial_pos: null
      initial_quat: null
      goal_pos: null
      goal_tolerance: 0.36    # turtlebot bodywidth
      goal_in_polar: false
      path_range: [1.0, 10.0]
      visualize_goal: true
      visualize_path: false
      n_vis_waypoints: 25
      reward_type: geodesic
      termination_config:
        max_collisions: 500
        max_steps: 500
        fall_height: 0.03
      reward_config:
        r_potential: 1.0
        r_collision: 0.1
        r_pointgoal: 10.0
    ```

### Runtime

`Environment` instance has a `task` attribute that is an instance of the specified task class.
Internally, `Environment`'s `reset` method will call the task's `reset` method, `step` method will call the task's `step` method, and the `get_obs` method will call the task's `get_obs` method.

## Types
**`OmniGibson`** currently supports 5 types of tasks, 7 types of termination conditions, and 5 types of reward functions.

### `Task`

<table markdown="span">
    <tr>
        <td valign="top">
            [**`DummyTask`**](../reference/tasks/dummy_task.md)<br><br>
            Dummy task with trivial implementations.
            <ul>
                <li>`termination_conditions`: empty dict.</li>
                <li>`reward_functions`: empty dict.</li>
                <li>`_get_obs()`: empty dict.</li>
                <li>`_reset_scene()`: default.</li>
                <li>`_reset_agent()`: default.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PointNavigationTask`**](../reference/tasks/point_navigation_task.md)<br><br>
            PointGoal navigation task with fixed / randomized initial pose and goal location.
            <ul>
                <li>`termination_conditions`: `MaxCollision`, `Timeout`, `PointGoal`.</li>
                <li>`reward_functions`: `PotentialReward`, `CollisionReward`, `PointGoalReward`.</li>
                <li>`_get_obs()`: returns relative xy position to the goal, and the agent's current linear and angular velocities.</li>
                <li>`_reset_scene()`: default.</li>
                <li>`_reset_agent()`: sample initial pose and goal location.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PointReachingTask`**](../reference/tasks/point_reaching_task.md)<br><br>
            Similar to PointNavigationTask, except the goal is specified with respect to the robot's end effector.
            <ul>
                <li>`termination_conditions`: `MaxCollision`, `Timeout`, `PointGoal`.</li>
                <li>`reward_functions`: `PotentialReward`, `CollisionReward`, `PointGoalReward`.</li>
                <li>`_get_obs()`: returns the goal position and the end effector's position in the robot's frame, and the agent's current linear and angular velocities.</li>
                <li>`_reset_scene()`: default.</li>
                <li>`_reset_agent()`: sample initial pose and goal location.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`GraspTask`**](../reference/tasks/grasp_task.md)<br><br>
            Grasp task for a single object.
            <ul>
                <li>`termination_conditions`: `Timeout`.</li>
                <li>`reward_functions`: `GraspReward`.</li>
                <li>`_get_obs()`: returns the object's pose in the robot's frame</li>
                <li>`_reset_scene()`: reset pose for objects in `_objects_config`.</li>
                <li>`_reset_agent()`: randomize the robot's pose and joint configurations.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`BehaviorTask`**](../reference/tasks/behavior_task.md)<br><br>
            BEHAVIOR task of long-horizon household activity.
            <ul>
                <li>`termination_conditions`: `Timeout`, `PredicateGoal`.</li>
                <li>`reward_functions`: `PotentialReward`.</li>
                <li>`_get_obs()`: returns the existence, pose, and in-gripper information of all task relevant objects</li>
                <li>`_reset_scene()`: default.</li>
                <li>`_reset_agent()`: default.</li>
            </ul>
        </td>
    </tr>
</table>

!!! info annotate "Follow our tutorial on BEHAVIOR tasks!"
    To better understand how to use / sample / load / customize BEHAVIOR tasks, please read our [BEHAVIOR tasks documentation](../behavior_components/behavior_tasks.md)!

### `TerminationCondition`
<table markdown="span">
    <tr>
        <td valign="top">
            [**`Timeout`**](../reference/termination_conditions/timeout.md)<br><br>
            `FailureCondition`: episode terminates if `max_step` steps have passed.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`Falling`**](../reference/termination_conditions/falling.md)<br><br>
            `FailureCondition`: episode terminates if the robot can no longer function (i.e.: falls below the floor height by at least
    `fall_height` or tilt too much by at least `tilt_tolerance`).
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`MaxCollision`**](../reference/termination_conditions/max_collision.md)<br><br>
            `FailureCondition`: episode terminates if the robot has collided more than `max_collisions` times.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PointGoal`**](../reference/termination_conditions/point_goal.md)<br><br>
            `SuccessCondition`: episode terminates if point goal is reached within `distance_tol` by the robot's base.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`ReachingGoal`**](../reference/termination_conditions/reaching_goal.md)<br><br>
            `SuccessCondition`: episode terminates if reaching goal is reached within `distance_tol` by the robot's end effector.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`GraspGoal`**](../reference/termination_conditions/grasp_goal.md)<br><br>
            `SuccessCondition`: episode terminates if target object has been grasped (by assistive grasping).
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PredicateGoal`**](../reference/termination_conditions/predicate_goal.md)<br><br>
            `SuccessCondition`: episode terminates if all the goal predicates of `BehaviorTask` are satisfied.
        </td>
    </tr>
</table>

### `RewardFunction`

<table markdown="span">
    <tr>
        <td valign="top">
            [**`CollisionReward`**](../reference/reward_functions/collision_reward.md)<br><br>
            Penalization of robot collision with non-floor objects, with a negative weight `r_collision`.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PointGoalReward`**](../reference/reward_functions/point_goal_reward.md)<br><br>
            Reward for reaching the goal with the robot's base, with a positive weight `r_pointgoal`.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`ReachingGoalReward`**](../reference/reward_functions/reaching_goal_reward.md)<br><br>
            Reward for reaching the goal with the robot's end-effector, with a positive weight `r_reach`.
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PotentialReward`**](../reference/reward_functions/potential_reward.md)<br><br>
            Reward for decreasing some arbitrary potential function value, with a positive weight `r_potential`.
            It assumes the task already has `get_potential` implemented.
            Generally low potential is preferred (e.g. a common potential for goal-directed task is the distance to goal).
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`GraspReward`**](../reference/reward_functions/grasp_reward.md)<br><br>
            Reward for grasping an object. It not only evaluates the success of object grasping but also considers various penalties and efficiencies.
            The reward is calculated based on several factors:
            <ul>
                <li>Grasping reward: A positive reward is given if the robot is currently grasping the specified object.</li>
                <li>Distance reward: A reward based on the inverse exponential distance between the end-effector and the object.</li>
                <li>Regularization penalty: Penalizes large magnitude actions to encourage smoother and more energy-efficient movements.</li>
                <li>Position and orientation penalties: Discourages excessive movement of the end-effector.</li>
                <li>Collision penalty: Penalizes collisions with the environment or other objects.</li>
            </ul>
        </td>
    </tr>
</table>
