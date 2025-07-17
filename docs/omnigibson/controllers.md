# :material-knob: **Controllers**

## Description

In **`OmniGibson`**, `Controller`s convert high-level actions into low-level joint motor (position, velocity, or effort) controls for a subset of an individual [`Robot`](./robots.md)'s joints.

In an [`Environment`](./environments.md) instance, actions are passed to controllers via the `env.step(action)` call, resulting in the following behavior:

<div class="annotate" markdown>
- When `env.step(action)` is called, actions are parsed and passed to the respective robot owned by the environment (`env.robots`) via `robot.apply_action(action)`
- For a given robot, its `action` is parsed and passed to the respective controllers owned by the robot (`robot.controllers`) via `controller.update_goal(command)`
- For a given controller, the inputted `command` is preprocessed (re-scaled and shifted) and then converted into an internally tracked `goal`
- Each time a physic step occurs (1), all controllers computes and deploys their desired joint controls via `controller.compute_control()` towards reaching their respective `goal`s
</div>

1. Note that because environments operate at `action_frequency <= physics_frequency`, this means that a controller may take _multiple_ control steps per single `env.step(action)` call!

**`OmniGibson`** supports multiple types of controllers, which are intended to control a specific subset of a robot's set of joints. Some are more general (such as the `JointController`, which can broadly be applied to any part of a robot), while others are more specific to a robot's morphology (such as the `InverseKinematicsController`, which is intended to be used to control a manipulation robot's end-effector pose).

It is important to note that a single robot can potentially own multiple controllers. For example, `Turtlebot` only owns a single controller (to control its two-wheeled base), whereas the mobile-manipulator `Fetch` robot owns four (one to control its base, head, trunk + arm, and gripper). This allows for modular action space composition, where fine-grained modification of the action space can be achieved by modifying / swapping out individual controllers. For more information about the specific number of controllers each robot has, please see our [list of robots](./robots.md#types).

## Usage

### Definition

Controllers can be specified in the config that is passed to the `Environment` constructor via the `['robots'][i]['controller_config']` key. This is expected to be a nested dictionary, mapping controller name (1) to the desired specific controller configuration. the desired configuration for a single robot to be created. For each individual controller dict, the `name` key is required and specifies the desired controller class. Additional keys can be specified and will be passed directly to the specific controller class constructor. An example of a robot controller configuration is shown below in `.yaml` form:
{ .annotate }

1. See `robot.controller_order` for the full list of expected controller names for a given robot

??? code "single_fetch_controller_config_example.yaml"
    ``` yaml linenums="1"
    robots:
      - type: Fetch
        controller_config:
          base: 
            name: DifferentialDriveController
          arm_0:
            name: InverseKinematicsController
            kv: 2.0
          gripper_0:
            name: MultiFingerGripperController
            mode: binary
          camera:
            name: JointController
            use_delta_commands: False
    ```


### Runtime

Usually, actions are passed to robots, parsed, and passed to individual controllers via `env.step(action)` --> `robot.apply_action(action)` --> `controller.update_goal(command)`. However, specific controller commands can be directly deployed with this API outside of the `env.step()` loop. A controller's internal state can be cleared by calling `controller.reset()`, and no-op actions can computed via `compute_no_op_goal`.

Relevant properties, such as `control_type`, `control_dim`, `command_dim`, etc. are all queryable at runtime as well.


## Types
**`OmniGibson`** currently supports 6 controllers, consisting of 2 general joint controllers, 1 locomotion-specific controller, 2 arm manipulation-specific controllers, and 1 gripper-specific controller. Below, we provide a brief overview of each controller type:

### General Controllers
These are general-purpose controllers that are agnostic to a robot's morphology, and therefore can be used on any robot.

<table markdown="span">
    <tr>
        <td valign="top">
            [**`JointController`**](../reference/controllers/joint_controller.md)<br><br>
            Directly controls individual joints. Either outputs low-level joint position or velocity controls if `use_impedance=False`, otherwise will internally compensate the desired gains with the robot's mass matrix and output joint effort controls.<br><br>
            <ul>
                <li>_Command Dim_: n_joints</li>
                <li>_Command Description_: desired per-joint `[q_0, q_1, ...q_n]` position / velocity / effort setpoints, which are assumed to be absolute joint values unless `use_delta` is set</li>
                <li>_Control Dim_: n_joints</li>
                <li>_Control Type_: position / velocity / effort</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`NullJointController`**](../reference/controllers/null_joint_controller.md)<br><br>
            Directly controls individual joints via an internally stored `default_command`. Inputted commands will be ignored unless `default_command` is updated.<br><br>
            <ul>
                <li>_Command Dim_: n_joints</li>
                <li>_Command Description_: `[q_0, ..., q_n]` N/A </li>
                <li>_Control Dim_: n_joints</li>
                <li>_Control Type_: position / velocity / effort</li>
            </ul>
        </td>
    </tr>
</table>

### Locomotion Controllers
These are controllers specifically meant for robots with navigation capabilities.

<table markdown="span" width="100%">
    <tr>
        <td valign="top" width="100%">
            [**`DifferentialDriveController`**](../reference/controllers/dd_controller.md)<br><br>
            Commands 2-wheeled robots by setting linear / angular velocity setpoints and converting them into per-joint velocity control.<br><br>
            <ul>
                <li>_Command Dim_: n_joints</li>
                <li>_Command Description_: desired `[lin_vel, ang_vel]` setpoints </li>
                <li>_Control Dim_: 2</li>
                <li>_Control Type_: velocity</li>
            </ul>
        </td>
    </tr>
</table>


### Manipulation Arm Controllers
These are controllers specifically meant for robots with manipulation capabilities, and are intended to control a robot's end-effector pose

<table markdown="span">
    <tr>
        <td valign="top">
            [**`InverseKinematicsController`**](../reference/controllers/ik_controller.md)<br><br>
            Controls a robot's end-effector by iteratively solving inverse kinematics to output a desired joint configuration to reach the desired end effector pose, and then runs an underlying `JointController` to reach the target joint configuration. Multiple modes are available, and dictate both the command dimension and behavior of the controller. `condition_on_current_position` can be set to seed the IK solver with the robot's current joint state, and `use_impedance` can be set if the robot's per-joint inertia should be taken into account when attempting to reach the target joint configuration.<br><br>
            Note: Orientation convention is axis-angle `[ax,ay,az]` representation, and commands are expressed in the robot base frame unless otherwise noted.<br><br>
            <ul>
                <li>_Command Dim_: 3 / 6</li>
                <li>_Command Description_: desired pose command, depending on `mode`: <ul>
                      <li>`absolute_pose`: 6DOF `[x,y,z,ax,ay,az]` absolute position, absolute orientation</li>
                      <li>`pose_absolute_ori`: 6DOF `[dx,dy,dz,ax,ay,az]` delta position, absolute orientation</li>
                      <li>`pose_delta_ori`: 6DOF `[dx,dy,dz,dax,day,daz]` delta position, delta orientation</li>
                      <li>`position_fixed_ori`: 3DOF `[dx,dy,dz]` delta position, orientation setpoint is kept as fixed initial absolute orientation</li>
                      <li>`position_compliant_ori`: 3DOF `[dx,dy,dz]` delta position, delta orientation setpoint always kept as 0s (so can drift over time)</li>
            </ul></li>
                <li>_Control Dim_: n_arm_joints</li>
                <li>_Control Type_: position / effort</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`OperationalSpaceController`**](../reference/controllers/osc_controller.md)<br><br>
            Controls a robot's end-effector by applying the [operational space control](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf) algorithm to apply per-joint efforts to perturb the robot's end effector with impedances ("force") along all six (x,y,z,ax,ay,az) axes. Unlike `InverseKinematicsController`, this controller is inherently compliant and especially useful for contact-rich tasks or settings where fine-grained forces are required. For robots with >6 arm joints, an additional null command is used as a secondary objective and is defined as joint state `reset_joint_pos`.<br><br>
            Note: Orientation convention is axis-angle `[ax,ay,az]` representation, and commands are expressed in the robot base frame unless otherwise noted.<br><br>
            <ul>
                <li>_Command Dim_: 3 / 6</li>
                <li>_Command Description_: desired pose command, depending on `mode`: <ul>
                      <li>`absolute_pose`: 6DOF `[x,y,z,ax,ay,az]` absolute position, absolute orientation</li>
                      <li>`pose_absolute_ori`: 6DOF `[dx,dy,dz,ax,ay,az]` delta position, absolute orientation</li>
                      <li>`pose_delta_ori`: 6DOF `[dx,dy,dz,dax,day,daz]` delta position, delta orientation</li>
                      <li>`position_fixed_ori`: 3DOF `[dx,dy,dz]` delta position, orientation setpoint is kept as fixed initial absolute orientation</li>
                      <li>`position_compliant_ori`: 3DOF `[dx,dy,dz]` delta position, delta orientation setpoint always kept as 0s (so can drift over time)</li>
            </ul></li>
                <li>_Control Dim_: n_arm_joints</li>
                <li>_Control Type_: effort</li>
            </ul>
        </td>
    </tr>
</table>


### Manipulation Gripper Controllers
These are controllers specifically meant for robots with manipulation capabilities, and are intended to control a robot's end-effector gripper

<table markdown="span" width="100%">
    <tr>
        <td valign="top" width="100%">
            [**`MultiFingerGripperController`**](../reference/controllers/multi_finger_gripper_controller.md)<br><br>
            Commands a robot's gripper joints, with behavior defined via `mode`. By default, &lt;closed, open&gt; is assumed to correspond to &lt;q_lower_limit, q_upper_limit&gt; for each joint, though this can be manually set via the `closed_qpos` and `open_qpos` arguments.<br><br>
            <ul>
                <li>_Command Dim_: 1 / n_gripper_joints</li>
                <li>_Command Description_: desired gripper command, depending on `mode`: <ul>
                      <li>`binary`: 1DOF `[open / close]` binary command, where &gt;0 corresponds to open unless `inverted` is set, in which case &lt;0 corresponds to open</li>
                      <li>`smooth`: 1DOF `[q]` command, which gets broadcasted across all finger joints</li>
                      <li>`independent`: NDOF `[q_0, ..., q_n]` per-finger joint commands</li>
            </ul></li>
                <li>_Control Dim_: n_gripper_joints</li>
                <li>_Control Type_: position / velocity / effort</li>
            </ul>
        </td>
    </tr>
</table>
