# Robots

## Overview
We provide a wide variety of **Robots** that can be imported in OmniGibson.

To quickly see all of them, you can play the example:
```
python -m omnigibson.examples.robots.all_robots_visualizer
```

Below, we list the fully-supported robots:

| Agent Name     | Robot Type(s) | DOF | Information      | Controller(s) |
|:-------------: | :-------------: | :-------------: |:-------------: |:-------------|
| Mujoco Ant      | `LocomotionRobot`     | 8     | [OpenAI Link](https://blog.openai.com/roboschool/) | Base: {Torque, Velocity, Position} |
| Husky Robot     | `LocomotionRobot`     | 4     | [ROS](http://wiki.ros.org/Robots/Husky), [Manufacturer](https://www.clearpathrobotics.com/) | Base: {Torque, Velocity, Position} |
| TurtleBot       | `TwoWheeledRobot`     | 2     | [ROS](http://wiki.ros.org/Robots/TurtleBot), [Manufacturer](https://www.turtlebot.com/) | Base: {Torque, Velocity, Position, Differential Drive} |
| Freight         | `TwoWheeledRobot`     | 2     | [Fetch Robotics Link](https://fetchrobotics.com/robotics-platforms/freight-base/) | Base: {Torque, Velocity, Position, Differential Drive} |
| Fetch           | `TwoWheeledRobot` <br> `ManipulationRobot` <br> `ActiveCameraRobot`     | 10    | [Fetch Robotics Link](https://fetchrobotics.com/robotics-platforms/freight-base/) | Base: {Torque, Velocity, Position, Differential Drive} <br> Camera: {Torque, Velocity, Position} <br> Arm: {Torque, Velocity, Position, Inverse Kinematics} <br> Gripper: {Torque, Velocity, Position, Binary} |
| JackRabbot      | `TwoWheeledRobot` <br> `ManipulationRobot`     | 2 & 7 | [Stanford Project Link](http://cvgl.stanford.edu/projects/jackrabbot/) | Base: {Torque, Velocity, Position, Differential Drive} <br> Arm: {Torque, Velocity, Position, Inverse Kinematics} |
| LocoBot         | `TwoWheeledRobot`     | 2     | [ROS](http://wiki.ros.org/locobot), [Manufacturer](https://www.trossenrobotics.com/locobot-pyrobot-ros-rover.aspx) | Base: {Torque, Velocity, Position, Differential Drive} |

Typically, these robot classes take in the URDF file or MuJoCo XML file of an robot (in `omnigibson.assets_path`) and provide a `load` function that be invoked externally (usually by `import_object` of `Simulator`). The `load` function imports the robot into PyBullet.

### Robot Class Hierarchy
All robot classes inherit from `BaseRobot`, which provides the core interface for all Robot classes. From `BaseRobot`, there are additional abstract subclasses from which a robot can inherit from:

```
BaseRobot -- all robots inherit from this class
|-- LocomotionRobot -- any robot with navigation functionality
  |-- TwoWheeledRobot -- any locomotion robot with two parallel wheels and differential drive functionality
|-- ManipulationRobot -- any robot with one or more arms
|-- ActiveCameraRobot -- any robot with a head or another controllable camera
```

Note that because these abstract classes describe different aspects of a robot's morphology, a robot class can inherit from multiple abstract classes. For example, `Fetch` inherits from `TwoWheeledRobot`, `ManipulationRobot`, and `ActiveCameraRobot`.

Each abstract robot class implements useful functions for controlling and accessing robot properties. For example, `ManipulationRobot` contains functionalities to query the state of the arms, and implements multiple grasping modes, including some simplified grasping modes like "sticky mitten" that could be used for researchers less interested on grasp-control and/or focused on task planning. For creating new robot classes to import custom robots, it is highly recommended to follow our robot hierarchy, to best leverage the features designed in our abstract classes.

How are robot parameters specified? Each abstract robot class expects certain kwargs, which are optionally extended for specific robot classes. While default values (seen in each respective robot class) are loaded at runtime, these can also be easily overridden by specifying these kwargs in the constructor or in the config file that you pass into the OmniGibson environment constructor. The set of modifiable arguments and expected robot config structure for each robot can be found in [omnigibson/configs/robots](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/configs/robots). For description of what each specific keyword argument corresponds to, please see the respective robot's class docstring.

Our examples contain multiple demonstrations on how to specify, create and initialize robots, either through config files and the Environment API, or manually with the constructors.

### Robot Control
OmniGibson v2.0 implements modular controllers that can be assigned to specific components of the robot. The controller classes can be found in [omnigibson/controllers](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/controllers). They include very generic controllers, such as `JointController`, and some more morphology-specific controllers, such as `DifferentialDriveController` (designed to control a two-wheeled robot) and `InverseKinematicsController` (designed to control a robot arm in Cartesian space using an inverse kinematics solver to find the right arm configuration).

Robots requests controllers of specific types based on the abstract classes they derive from. For example, a robot inheriting from `TwoWheeledRobot` requires loading a controller for the robot's `base`, and must be a `JointController` or `DifferentialDriveController`. A robot inheriting from `ManipulationRobot` requires loading a controller for each of the robot's `arm`s (`JointController` or `InverseKinematicsController`), and corresponding `gripper` (`JointController`, `MultiFingerGripperController`, `NullJointController`).

How are controller parameters specified? Each abstract robot class implements default controller configurations for each supported controller, which are automatically loaded at runtime (you can see the default configs directly in the abstract class source code, e.g.: the `InverseKinematicsController` defaults in [manipulation_robot.py](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/robots/manipulation_robot.py)). However, you can easily override these defaults and set specific parameters in your config file that you pass into the OmniGibson environment constructor. The set of modifiable arguments and expected controller config structure for each controller can be found in [omnigibson/configs/controllers](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/configs/controllers). For description of what each specific keyword argument corresponds to, please see the respective controller's class docstring.

We also include an example demo script showcasing controlling different robots. Please see [Robot Control Example](robots.md#robot-control-example) or run:
```
python -m omnigibson.examples.robots.robot_control_example
```
You can select the robot, the controllers, the input to the controllers (random or teleop with the keyboard) and the scene, and test them.

Some useful functions worth pointing out:
- `{get/set}_{position/orientation/rpy/linear_velocity/angular_velocity}`: get and set the physical states of the robot base
- `apply_robot_action`: set motor control for each of the controllable joints. It currently supports four modes of control: joint torque, velocity, position, and differential drive for two-wheeled robots
- `calc_state`: compute robot states that might be useful for external applications
- `reset`: reset the robot joint states to their default values, particularly useful for mobile manipulators. By default, this will correspond to the robot's `default_joint_pos` property. However, this can be overridden in multiple ways. You can either directly overwrite `default_joint_pos` to your own values, or, after calling `robot.reset()` normally, immediately set the joints to a new value, calling `robot.set_joint_states(values)`. This can be useful for robots that have multiple reset configurations -- for example, Fetch can be Tucked or Untucked. By default, `fetch.reset()` configures Fetch into Untucked mode. If you want to set Fetch to be in Tucked mode, you can call `fetch.set_joint_states(fetch.tucked_default_joint_pos)` (though for Fetch we directly provide this functionality with fetch.tuck() and fetch.untuck()).

### BehaviorRobot
The BehaviorRobot is a special "robot", and is intended to be used in virtual reality as an avatar or as an autonomous agent to participate in the BEHAVIOR100 challenge. The embodiment is composed of two hands, a torso, and a head link. It largely follows the convention of previous "URDF" based robots, but contains multiple URDFs connected by floating joints(constraints).

The BehaviorRobot has an action space of 26 DoF listed below.
- Torso: 6 DoF delta pose - relative to torso frame from the previous frame
- Head: 6 DoF delta pose - relative to torso frame (where the torso will be after applying this frame's action)
- Left hand, right hand (in this order): 6 DoF delta pose - relative to torso frame (where the torso will be after applying this frame's action)
- Grasping left hand, Grasping right hand (in this order): delta of change in the fraction of the grasping action (between 0=hand fully open, and 1=hand fully closed)

The reference frame of each body part is shown below.


![brobot](images/behavior_robot.jpg)


## Examples
We provide multiple examples showcasing our robots' functionality, described below. These examples, together with the provided config files, should help you getting started with all robots and controllers. All of these examples can be found in [omnigibson/examples/robots](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/examples/robots)

### Visualizing Robots Example
In [all_robots_visualizer.py](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/examples/robots/all_robots_visualizer.py), we iterate over all of our supported robots, loading them into a scene and applying random actions for a few seconds. This demo allows you to visualize all the robots and their corresponding DOFs in our OmniGibson GUI.

### Inverse Kinematics Example
In [ik_example.py](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/examples/robots/ik_example.py), we showcase using pybullet's built-in Inverse Kinematics (IK) control in an interactive way. We load Fetch and a visual marker in the pybullet GUI. You can then move a visual marker in the GUI, and apply IK to cause Fetch's arm to converge towards the marker.

### Motion Planning Example
In [motion_planning_example.py](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/examples/robots/motion_planning_example.py), we showcase using our motion planning module in an interactive way. We load Fetch into an empty building (`Rs_int` scene). You can interact with the GUI to set navigation and manipulation targets, which Fetch will converge to using our motion planner.

### Robot Control Example
In [robot_control_example.py](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/examples/robots/robot_control_example.py), we showcase using our controllers to control our different robots. You can choose a robot, and specific set of controllers to control the robot, and then either deploy random actions or directly teleoperate the robot using your keyboard.

## Legacy Robots
We also include robots originally supported in Gibson / OmniGibson v1.0, but have not been ported to OmniGibson v2.0.
They are included in [omnigibson/robots/legacy](https://github.com/StanfordVL/OmniGibson/blob/master/omnigibson/robots/legacy)
in their original unaltered state, and are not expected to work out of the box with the current OmniGibson environments.
Interested users should consider modifying those robot class logics to be compatible with OmniGibson v2.0, or their own
repositories. We are happy to accept community PR contributions standardizing these robot classes with OmniGibson v2.0!

Below, we list the legacy robots that can be found in OmniGibson:

| Agent Name     | DOF | Information      | Controller |
|:-------------: | :-------------: |:-------------: |:-------------|
| Mujoco Humanoid | 17    | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Minitaur Robot  | 8     | [Robot Page](https://www.ghostrobotics.io/copy-of-robots), [Manufacturer](https://www.ghostrobotics.io/) | Sine Controller |
| Quadrotor       | 6     | [Paper](https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations) | Torque |





