---
icon: octicons/rocket-16
---

# 🕹️ **Importing a Custom Robot**

The page is currently under construction. 

## Preparation

In order to import a custom robot, You will need to first prepare your robot model file. For the next section we will assume you have the URDF file for the robots ready with all the corresponding meshes and textures. If your robot file is in other format (e.g. MCJF), please convert it to URDF format. If you already have the USD file for the robot model, you can skip the next section and go straight to the one afterwards.

## Convert Robot from URDF to USD

In this section, we will be using the URDF Importer in native Isaac Sim to convert our robot URDF model into USD format. Before we get started, it is strongly recommended that you read through the official [URDF Importer Tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/ext_omni_isaac_urdf.html). 

1. Create a directory with the name of the new robot under `omnigibson/data/assets/models`

2. Put your URDF file under this directory, for the STL, obj, mtl and texture files you can create a `meshes` directory and put them under there.

3. Launch Isaac Sim from the Omniverse Launcher. Then open the URDF Importer via `Isaac Utils` -> `Workflows` -> `URDF Importer`. Make sure you are in an empty stage before doing the following.

4. In the Import Options, uncheck `Fix Base Link` (we will have a parameter for this in OmniGibson). It is also recommended that you check the `Self Collision`. You can leave the rest unchanged.

5. In the `Import` section, choose the URDF file that you just put under the models directory. You can leave the Output Directory as it is (same as source). Press import to finish the conversion.


## Process USD Model

Now that we have the USD model, let's open it up in Isaac Sim and inspect it. 

1. Make sure the default prim or root link of the robot has `Articulation Root` property

    Select the default prim in `Stage` panel on the top right, go to the `Property` section at the bottom right, scroll down to the `Physics` section, you should see the `Articulation Root` section. Make sure the `Articulation Enabled` is checked. If you dont't see the section, scroll to top of the `Property` section, and `Add` -> `Physics` -> `Articulation Root`

2. Make sure every link has visual mesh and collision mesh in the correct shape. You can visually inspect this by clicking on every link in the `Stage` panel and view the highlighted visual mesh in orange and the visualization of collision mesh in green meshes. 

3. Make sure the physics is stable:

    - Create a fixed joint in the base: select the base link of the robot, then right click -> `Create` -> `Physics` -> `Joint` -> `Fixed Joint`
    
    - Click on the play button on the left toolbar, you should see the robot either standing still or falling down due to gravity, but there should be no abrupt movements.

    - If you seen robot moving awkwardly, this means there is something wrong with the robot physics. Some common issues are:
    
        - You enabled self collision, but the collision meshes are badly modeled and there are collision between robot links.

        - Some joints has bad damping/stiffness, max effort, friction, etc.

        - One or more of the robot links have off-the-scale mass values. 

    At this point, there is really no other better way then manually go through all the links and joints, and play with the parameters to see which part of the robot is wrong. 

4. If you have successfully get the robot USD model, then you can proceed to the next section to create the python class for the robot.


## Create the Robot Class
Now that we have the USD file for the robot, let's write our own robot class. For more information please refer to the [`robot module`](../modules/robots.md)

1. Create a python file under `omnigibson/robots` folder, name it to the new robot's name.

2. We will build our class based on the `Franka Panda class` at `omnigibson/robots/franka.py`. You can copy over the content from that file for now.

3. Change the class name and the parent class it inherits from: take a look at the robot module to learn the different robot classes, we will assume that we want to inherit from `ManipulationRobot` in this case. 

4. Change the member functions of the class to match the new robot, here are some of the more important ones that you might want to implement:

    - `model_name`: change it to the desired model name of the new robot

    - `controller_order`, `_default_controllers`, `_default_gripper_multi_finger_controller_configs`... refer to the controller module to learn how to overwrite these functions.

    - `_default_joint_pos`: change it to the default joint positions of the robot (which will be the default configuration when the robot is first imported into OmniGibson scenes)

    - `default_arm`, `arm_link_names`, `arm_joint_names`, `eef_link_names`, `finger_link_names`, `finger_joint_names`: inspect the robot in native Isaac Sim and change the names to the correct link name. 

    - `arm_control_idx`, `gripper_control_idx`: TODO

    - `usd_path`, `urdf_path`: change it to the correct file path.

    - `disabled_collision_pairs`: you can put pairs of robot links in here and OmniGibson will ignore the collision between them. This is really useful if we have some bad collision modeling on certain links, and we can filter out these collisions without loosing much physics realism.

    - `assisted_grasp_start_points`, `assisted_grasp_end_points`: you need to implement this if you want to use sticky grasp/assisted grasp on the new robot.

4. Now add your robot class to `omnigibson/robots/__init__.py`. This way you will be able to import the robot in other scripts.

5. You can now try import and control the robot by launching `python omnigibson/examples/robot/robot_control_examples.py`!
