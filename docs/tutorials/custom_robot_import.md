---
icon: octicons/rocket-16
---

# 🕹️ **Importing a Custom Robot**

The page is currently under construction. 

## Preparation

In order to import a custom robot, You will need to first prepare your robot model file. For the next section we will assume you have the URDF file for the robots ready with all the corresponding meshes and textures. If your robot file is in another format (e.g. MCJF), please convert it to URDF format. If you already have the robot model USD file, feel free to skip the next section and move onto [Create the Robot Class](#create-the-robot-class).

Below, we will walk through each step for importing a new custom robot into **OmniGibson**. We use [Hello Robotic]()'s [Stretch]() robot as an example, taken directly from their [official repo]().

## Convert from URDF to USD

In this section, we will be using the URDF Importer in native Isaac Sim to convert our robot URDF model into USD format. Before we get started, it is strongly recommended that you read through the official [URDF Importer Tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/ext_omni_isaac_urdf.html). 

1. Create a directory with the name of the new robot under `<PATH_TO_OG_ASSET_DIR>/models`. This is where all of our robot models live. In our case, we created a directory named `stretch`.

2. Put your URDF file under this directory. Additional asset files such as STL, obj, mtl, and texture files should be placed under a `meshes` directory (see our `stretch` directory as an example).

3. Launch Isaac Sim from the Omniverse Launcher. In an empty stage, open the URDF Importer via `Isaac Utils` -> `Workflows` -> `URDF Importer`.

4. In the `Import Options`, uncheck `Fix Base Link` (we will have a parameter for this in OmniGibson). We also recommend that you check the `Self Collision` flag. You can leave the rest unchanged.

5. In the `Import` section, choose the URDF file that you moved in Step 1. You can leave the Output Directory as it is (same as source). Press import to finish the conversion. If all goes well, you should see the imported robot model in the current stage. In our case, the Stretch robot model looks like the following:


![Stretch Robot Import 0](../assets/tutorials/stretch-import-0.png)


## Process USD Model

Now that we have the USD model, let's open it up in Isaac Sim and inspect it.

1. In IsaacSim, begin by first Opening a New Stage. Then, Open the newly imported robot model USD file.

2. Make sure the default prim or root link of the robot has `Articulation Root` property

    Select the default prim in `Stage` panel on the top right, go to the `Property` section at the bottom right, scroll down to the `Physics` section, you should see the `Articulation Root` section. Make sure the `Articulation Enabled` is checked. If you dont't see the section, scroll to top of the `Property` section, and `Add` -> `Physics` -> `Articulation Root`
   
   ![Stretch Robot Import 2](../assets/tutorials/stretch-import-2.png)

3. Make sure every link has visual mesh and collision mesh in the correct shape. You can visually inspect this by clicking on every link in the `Stage` panel and view the highlighted visual mesh in orange. To visualize all collision meshes, click on the Eye Icon at the top and select `Show By Type` -> `Physics` -> `Colliders` -> `All`. This will outline all the collision meshes in green. If any collision meshes do not look as expected, please inspect the original collision mesh referenced in the URDF. Note that IsaacSim cannot import a pre-convex-decomposed collision mesh, and so such a collision mesh must be manually split and explicitly defined as individual sub-meshes in the URDF before importing. In our case, the Stretch robot model already comes with rough cubic approximations of its meshes.


   ![Stretch Robot Import 3](../assets/tutorials/stretch-import-3.png)

4. Make sure the physics is stable:

    - Create a fixed joint in the base: select the base link of the robot, then right click -> `Create` -> `Physics` -> `Joint` -> `Fixed Joint`
    
    - Click on the play button on the left toolbar, you should see the robot either standing still or falling down due to gravity, but there should be no abrupt movements.

    - If you observe the robot moving strangely, this suggests that there is something wrong with the robot physics. Some common issues we've observed are:
    
        - Self-collision is enabled, but the collision meshes are badly modeled and there are collision between robot links.

        - Some joints have bad damping/stiffness, max effort, friction, etc.

        - One or more of the robot links have off-the-scale mass values. 

    At this point, there is unfortunately no better way then to manually go through each of the individual links and joints in the Stage and examine / tune the parameters to determine which aspect of the model is causing physics problems. If you experience significant difficulties, please post on our [Discord channel](https://discord.gg/bccR5vGFEx).

5. The robot additionally needs to be equipped with sensors, such as cameras and / or LIDARs. To add a sensor to the robot, select the link under which the sensor should be attached, and select the appropriate sensor:

   - **LIDAR**: From the top taskbar, select `Create` -> `Isaac` -> `Sensors` -> `PhysX Lidar` -> `Rotating`
   - **Camera**: From the top taskbar, select `Create` -> `Camera`

   You can rename the generated sensors as needed. Note that it may be necessary to rotate / offset the sensors so that the pose is unobstructed and the orientation is correct. This can be achieved by modifying the `Translate` and `Rotate` properties in the `Lidar` sensor, or the `Translate` and `Orient` properties in the `Camera` sensor. Note that the local camera convention is z-backwards, y-up. Additional default values can be specified in each sensor's respective properties, such as `Clipping Range` and `Focal Length` in the `Camera` sensor.

   In our case, we created a LIDAR at the `laser` link (offset by 0.01m in the z direction), and cameras at the `camera_link` link (offset by 0.005m in the x direction and -90 degrees about the y-axis) and `gripper_camera_link` link (offset by 0.01m in the x direction and 90 / -90 degrees about the x-axis / y-axis). 

   ![Stretch Robot Import 5a](../assets/tutorials/stretch-import-5a.png)
   ![Stretch Robot Import 5b](../assets/tutorials/stretch-import-5b.png)
   ![Stretch Robot Import 5c](../assets/tutorials/stretch-import-5c.png)

6. Finally, save your USD!

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
