# :material-robot-industrial: **Importing a Custom Robot**

While OmniGibson assets includes a set of commonly-used robots, users might still want to import robot model of their own. This tutorial introduces users

## Preparation

In order to import a custom robot, You will need to first prepare your robot model file. For the next section we will assume you have the URDF file for the robots ready with all the corresponding meshes and textures. If your robot file is in another format (e.g. MJCF), please convert it to URDF format. If you already have the robot model USD file, feel free to skip the next section and move onto [Author the Robot Definition](#author-the-robot-definition).

!!! note "Data-driven robots (no Python subclass needed)"
    Since the Isaac Sim 5.1 migration, a robot is defined entirely by data — an imported USD plus a single `RobotDefinition` YAML at `<gm.DATA_PATH>/<dataset>/models/<name>/<name>.yaml`, which OmniGibson auto-discovers at startup. The old "write `stretch.py` and edit `robots/__init__.py`" workflow is obsolete; see [Author the Robot Definition](#author-the-robot-definition).

Below, we will walk through each step for importing a new custom robot into **OmniGibson**. We use [Hello Robotic](https://hello-robot.com/)'s [Stretch](https://hello-robot.com/stretch-3-product) robot as an example, taken directly from their [official repo](https://github.com/hello-robot/stretch_urdf).

## Convert from URDF to USD
There are two ways to convert our raw robot URDF into an OmniGibson-compatible USD file. The first is by using our integrated script, while the other method is using IsaacSim's native URDF-to-USD converter via the GUI. We highly recommend our script version, as it both wraps the same functionality from the underlying IsaacSim converter as well as providing useful features such as automatic convex decomposition of collision meshes, programmatic adding of sensors, etc.

### Option 1: Using Our 1-Liner Script (Recommended)
Our custom robot importer [`import_custom_robot.py`](https://github.com/StanfordVL/OmniGibson/tree/main/omnigibson/examples/robots/import_custom_robot.py) wraps the native URDF Importer from Isaac Sim to convert our robot URDF model into USD format. Please see the following steps for running this script:

1. All that is required is a single source config yaml file that dictates how the URDF should be post-processed when being converted into a USD. You can run `import_custom_robot.py --help` to see a detailed example configuration used, which is also shown below (`r1_pro_source_config.yaml`) for your convenience.
2. By default, output files are written to `<gm.DATA_PATH>/omnigibson-robot-assets/objects/robot/<name>/`. The destination dataset is configurable via the optional `dataset_name` key in the source config (default `omnigibson-robot-assets`). The imported asset is **not yet discoverable** at this location — it must be **copied** into the `models/` tree and paired with a `RobotDefinition` YAML. See [Register the Robot](#register-the-robot).

Some notes about the importing script:

- The importing procedure can be summarized as follows:

  1. Copy the raw URDF file + any dependencies into the output dataset directory (`<gm.DATA_PATH>/<dataset>/objects/robot/<name>/`, where `<dataset>` defaults to `omnigibson-robot-assets`)
  2. Updates the URDF + meshes to ensure all scaling is positive
  3. Generates collision meshes for each robot link (as specified by the source config)
  4. Generates metadata necessary for **OmniGibson**
  5. Converts the post-processed URDF into USD (technically, USDA) format
  6. Generates relevant semantic metadata for the given object, given its category
  7. Generates visual spheres, cameras, and lidars (in that order, as specified by the source config)
  8. Updates wheel approximations (as specified by the source config)
  9. Generates holonomic base joints (as specified by the source config)
  10. Generates configuration files needed for CuRobo motion planning (as specified by the source config)

- If `merge_fixed_joints=true`, all robot links that are connected to a parent link via a fixed joint will be merged directly into the parent joint. This means that the USD will _not_ contain these links! However, this collapsing occurs during the final URDF to USD conversion, meaning that these links _can_ be referenced beforehand (e.g.: when specifying desired per-link collision decomposition behavior)
- [CuRobo](https://curobo.org/index.html) is a highly performant motion planner that is used in **OmniGibson** for specific use-cases, such as skills. However, CuRobo requires a manually-specified sphere representation of the robot to be defined. These values can be generated using [IsaacSim's interactive GUI](https://curobo.org/tutorials/1_robot_configuration.html#robot-collision-representation), and should be exported and copied into the robot source config yaml file used for importing into **OmniGibson**. You can see the set of values used for the R1 robot below. For more information regarding specific keys specified, please see CuRobo's [Configuring a New Robot](https://curobo.org/tutorials/1_robot_configuration.html) tutorial.

??? code "r1_pro_source_config.yaml"
    ``` yaml linenums="1"

    urdf_path: r1_pro_source.urdf       # (str) Absolute path to robot URDF to import
    name: r1                            # (str) Name to assign to robot
    headless: false                     # (bool) if set, run without GUI
    overwrite: true                     # (bool) if set, overwrite any existing files
    merge_fixed_joints: false           # (bool) whether to merge fixed joints in the robot hierarchy or not
    base_motion:
      wheel_links:                      # (list of str): links corresponding to wheels
        - wheel_link1
        - wheel_link2
        - wheel_link3
      wheel_joints:                     # (list of str): joints corresponding to wheel motion
        - servo_joint1
        - servo_joint2
        - servo_joint3
        - wheel_joint1
        - wheel_joint2
        - wheel_joint3
      use_sphere_wheels: true           # (bool) whether to use sphere approximation for wheels (better stability)
      use_holonomic_joints: true        # (bool) whether to use joints to approximate a holonomic base. In this case, all
                                        #       wheel-related joints will be made into fixed joints, and 6 additional
                                        #       "virtual" joints will be added to the robot's base capturing 6DOF movement,
                                        #       with the (x,y,rz) joints being controllable by motors
    collision:
      decompose_method: coacd           # (str) [coacd, convex, or null] collision decomposition method
      hull_count: 8                     # (int) per-mesh max hull count to use during decomposition, only relevant for coacd
      coacd_links: []                   # (list of str): links that should use CoACD to decompose collision meshes
      convex_links:                     # (list of str): links that should use convex hull to decompose collision meshes
        - base_link
        - wheel_link1
        - wheel_link2
        - wheel_link3
        - torso_link1
        - torso_link2
        - torso_link3
        - torso_link4
        - left_arm_link1
        - left_arm_link4
        - left_arm_link5
        - right_arm_link1
        - right_arm_link4
        - right_arm_link5
      no_decompose_links: []            # (list of str): links that should not have any post-processing done to them
      no_collision_links:               # (list of str) links that will have any associated collision meshes removed
        - servo_link1
        - servo_link2
        - servo_link3
    eef_vis_links:                      # (list of dict) information for adding cameras to robot
      - link: left_eef_link             # same format as @camera_links
        parent_link: left_arm_link6
        offset:
          position: [0, 0, 0.06]
          orientation: [0, 0, 0, 1]
      - link: right_eef_link            # same format as @camera_links
        parent_link: right_arm_link6
        offset:
          position: [0, 0, 0.06]
          orientation: [0, 0, 0, 1]
    camera_links:                       # (list of dict) information for adding cameras to robot
      - link: eyes                      # (str) link name to add camera. Must exist if @parent_link is null, else will be
                                        #       added as a child of the parent
        parent_link: torso_link4        # (str) optional parent link to use if adding new link
        offset:                         # (dict) local pos,ori offset values. if @parent_link is specified, defines offset
                                        #       between @parent_link and @link specified in @parent_link's frame.
                                        #       Otherwise, specifies offset of generated prim relative to @link's frame
          position: [0.0732, 0, 0.4525]                     # (3-tuple) (x,y,z) offset -- this is done BEFORE the rotation
          orientation: [0.4056, -0.4056, -0.5792, 0.5792]   # (4-tuple) (x,y,z,w) offset
      - link: left_eef_link
        parent_link: null
        offset:
          position: [0.05, 0, -0.05]
          orientation: [-0.7011, -0.7011, -0.0923, -0.0923]
      - link: right_eef_link
        parent_link: null
        offset:
          position: [0.05, 0, -0.05]
          orientation: [-0.7011, -0.7011, -0.0923, -0.0923]
    lidar_links: []                     # (list of dict) information for adding cameras to robot
    curobo:
      eef_to_gripper_info:              # (dict) Maps EEF link name to corresponding gripper links / joints
        right_eef_link:
          links: ["right_gripper_link1", "right_gripper_link2"]
          joints: ["right_gripper_axis1", "right_gripper_axis2"]
        left_eef_link:
          links: ["left_gripper_link1", "left_gripper_link2"]
          joints: ["left_gripper_axis1", "left_gripper_axis2"]
      flip_joint_limits: []             # (list of str) any joints that have a negative axis specified in the
                                        #       source URDF
      lock_joints: {}                   # (dict) Maps joint name to "locked" joint configuration. Any joints
                                        #       specified here will not be considered active when motion planning
                                        #       NOTE: All gripper joints and non-controllable holonomic joints
                                        #       will automatically be added here
      self_collision_ignore:            # (dict) Maps link name to list of other ignore links to ignore collisions
                                        #       with. Note that bi-directional specification is not necessary,
                                        #       e.g.: "torso_link1" does not need to be specified in
                                        #       "torso_link2"'s list if "torso_link2" is already specified in
                                        #       "torso_link1"'s list
        base_link: ["torso_link1", "wheel_link1", "wheel_link2", "wheel_link3"]
        torso_link1: ["torso_link2"]
        torso_link2: ["torso_link3", "torso_link4"]
        torso_link3: ["torso_link4"]
        torso_link4: ["left_arm_link1", "right_arm_link1", "left_arm_link2", "right_arm_link2"]
        left_arm_link1: ["left_arm_link2"]
        left_arm_link2: ["left_arm_link3"]
        left_arm_link3: ["left_arm_link4"]
        left_arm_link4: ["left_arm_link5"]
        left_arm_link5: ["left_arm_link6"]
        left_arm_link6: ["left_gripper_link1", "left_gripper_link2"]
        right_arm_link1: ["right_arm_link2"]
        right_arm_link2: ["right_arm_link3"]
        right_arm_link3: ["right_arm_link4"]
        right_arm_link4: ["right_arm_link5"]
        right_arm_link5: ["right_arm_link6"]
        right_arm_link6: ["right_gripper_link1", "right_gripper_link2"]
        left_gripper_link1: ["left_gripper_link2"]
        right_gripper_link1: ["right_gripper_link2"]
      collision_spheres:                # (dict) Maps link name to list of collision sphere representations,
                                        #       where each sphere is defined by its (x,y,z) "center" and "radius"
                                        #       values. This defines the collision geometry during motion planning
        base_link:
          - "center": [-0.009, -0.094, 0.131]
            "radius": 0.09128
          - "center": [-0.021, 0.087, 0.121]
            "radius": 0.0906
          - "center": [0.019, 0.137, 0.198]
            "radius": 0.07971
          - "center": [0.019, -0.14, 0.209]
            "radius": 0.07563
          - "center": [0.007, -0.018, 0.115]
            "radius": 0.08448
          - "center": [0.119, -0.176, 0.209]
            "radius": 0.05998
          - "center": [0.137, 0.118, 0.208]
            "radius": 0.05862
          - "center": [-0.152, -0.049, 0.204]
            "radius": 0.05454
        torso_link1:
          - "center": [-0.001, -0.014, -0.057]
            "radius": 0.1
          - "center": [-0.001, -0.127, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.219, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.29, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.375, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.419, -0.064]
            "radius": 0.07
        torso_link2:
          - "center": [-0.001, -0.086, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.194, -0.064]
            "radius": 0.07
          - "center": [-0.001, -0.31, -0.064]
            "radius": 0.07
        torso_link4:
          - "center": [0.005, -0.001, 0.062]
            "radius": 0.1
          - "center": [0.005, -0.001, 0.245]
            "radius": 0.15
          - "center": [0.005, -0.001, 0.458]
            "radius": 0.1
          - "center": [0.002, 0.126, 0.305]
            "radius": 0.08
          - "center": [0.002, -0.126, 0.305]
            "radius": 0.08
        left_arm_link1:
          - "center": [0.001, 0.0, 0.069]
            "radius": 0.06
        left_arm_link2:
          - "center": [-0.062, -0.016, -0.03]
            "radius": 0.06
          - "center": [-0.135, -0.019, -0.03]
            "radius": 0.06
          - "center": [-0.224, -0.019, -0.03]
            "radius": 0.06
          - "center": [-0.31, -0.022, -0.03]
            "radius": 0.06
          - "center": [-0.34, -0.027, -0.03]
            "radius": 0.06
        left_arm_link3:
          - "center": [0.037, -0.058, -0.044]
            "radius": 0.05
          - "center": [0.095, -0.08, -0.044]
            "radius": 0.03
          - "center": [0.135, -0.08, -0.043]
            "radius": 0.03
          - "center": [0.176, -0.08, -0.043]
            "radius": 0.03
          - "center": [0.22, -0.077, -0.043]
            "radius": 0.03
        left_arm_link4:
          - "center": [-0.002, 0.0, 0.276]
            "radius": 0.04
        left_arm_link5:
          - "center": [0.059, -0.001, -0.021]
            "radius": 0.035
        left_arm_link6:
          - "center": [0.0, 0.0, 0.04]
            "radius": 0.04
        right_arm_link1:
          - "center": [0.001, 0.0, 0.069]
            "radius": 0.06
        right_arm_link2:
          - "center": [-0.062, -0.016, -0.03]
            "radius": 0.06
          - "center": [-0.135, -0.019, -0.03]
            "radius": 0.06
          - "center": [-0.224, -0.019, -0.03]
            "radius": 0.06
          - "center": [-0.31, -0.022, -0.03]
            "radius": 0.06
          - "center": [-0.34, -0.027, -0.03]
            "radius": 0.06
        right_arm_link3:
          - "center": [0.037, -0.058, -0.044]
            "radius": 0.05
          - "center": [0.095, -0.08, -0.044]
            "radius": 0.03
          - "center": [0.135, -0.08, -0.043]
            "radius": 0.03
          - "center": [0.176, -0.08, -0.043]
            "radius": 0.03
          - "center": [0.22, -0.077, -0.043]
            "radius": 0.03
        right_arm_link4:
          - "center": [-0.002, 0.0, 0.276]
            "radius": 0.04
        right_arm_link5:
          - "center": [0.059, -0.001, -0.021]
            "radius": 0.035
        right_arm_link6:
          - "center": [-0.0, 0.0, 0.04]
            "radius": 0.035
        wheel_link1:
          - "center": [-0.0, 0.0, -0.03]
            "radius": 0.06
        wheel_link2:
          - "center": [0.0, 0.0, 0.03]
            "radius": 0.06
        wheel_link3:
          - "center": [0.0, 0.0, -0.03]
            "radius": 0.06
        left_gripper_link1:
          - "center": [-0.03, 0.0, -0.002]
            "radius": 0.008
          - "center": [-0.01, 0.0, -0.003]
            "radius": 0.007
          - "center": [0.005, 0.0, -0.005]
            "radius": 0.005
          - "center": [0.02, 0.0, -0.007]
            "radius": 0.003
        left_gripper_link2:
          - "center": [-0.03, 0.0, -0.002]
            "radius": 0.008
          - "center": [-0.01, 0.0, -0.003]
            "radius": 0.007
          - "center": [0.005, 0.0, -0.005]
            "radius": 0.005
          - "center": [0.02, 0.0, -0.007]
            "radius": 0.003
        right_gripper_link1:
          - "center": [-0.03, 0.0, -0.002]
            "radius": 0.008
          - "center": [-0.01, -0.0, -0.003]
            "radius": 0.007
          - "center": [0.005, -0.0, -0.005]
            "radius": 0.005
          - "center": [0.02, -0.0, -0.007]
            "radius": 0.003
        right_gripper_link2:
          - "center": [-0.03, 0.0, -0.002]
            "radius": 0.008
          - "center": [-0.01, 0.0, -0.003]
            "radius": 0.007
          - "center": [0.005, 0.0, -0.005]
            "radius": 0.005
          - "center": [0.02, 0.0, -0.007]
            "radius": 0.003
      default_qpos:                               # (list of float): Default joint configuration
        - 0.0
        - 0.0
        - 0.0
        - 0.0
        - 0.0
        - 0.0
        - 1.906
        - 1.906
        - -0.991
        - -0.991
        - 1.571
        - 1.571
        - 0.915
        - 0.915
        - -1.571
        - -1.571
        - 0.03
        - 0.03
        - 0.03
        - 0.03

    ```


### Preprocessing a messy URDF

Vendor URDFs (especially CAD exports) often contain artifacts that break the importer. The helper module [`omnigibson/utils/urdf_preprocessing.py`](https://github.com/StanfordVL/OmniGibson/tree/main/omnigibson/utils/urdf_preprocessing.py) operates on a parsed `xml.etree.ElementTree`, so you can clean a URDF *before* importing it. Start by **auditing**:

```python
from omnigibson.utils.urdf_preprocessing import urdf_audit
print(urdf_audit("my_robot.urdf"))
# -> n_links / n_joints / joint_type_counts / mimic_joints / mesh_refs / mesh_formats /
#    package_uris / nonascii_paths / broken_relative_paths / links_without_inertial / root_link
```

Each cleaner helper mutates the tree in place and returns the number of changes it made. The one exception is `sanitize_names`, which returns its `{old: new}` rename map.

| Helper | Fixes | When you need it |
| --- | --- | --- |
| `rewrite_mesh_paths(tree, [(old, new), ...])` | Rewrites `<mesh filename>` substrings | Converting `package://` URIs or broken relative paths (`../../`) into absolute paths |
| `stage_meshes(tree, dest_dir)` | Copies every mesh into a clean dir under a **USD-safe basename** (space/paren -> `_`), dedupes shared sources | Mesh **filenames** contain spaces/parens — the importer names each geometry prim after the mesh basename, so a space yields an *ill-formed SdfPath* and the whole import fails. Run **after** `rewrite_mesh_paths` (needs resolvable paths) |
| `sanitize_names(tree)` | Replaces space/paren in **link & joint names** (and all parent/child/mimic refs) | The importer silently renames such links, breaking downstream name matching. Returns the rename map so you can update your config/definition references identically |
| `strip_mimic_joints(tree)` | Removes all `<mimic>` elements | The importer writes **no DriveAPI** on a mimic joint, leaving the mimicking finger non-drivable. Let the runtime gripper controller drive both fingers instead |
| `override_joint_limits(tree, {joint: {attr: val}})` | Sets `<limit>` attributes | Joints shipped with all-zero `<limit effort="0" ... velocity="0"/>` (common for grippers/wheels) are non-drivable in sim |
| `set_joint_dynamics(tree, joints, **attrs)` | Sets `<dynamics>` (e.g. `friction=0`) | Sluggish/jerky drive from CAD friction values |
| `link_has_dedicated_collision(link_el)` | Reports whether a link already has a real collision proxy | Used internally by the importer's auto-detect (see [Collision handling](#collision-handling)) |

A typical end-to-end pass (the recipe used for Gento Luna — `package://` URIs, spaces/parens in mesh and link/joint names, mimic joints, zeroed limits):

```python
import xml.etree.ElementTree as ET
from omnigibson.utils.urdf_preprocessing import (
    rewrite_mesh_paths, stage_meshes, strip_mimic_joints, sanitize_names,
    override_joint_limits, urdf_audit,
)

tree = ET.parse("vendor/robot.urdf")
rewrite_mesh_paths(tree, [("package://robot_description/", "/abs/robot_description/")])  # -> absolute
stage_meshes(tree, "staging/meshes_clean")            # -> space-free mesh basenames
strip_mimic_joints(tree)                              # -> drivable fingers
renames = sanitize_names(tree)                        # -> space-free link/joint names (keep the map!)
override_joint_limits(tree, {                         # -> repair zeroed limits
    renames.get("left finger_joint", "left finger_joint"): {"effort": 30, "velocity": 0.1, "lower": 0, "upper": 0.05},
})
tree.write("staging/robot.urdf", encoding="utf-8", xml_declaration=True)
assert urdf_audit("staging/robot.urdf")["package_uris"] == 0  # confirm it is now clean
```

Point your source config's `urdf_path` at the cleaned URDF. If `sanitize_names` renamed anything, use the **sanitized** names everywhere downstream (source config and definition YAML) — they are the values in the returned `renames` map.

### Option 2: Using IsaacSim's Native URDF-to-USD Converter
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

    Select the default prim in `Stage` panel on the top right, go to the `Property` section at the bottom right, scroll down to the `Physics` section, you should see the `Articulation Root` section. Make sure the `Articulation Enabled` is checked. If you don't see the section, scroll to top of the `Property` section, and `Add` -> `Physics` -> `Articulation Root`

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

6. Finally, save your USD (as a USDA file)! Note that you need to remove the fixed link created at step 4 before saving. Please save the file to `<gm.DATA_PATH>/omnigibson-robot-assets/models/<name>/usd/<name>.usda`.

## Register the Robot

The import script writes the converted asset to `<gm.DATA_PATH>/<dataset>/objects/robot/<name>/`, but OmniGibson discovers robots under the **`models/`** tree. Two steps make the robot loadable:

1. **Copy** the imported asset from `objects/robot/<name>/` into `models/<name>/`. Use the bundled helper (idempotent):

    ```python
    from omnigibson.utils.urdf_preprocessing import copy_robot_to_models
    from omnigibson.macros import gm
    copy_robot_to_models("my_robot", gm.DATA_PATH)  # dataset_name defaults to "omnigibson-robot-assets"
    ```

2. **Author** a `RobotDefinition` YAML (next section) and place it at `<gm.DATA_PATH>/<dataset>/models/<name>/<name>.yaml`. At startup, `REGISTERED_ROBOTS` globs `<gm.DATA_PATH>/*/models/*/*.yaml`, so a correctly-placed definition is auto-registered under its file stem — **no Python class and no `robots/__init__.py` edits required**. A single concrete `Robot` class is instantiated for *every* robot; capabilities are composed from whichever definition sub-config blocks are present (not from inheritance). You can then load the robot via `model: <name>` (lowercase) in any environment/robot config.

## Author the Robot Definition

A robot is described by a single `RobotDefinition` YAML (schema in [`definition_schema.py`](https://github.com/StanfordVL/OmniGibson/tree/main/omnigibson/robots/definition_schema.py)). Only `raw_controller_order` and `default_controllers` are required; each capability is opted into by adding its sub-config block. OmniGibson infers the robot's interfaces from which blocks are present — replacing the old multiple-inheritance class.

| Sub-config block | Enables | Key fields |
| --- | --- | --- |
| `manipulation` | Arm(s) + gripper(s) | `n_arms`, `arm_names`, `arm_{link,joint}_names`, `eef_link_names`, `finger_{link,joint}_names`, `arm_workspace_range` |
| `holonomic_base` | Holonomic (x, y, rz) base | `force_sphere_wheel_approximation` |
| `two_wheel` | Differential-drive base | `wheel_radius`, `wheel_axle_length` |
| `locomotion` | Base joints / floor contact | `base_joint_names`, `floor_touching_base_link_names` |
| `articulated_trunk` | Controllable torso | `trunk_joint_names`, `trunk_link_names` |
| `active_camera` | Controllable head/camera | `camera_joint_names` |
| `mobile_manipulation` | Tuck/untuck arm poses | `untucked_default_joint_pos`, `tucked_default_joint_pos` |

Top-level fields worth calling out:

- **`default_joint_pos`** — rest pose (one value per DOF). **Required if any arm uses an `InverseKinematicsController`** (it supplies the IK null-space reset pose); a missing value crashes at load. Zeros are a valid starting point to refine later.
- **`self_collisions`** (`true`/`false`/unset) — overrides the robot's default self-collision setting at load. Auto-generated hulls often overlap at rest, so NVIDIA's guidance is `false` (relying on planner spheres / `disabled_collision_pairs`). Leave unset to keep the default.
- **`disabled_collision_pairs`** / **`disabled_collision_link_names`** — filter specific colliding *pairs*, or disable all collision on named links. Prefer these with `self_collisions: true` to keep self-collision on for the rest of the robot.
- **`base_footprint_link_name`** — root link a synthesized holonomic base attaches to.

Below is the in-repo R1 definition — a complete, working dual-arm holonomic-base example:

??? code "r1.yaml (RobotDefinition)"
    ``` yaml linenums="1"
    raw_controller_order: ["base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]
    linear_velocity_gain_for_primitives: 0.3
    angular_velocity_gain_for_primitives: 0.2
    default_controllers:
      base: HolonomicBaseJointController
      trunk: JointController
      arm_left: InverseKinematicsController
      arm_right: InverseKinematicsController
      gripper_left: MultiFingerGripperController
      gripper_right: MultiFingerGripperController
    disabled_collision_pairs:
      - ["left_gripper_link1", "left_gripper_link2"]
      - ["right_gripper_link1", "right_gripper_link2"]
      - ["base_link", "wheel_link1"]
      - ["base_link", "wheel_link2"]
      - ["base_link", "wheel_link3"]
      - ["torso_link2", "torso_link4"]
    base_footprint_link_name: "base_link"

    holonomic_base:
      force_sphere_wheel_approximation: true

    locomotion:
      base_joint_names: ["base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint"]
      floor_touching_base_link_names: ["wheel_link1", "wheel_link2", "wheel_link3"]

    articulated_trunk:
      trunk_link_names: ["torso_link1", "torso_link2", "torso_link3", "torso_link4"]
      trunk_joint_names: ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]

    mobile_manipulation:
      untucked_default_joint_pos: [5.4849e-06, -1.2843e-05, 4.7727e-03, 2.5797e-03, 1.0873e-03, -2.1720e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.906, 1.906, -0.991, -0.991, 1.571, 1.571, 0.915, 0.915, -1.571, -1.571, 0.05, 0.05, 0.05, 0.05]
      tucked_default_joint_pos: [5.4849e-06, -1.2843e-05, 4.7727e-03, 2.5797e-03, 1.0873e-03, -2.1720e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05]

    manipulation:
      n_arms: 2
      arm_names: ["left", "right"]
      arm_link_names:
        left: ["left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4", "left_arm_link5", "left_arm_link6"]
        right: ["right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4", "right_arm_link5", "right_arm_link6"]
      arm_joint_names:
        left: ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6"]
        right: ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6"]
      eef_link_names:
        left: "left_eef_link"
        right: "right_eef_link"
      finger_link_names:
        left: ["left_gripper_link1", "left_gripper_link2"]
        right: ["right_gripper_link1", "right_gripper_link2"]
      finger_joint_names:
        left: ["left_gripper_axis1", "left_gripper_axis2"]
        right: ["right_gripper_axis1", "right_gripper_axis2"]
      arm_workspace_range:
        left: [-45, 45]
        right: [-45, 45]
    ```

If your robot is a manipulation robot, it additionally needs a **CuRobo** description YAML (for end-effector motion planning) under `models/<name>/curobo/`. When you import with `import_custom_robot.py`, **these are generated automatically** from the `curobo` block in your source config (shown in the R1 source config above) — you do not write them by hand. The generated file looks like:



??? code "r1_description_curobo_default.yaml"
    ``` yaml linenums="1"

    robot_cfg:
      base_link: base_footprint_x
      collision_link_names:
      - base_link
      - torso_link1
      - torso_link2
      - torso_link4
      - left_arm_link1
      - left_arm_link2
      - left_arm_link3
      - left_arm_link4
      - left_arm_link5
      - left_arm_link6
      - right_arm_link1
      - right_arm_link2
      - right_arm_link3
      - right_arm_link4
      - right_arm_link5
      - right_arm_link6
      - wheel_link1
      - wheel_link2
      - wheel_link3
      - left_gripper_link1
      - left_gripper_link2
      - right_gripper_link1
      - right_gripper_link2
      - attached_object_right_eef_link
      - attached_object_left_eef_link
      collision_sphere_buffer: 0.002
      collision_spheres:
        base_link:
        - center:
          - -0.009
          - -0.094
          - 0.131
          radius: 0.09128
        - center:
          - -0.021
          - 0.087
          - 0.121
          radius: 0.0906
        - center:
          - 0.019
          - 0.137
          - 0.198
          radius: 0.07971
        - center:
          - 0.019
          - -0.14
          - 0.209
          radius: 0.07563
        - center:
          - 0.007
          - -0.018
          - 0.115
          radius: 0.08448
        - center:
          - 0.119
          - -0.176
          - 0.209
          radius: 0.05998
        - center:
          - 0.137
          - 0.118
          - 0.208
          radius: 0.05862
        - center:
          - -0.152
          - -0.049
          - 0.204
          radius: 0.05454
        left_arm_link1:
        - center:
          - 0.001
          - 0.0
          - 0.069
          radius: 0.06
        left_arm_link2:
        - center:
          - -0.062
          - -0.016
          - -0.03
          radius: 0.06
        - center:
          - -0.135
          - -0.019
          - -0.03
          radius: 0.06
        - center:
          - -0.224
          - -0.019
          - -0.03
          radius: 0.06
        - center:
          - -0.31
          - -0.022
          - -0.03
          radius: 0.06
        - center:
          - -0.34
          - -0.027
          - -0.03
          radius: 0.06
        left_arm_link3:
        - center:
          - 0.037
          - -0.058
          - -0.044
          radius: 0.05
        - center:
          - 0.095
          - -0.08
          - -0.044
          radius: 0.03
        - center:
          - 0.135
          - -0.08
          - -0.043
          radius: 0.03
        - center:
          - 0.176
          - -0.08
          - -0.043
          radius: 0.03
        - center:
          - 0.22
          - -0.077
          - -0.043
          radius: 0.03
        left_arm_link4:
        - center:
          - -0.002
          - 0.0
          - 0.276
          radius: 0.04
        left_arm_link5:
        - center:
          - 0.059
          - -0.001
          - -0.021
          radius: 0.035
        left_arm_link6:
        - center:
          - 0.0
          - 0.0
          - 0.04
          radius: 0.04
        left_gripper_link1:
        - center:
          - -0.03
          - 0.0
          - -0.002
          radius: 0.008
        - center:
          - -0.01
          - 0.0
          - -0.003
          radius: 0.007
        - center:
          - 0.005
          - 0.0
          - -0.005
          radius: 0.005
        - center:
          - 0.02
          - 0.0
          - -0.007
          radius: 0.003
        left_gripper_link2:
        - center:
          - -0.03
          - 0.0
          - -0.002
          radius: 0.008
        - center:
          - -0.01
          - 0.0
          - -0.003
          radius: 0.007
        - center:
          - 0.005
          - 0.0
          - -0.005
          radius: 0.005
        - center:
          - 0.02
          - 0.0
          - -0.007
          radius: 0.003
        right_arm_link1:
        - center:
          - 0.001
          - 0.0
          - 0.069
          radius: 0.06
        right_arm_link2:
        - center:
          - -0.062
          - -0.016
          - -0.03
          radius: 0.06
        - center:
          - -0.135
          - -0.019
          - -0.03
          radius: 0.06
        - center:
          - -0.224
          - -0.019
          - -0.03
          radius: 0.06
        - center:
          - -0.31
          - -0.022
          - -0.03
          radius: 0.06
        - center:
          - -0.34
          - -0.027
          - -0.03
          radius: 0.06
        right_arm_link3:
        - center:
          - 0.037
          - -0.058
          - -0.044
          radius: 0.05
        - center:
          - 0.095
          - -0.08
          - -0.044
          radius: 0.03
        - center:
          - 0.135
          - -0.08
          - -0.043
          radius: 0.03
        - center:
          - 0.176
          - -0.08
          - -0.043
          radius: 0.03
        - center:
          - 0.22
          - -0.077
          - -0.043
          radius: 0.03
        right_arm_link4:
        - center:
          - -0.002
          - 0.0
          - 0.276
          radius: 0.04
        right_arm_link5:
        - center:
          - 0.059
          - -0.001
          - -0.021
          radius: 0.035
        right_arm_link6:
        - center:
          - -0.0
          - 0.0
          - 0.04
          radius: 0.035
        right_gripper_link1:
        - center:
          - -0.03
          - 0.0
          - -0.002
          radius: 0.008
        - center:
          - -0.01
          - -0.0
          - -0.003
          radius: 0.007
        - center:
          - 0.005
          - -0.0
          - -0.005
          radius: 0.005
        - center:
          - 0.02
          - -0.0
          - -0.007
          radius: 0.003
        right_gripper_link2:
        - center:
          - -0.03
          - 0.0
          - -0.002
          radius: 0.008
        - center:
          - -0.01
          - 0.0
          - -0.003
          radius: 0.007
        - center:
          - 0.005
          - 0.0
          - -0.005
          radius: 0.005
        - center:
          - 0.02
          - 0.0
          - -0.007
          radius: 0.003
        torso_link1:
        - center:
          - -0.001
          - -0.014
          - -0.057
          radius: 0.1
        - center:
          - -0.001
          - -0.127
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.219
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.29
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.375
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.419
          - -0.064
          radius: 0.07
        torso_link2:
        - center:
          - -0.001
          - -0.086
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.194
          - -0.064
          radius: 0.07
        - center:
          - -0.001
          - -0.31
          - -0.064
          radius: 0.07
        torso_link4:
        - center:
          - 0.005
          - -0.001
          - 0.062
          radius: 0.1
        - center:
          - 0.005
          - -0.001
          - 0.245
          radius: 0.15
        - center:
          - 0.005
          - -0.001
          - 0.458
          radius: 0.1
        - center:
          - 0.002
          - 0.126
          - 0.305
          radius: 0.08
        - center:
          - 0.002
          - -0.126
          - 0.305
          radius: 0.08
        wheel_link1:
        - center:
          - -0.0
          - 0.0
          - -0.03
          radius: 0.06
        wheel_link2:
        - center:
          - 0.0
          - 0.0
          - 0.03
          radius: 0.06
        wheel_link3:
        - center:
          - 0.0
          - 0.0
          - -0.03
          radius: 0.06
      cspace:
      - base_footprint_x_joint
      - base_footprint_y_joint
      - base_footprint_z_joint
      - base_footprint_rx_joint
      - base_footprint_ry_joint
      - base_footprint_rz_joint
      - torso_joint1
      - torso_joint2
      - torso_joint3
      - torso_joint4
      - left_arm_joint1
      - right_arm_joint1
      - left_arm_joint2
      - left_arm_joint3
      - left_arm_joint4
      - left_arm_joint5
      - left_arm_joint6
      - left_gripper_axis1
      - left_gripper_axis2
      - right_arm_joint2
      - right_arm_joint3
      - right_arm_joint4
      - right_arm_joint5
      - right_arm_joint6
      - right_gripper_axis1
      - right_gripper_axis2
      cspace_distance_weight:
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      ee_link: right_eef_link
      external_asset_path: null
      extra_collision_spheres:
        attached_object_left_eef_link: 32
        attached_object_right_eef_link: 32
      extra_links:
        attached_object_left_eef_link:
          fixed_transform:
          - 0
          - 0
          - 0
          - 1
          - 0
          - 0
          - 0
          joint_name: attached_object_left_eef_link_joint
          joint_type: FIXED
          link_name: attached_object_left_eef_link
          parent_link_name: left_eef_link
        attached_object_right_eef_link:
          fixed_transform:
          - 0
          - 0
          - 0
          - 1
          - 0
          - 0
          - 0
          joint_name: attached_object_right_eef_link_joint
          joint_type: FIXED
          link_name: attached_object_right_eef_link
          parent_link_name: right_eef_link
      link_names:
      - left_eef_link
      lock_joints:
        base_footprint_rx_joint: 0.0
        base_footprint_ry_joint: 0.0
        base_footprint_z_joint: 0.0
        left_gripper_axis1: 0.05000000074505806
        left_gripper_axis2: 0.05000000074505806
        right_gripper_axis1: 0.05000000074505806
        right_gripper_axis2: 0.05000000074505806
      max_acceleration: 15.0
      max_jerk: 500.0
      mesh_link_names:
      - base_link
      - torso_link1
      - torso_link2
      - torso_link4
      - left_arm_link1
      - left_arm_link2
      - left_arm_link3
      - left_arm_link4
      - left_arm_link5
      - left_arm_link6
      - right_arm_link1
      - right_arm_link2
      - right_arm_link3
      - right_arm_link4
      - right_arm_link5
      - right_arm_link6
      - wheel_link1
      - wheel_link2
      - wheel_link3
      - left_gripper_link1
      - left_gripper_link2
      - right_gripper_link1
      - right_gripper_link2
      - attached_object_right_eef_link
      - attached_object_left_eef_link
      null_space_weight:
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      - 1
      retract_config:
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0.0
      - 0.0
      - 0.0
      - 0.0
      - 0.0
      - 0.0
      - 1.906
      - 1.906
      - -0.991
      - -0.991
      - 1.571
      - 1.571
      - 0.915
      - 0.915
      - -1.571
      - -1.571
      - 0.03
      - 0.03
      - 0.03
      - 0.03
      self_collision_buffer:
        base_link: 0.02
      self_collision_ignore:
        base_link:
        - torso_link1
        - wheel_link1
        - wheel_link2
        - wheel_link3
        left_arm_link1:
        - left_arm_link2
        left_arm_link2:
        - left_arm_link3
        left_arm_link3:
        - left_arm_link4
        left_arm_link4:
        - left_arm_link5
        left_arm_link5:
        - left_arm_link6
        left_arm_link6:
        - left_gripper_link1
        - left_gripper_link2
        left_gripper_link1:
        - left_gripper_link2
        - attached_object_left_eef_link
        left_gripper_link2:
        - attached_object_left_eef_link
        right_arm_link1:
        - right_arm_link2
        right_arm_link2:
        - right_arm_link3
        right_arm_link3:
        - right_arm_link4
        right_arm_link4:
        - right_arm_link5
        right_arm_link5:
        - right_arm_link6
        right_arm_link6:
        - right_gripper_link1
        - right_gripper_link2
        right_gripper_link1:
        - right_gripper_link2
        - attached_object_right_eef_link
        right_gripper_link2:
        - attached_object_right_eef_link
        torso_link1:
        - torso_link2
        torso_link2:
        - torso_link3
        - torso_link4
        torso_link3:
        - torso_link4
        torso_link4:
        - left_arm_link1
        - right_arm_link1
        - left_arm_link2
        - right_arm_link2
      usd_flip_joint_limits: []
      usd_flip_joints: {}
      usd_robot_root: /r1
      use_global_cumul: true
    ```


## Collision handling

The importer **auto-detects** existing collision proxies (`link_has_dedicated_collision`): a link whose `<collision>` is a primitive or a mesh *different* from its visual mesh is **preserved**; a link is only regenerated when its collision is missing or identical to the visual mesh (regenerating a hull from a dense visual mesh balloons it and causes self-collision instability). The `collision.{coacd_links, convex_links, no_decompose_links, no_collision_links}` lists are **overrides** on top of this:

- Ships good collision meshes -> leave the lists empty (preserved automatically).
- Ships none -> pick `convex` (fast, one hull/link; good with `self_collisions: false`) or `coacd` (slower, multi-hull; `pip install coacd`). A single convex base hull + `use_sphere_wheels` keeps a mobile base flat.
- Put sensor/decoration frames (cameras, IMUs, F/T sensors) in `no_collision_links`.

Some additional notes:

- **Spaces/parens/non-ASCII in names or mesh filenames** are the most common hard failure: the importer renames offending links/joints and names each geometry prim after the mesh basename, so a space yields an ill-formed USD path and the import fails (surfacing as a `null prim` error). Run `stage_meshes` and `sanitize_names`, then re-`urdf_audit` until `package_uris`/`broken_relative_paths`/`mimic_joints`/`nonascii_paths` are `0`.
- **Isaac Sim 5.1 no longer merges fixed-joint links that carry mass/inertia** (4.x did). These leftover sub-parts can self-collide at rest — disable them individually via `disabled_collision_link_names`, unless the link needs collision (e.g. gripper fingertips).
- **IK arms require `default_joint_pos`**, or load crashes.
- **CAD exports often ship bad inertials / off-scale masses / zeroed limits** — audit and repair (`override_joint_limits`, `set_joint_dynamics`) first.

## Deploy Your Robot!

You can now test your custom robot by launching the control example:

```bash
python omnigibson/examples/robots/robot_control_example.py
```

Try different controller options and teleop the robot with your keyboard. If you observe poor joint behavior, inspect and tune the relevant joint parameters as needed — this also exposes bugs that may have crept in along the way (missing/bad joint limits, collisions, etc.). Please refer to the Franka or Fetch robots as a baseline for joint parameters that work well.

!!! tip "Validation gates"
    A quick programmatic check helps: load into `Rs_int`, step a few dozen frames with a **zero action** and assert the robot settles (max joint velocity -> ~0) without drifting, then nudge each controller's action slice and assert its joints respond. Note `get_joint_positions()`/`get_joint_velocities()` return **torch tensors** — `.detach().cpu().numpy()` before NumPy ops.

This is what our newly imported Stretch robot looks like in action:

 ![Stretch Import Test](../assets/tutorials/stretch-import-test.png)
