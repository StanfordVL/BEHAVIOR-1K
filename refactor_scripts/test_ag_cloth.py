from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.objects.dataset_object import DatasetObject
from igibson.robots.tiago import Tiago
from igibson.utils.constants import PrimType
import numpy as np


DISHTOWEL_FILE = f"{ig_dataset_path}/objects/dishtowel/Tag_Dishtowel_Basket_Weave_Red/usd/Tag_Dishtowel_Basket_Weave_Red.usd"
DISH_TOWEL_ORN = np.array([0.5, 0., 0., 0.8660254])
ARM = "left"
DIST_THRESHOLD = 0.15

sim = Simulator()

# Load scene
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

robot = Tiago("/World/Tiago", grasping_mode="sticky", obs_modalities=[])
sim.import_object(robot)

dishtowel = DatasetObject(
    "/World/dishtowel_0",
    usd_path=DISHTOWEL_FILE,
    name="dishtowel_0",
    category="dishtowel",
    prim_type=PrimType.CLOTH,
)
sim.import_object(dishtowel)
dishtowel.set_position([0, 0, 10])

# Initialize the robot and the object
sim.play()
robot.reset()
robot.untuck()
sim.step()
container_box_pos = robot.eef_links[ARM].visual_meshes["container_box"].get_position()
sim.stop()

# Set the dishtowel to be inside the gripper
dishtowel.set_position(container_box_pos)
dishtowel.set_orientation(DISH_TOWEL_ORN)

sim.play()
robot.reset()
robot.untuck()
sim.step()

# Close the gripper a bit more so that after exactly one sim step, the gripper fingers are close enough to activate AG
for arm in robot.arm_names:
    for finger_joint_name in robot.finger_joint_names[arm]:
        finger_joint = robot.joints[finger_joint_name]
        finger_joint.set_pos(0.02)

action = np.zeros(robot.action_dim)
gripper_idx = robot.controller_action_idx["gripper_{}".format(ARM)]

# apply grasping
action[gripper_idx] = -1.0

robot.apply_action(action)
sim.step()
assert robot._ag_obj_in_hand[ARM] is None, "AG should not activate because the fingers are still too far apart."

robot.apply_action(action)
sim.step()
assert robot._ag_obj_in_hand[ARM] is not None, "AG should have activated because the fingers should be close enough."

# Start moving the arm joints and AG should always be on
for arm_joint in robot.controller_action_idx["arm_left"]:
    action[:] = 0.0
    action[gripper_idx] = -1.0

    action[arm_joint] = 1
    for _ in range(50):
        robot.apply_action(action)
        sim.step()
        assert robot._ag_obj_in_hand[ARM] is not None, "AG should still be active when moving arm around."

    action[arm_joint] = -1
    for _ in range(50):
        robot.apply_action(action)
        sim.step()
        assert robot._ag_obj_in_hand[ARM] is not None, "AG should still be active when moving arm around."

assert np.linalg.norm(dishtowel.get_position() - robot.eef_links[ARM].get_position()) < DIST_THRESHOLD, \
    "dishtowel should be close to the gripper."

# Release grasping
action[:] = 0.0
action[gripper_idx] = 1

for _ in range(200):
    robot.apply_action(action)
    sim.step()

# Check if AG has be released and the cloth has fallen down
assert robot._ag_obj_in_hand[ARM] is None, "AG should have been released."
assert np.linalg.norm(dishtowel.get_position() - robot.eef_links[ARM].get_position()) > DIST_THRESHOLD, \
    "dishtowel should be far away from the gripper."