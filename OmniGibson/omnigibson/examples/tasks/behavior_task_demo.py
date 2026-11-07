import json
import logging
import os
import re
import time
import cv2
import numpy as np
import torch as th
from gello.robots.sim_robot.og_teleop_utils import generate_robot_config

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.sensors.minimap_sensor import MinimapSensor
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import KeyboardEventHandler, KeyboardRobotController, choose_from_options


# TODO: Remove hardcoded robot model name
ROBOT_NAME = "R1Pro"
# Module-level cache for task index
_TASK_INDEX = None

gm.ENABLE_FLATCACHE = True
og.log.setLevel(logging.INFO)


def build_behavior_task_index():
    """
    Build an index mapping (task_name, def_id, inst_id) -> scene info.

    Scans the behavior-instances dataset directory and parses filenames to extract
    scene_model, task_name, activity_definition_id, and activity_instance_id.

    Returns:
        dict: Mapping from (task_name, activity_definition_id, activity_instance_id) to
              {"scene_model": str, "template_path": str}
    """
    global _TASK_INDEX
    if _TASK_INDEX is not None:
        return _TASK_INDEX

    _TASK_INDEX = {}
    instances_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "scenes")

    if not os.path.isdir(instances_path):
        og.log.warning(f"behavior-instances path not found: {instances_path}")
        return _TASK_INDEX

    # Pattern: {scene_model}_task_{task_name}_{def_id}_{inst_id}_template.json
    pattern = re.compile(r"^(.+)_task_(.+)_(\d+)_(\d+)_template\.json$")

    for scene_model in os.listdir(instances_path):
        json_dir = os.path.join(instances_path, scene_model, "json")
        if not os.path.isdir(json_dir):
            continue

        for filename in os.listdir(json_dir):
            match = pattern.match(filename)
            if match:
                parsed_scene, task_name, def_id, inst_id = match.groups()
                # Verify scene_model matches directory
                if parsed_scene == scene_model:
                    key = (task_name, int(def_id), int(inst_id))
                    _TASK_INDEX[key] = {
                        "scene_model": scene_model,
                        "template_path": os.path.join(json_dir, filename),
                    }

    og.log.info(f"Built behavior task index with {len(_TASK_INDEX)} entries")
    return _TASK_INDEX


def load_task_config_from_instances(task_name, activity_definition_id, activity_instance_id):
    """
    Load task config from behavior-instances dataset.

    Args:
        task_name: Name of the task/activity
        activity_definition_id: Activity definition ID (integer)
        activity_instance_id: Activity instance ID (integer)

    Returns:
        tuple: (task_cfg, template_path) where task_cfg is compatible with existing functions

    Raises:
        FileNotFoundError: If no template found for the given task parameters
    """
    index = build_behavior_task_index()
    key = (task_name, activity_definition_id, activity_instance_id)

    if key not in index:
        raise FileNotFoundError(
            f"No template found for task '{task_name}' with "
            f"activity_definition_id={activity_definition_id}, "
            f"activity_instance_id={activity_instance_id}"
        )

    info = index[key]
    template_path = info["template_path"]
    scene_model = info["scene_model"]

    # Read robot pose from template metadata
    with open(template_path, "r") as f:
        data = json.load(f)

    robot_poses = data["metadata"]["task"]["robot_poses"]

    task_cfg = {
        "scene_model": scene_model,
        "robot_start_position": robot_poses[ROBOT_NAME][0]["position"],
        "robot_start_orientation": robot_poses[ROBOT_NAME][0]["orientation"],
        "load_room_instances": None,
    }

    return task_cfg, template_path


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )

        # Add to user responses
        controller_choices[controller_name] = choice

    return controller_choices


def main(
    task_name: str,
    activity_definition_id: int = 0,
    activity_instance_id: int = 0,
    quickstart: bool = False,
    record_video: bool = False,
    video_path: str = None,
    enable_camera_teleop: bool = False,
    highlight_tro: bool = False,
    enable_minimap: bool = False,
):
    """
    Teleoperate a robot in a Behavior scene.

    Args:
        task_name: Name of the task/activity
        activity_definition_id: Activity definition ID (default 0)
        activity_instance_id: Activity instance ID (default 0)
        quickstart: Use default controller settings
        record_video: Enable video recording
        video_path: Path to save video
        enable_camera_teleop: Enable camera teleoperation
        highlight_tro: Highlight task-relevant objects
        enable_minimap: Enable minimap display
    """

    # Load task config from 2025-challenge-task-instances dataset
    task_cfg, template_path = load_task_config_from_instances(
        task_name, activity_definition_id, 0
    )  # Shared template _0 is used for all activity instances
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["robots"] = [
        generate_robot_config(
            task_name=task_name,
            task_cfg=task_cfg,
        )
    ]
    # Enable sensors by setting observation modalities (required for camera-based tasks)
    cfg["robots"][0]["obs_modalities"] = ["rgb"]

    # Set task activity_definition_id and activity_instance_id in config
    cfg["task"]["activity_definition_id"] = activity_definition_id
    cfg["task"]["activity_instance_id"] = activity_instance_id

    # Load the template file directly as scene_file
    og.log.info(f"Loading scene from template: {template_path}")
    cfg["scene"]["scene_file"] = template_path

    env = og.Environment(configs=cfg)

    # load robot
    robot = env.robots[0]

    # load task instance
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=activity_instance_id,
    )
    tro_file_path = os.path.join(
        get_task_instance_path(scene_model),
        f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
    )
    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    og.log.info(f"Loaded tro_state for task instance {activity_instance_id}")

    for tro_key, tro_value in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = tro_value
            robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
            robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            # Write robot poses to scene metadata
            env.scene.write_task_metadata(key=tro_key, data=tro_value)
        else:
            env.task.object_scope[tro_key].load_state(tro_value, serialized=False)

    # Try to ensure that all task-relevant objects are stable
    # They should already be stable from the sampled instance, but there is some issue where loading the state
    # causes some jitter (maybe for small mass / thin objects?)
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    # Highlight task-relevant objects if requested
    if highlight_tro:
        tro_color = [0.1, 1.0, 0.92]
        highlighted_tros = []
        for tro_name, tro_entity in env.task.object_scope.items():
            # Skip systems and non-existent entities
            if hasattr(tro_entity, "is_system") and tro_entity.is_system:
                continue
            if hasattr(tro_entity, "exists") and not tro_entity.exists:
                continue
            # Skip the robot (agent) and floor objects
            if "agent" in tro_name.lower() or "floor" in tro_name.lower():
                continue
            # Set custom highlight color and enable highlighting
            if hasattr(tro_entity, "set_highlight_properties"):
                tro_entity.set_highlight_properties(color=tro_color)
                tro_entity.highlighted = True
                highlighted_tros.append(tro_name)
        if highlighted_tros:
            og.log.info(f"Highlighted {len(highlighted_tros)} task-relevant objects (cyan): {highlighted_tros}")
        else:
            og.log.info("No task-relevant objects found to highlight")

    # Teleoperate robot with keyboard
    control_mode = "teleop"

    controller_choices = {
        "base": "HolonomicBaseJointController",
        "trunk": "JointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot)

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()
    # Reset environment
    env.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # # Increase movement speed by scaling all keypress values
    # speed_multiplier = 5.0  # Adjust this value to control how much faster (2.0 = 2x faster)
    # for key, mapping in action_generator.keypress_mapping.items():
    #     if mapping["val"] is not None:
    #         mapping["val"] *= speed_multiplier
    # og.log.info(f"Keyboard teleop speed multiplier: {speed_multiplier}x")

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info
    action_generator.print_keyboard_teleop_info()

    # Move camera to a good position (currently set for picking_up_trash_0_0)
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )
    if enable_camera_teleop:
        og.sim.enable_viewer_camera_teleoperation()

    # Create minimap sensor if enabled
    minimap_sensor = None
    if enable_minimap:
        minimap_sensor = MinimapSensor(
            scene=env.scene,
            robot=robot,
            name="minimap",
            resolution=224,
        )
        minimap_sensor.load(env.scene)
        minimap_sensor.initialize()
        og.log.info("Minimap sensor created and initialized")

    # Record waypoints
    waypoints = []

    def add_waypoint():
        nonlocal waypoints
        pos = robot.get_position_orientation()[0]
        og.log.info(f"Added waypoint at {pos}")
        waypoints.append(pos)

    def clear_waypoints():
        nonlocal waypoints
        og.log.info("Cleared all waypoints!")
        waypoints = []

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.W,
        callback_fn=add_waypoint,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.X,
        callback_fn=clear_waypoints,
    )

    og.log.info("\t W: Save the current robot pose as a waypoint")
    og.log.info("\t X: Clear all waypoints")

    # Record video
    unique_id = str(int(time.time()))

    def get_save_dir():
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos")
        os.makedirs(
            os.path.join(base_dir, task_name, str(activity_definition_id), str(activity_instance_id), unique_id),
            exist_ok=True,
        )
        return os.path.join(base_dir, task_name, str(activity_definition_id), str(activity_instance_id), unique_id)

    video_writer = None
    video_saved = False
    if record_video:
        if video_path is None:
            save_dir = get_save_dir()
            video_path = os.path.join(
                save_dir, f"video_{task_name}_{activity_definition_id}_{activity_instance_id}.mp4"
            )
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
        video_writer = create_video_writer(
            fpath=video_path,
            resolution=(672, 1568),  # height, width - includes viewer camera (448x448) on right
        )
        og.log.info(f"Recording video to: {video_path}")

    def save_video():
        nonlocal video_writer, video_saved
        if video_writer is None:
            og.log.info("No video to save (recording not enabled)")
            return
        if video_saved:
            og.log.info("Video already saved!")
            return
        try:
            container, stream = video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            video_saved = True
            video_writer = None
            # save waypoints (convert tensors to lists for JSON serialization)
            waypoints_serializable = [wp.tolist() if hasattr(wp, "tolist") else wp for wp in waypoints]
            with open(
                os.path.join(
                    os.path.dirname(video_path),
                    f"waypoints_{task_name}_{activity_definition_id}_{activity_instance_id}.json",
                ),
                "w",
            ) as f:
                json.dump(waypoints_serializable, f)

            og.log.info(f"Video saved to: {video_path}")
            og.log.info(
                f"Waypoints saved to: {os.path.join(os.path.dirname(video_path), f'waypoints_{task_name}_{activity_definition_id}_{activity_instance_id}.json')}"
            )

        except Exception as e:
            og.log.info(f"Error saving video: {e}")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Z,
        callback_fn=save_video,
    )
    og.log.info("\t Z: Save video recording and waypoints")

    # Front-view camera capture functionality
    # Use "head" key from ROBOT_CAMERA_NAMES to avoid hardcoding camera name
    front_view_camera_key = "head"
    front_view_camera_name = ROBOT_CAMERA_NAMES[robot.model_name][front_view_camera_key]
    front_view_save_counter = [0]  # Use list to allow modification in closure

    def save_front_view_camera():
        """Save front-view camera image and pose when Y is pressed."""
        # Get the camera sensor
        camera_sensor_name = front_view_camera_name.split("::")[1]
        camera = robot.sensors[camera_sensor_name]

        # Get camera pose using the same logic as preprocess_obs
        # camera.get_position_orientation() returns the most recent camera poses, but since og render is "async",
        # it may not be in sync with the visual observations. camera.camera_parameters["cameraViewTransform"]
        # is guaranteed to be in sync with the visual observations.
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            # First time query returns all zeros, fallback to direct method
            cam_pos, cam_quat = camera.get_position_orientation()
        else:
            # Extract pose from the synchronized cameraViewTransform matrix
            cam_pos, cam_quat = T.mat2pose(
                th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32)
            )
        og.log.info(f"Front-view camera ({front_view_camera_name}) pose:")
        og.log.info(f"  Position: {cam_pos.tolist()}")
        og.log.info(f"  Orientation: {cam_quat.tolist()}")

        # Determine save directory (same as video_path directory)
        if video_path is not None:
            save_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
        else:
            save_dir = get_save_dir()
        os.makedirs(save_dir, exist_ok=True)

        # Get current observation for the camera image
        current_obs, _ = env.get_obs()
        current_obs = flatten_obs_dict(current_obs)

        # Get the front-view camera RGB image
        rgb_key = front_view_camera_name + "::rgb"
        if rgb_key in current_obs:
            front_view_rgb = current_obs[rgb_key][:, :, :3].numpy()

            # Save image (convert RGB to BGR for cv2)
            img_filename = f"front_view_camera_{front_view_save_counter[0]}.png"
            img_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(front_view_rgb, cv2.COLOR_RGB2BGR))

            # Save pose as JSON
            pose_filename = f"front_view_camera_pose_{front_view_save_counter[0]}.json"
            pose_path = os.path.join(save_dir, pose_filename)
            pose_data = {
                "position": cam_pos.tolist(),
                "orientation": cam_quat.tolist(),
                "camera_name": front_view_camera_name,
            }
            with open(pose_path, "w") as f:
                json.dump(pose_data, f, indent=2)

            og.log.info(f"Saved front-view camera image to: {img_path}")
            og.log.info(f"Saved front-view camera pose to: {pose_path}")

            front_view_save_counter[0] += 1
        else:
            og.log.warning(f"Front-view camera RGB key '{rgb_key}' not found in observations")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Y,
        callback_fn=save_front_view_camera,
    )
    og.log.info("\t Y: Save front-view camera image and pose")

    def write_video_frame(obs):
        """Write current observation frame to video."""
        if video_writer is None:
            return
        # Get RGB observations from robot cameras
        left_wrist_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES[robot.model_name]["left_wrist"] + "::rgb"][:, :, :3].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES[robot.model_name]["right_wrist"] + "::rgb"][:, :, :3].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES[robot.model_name]["head"] + "::rgb"][:, :, :3].numpy(),
            (672, 672),  # Width x Height - match left panel height
        )
        # Get viewer camera RGB
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        viewer_rgb = cv2.resize(
            viewer_obs["rgb"][:, :, :3].numpy(),  # Remove alpha channel if present
            (672, 672),  # Width x Height - match left panel height
        )
        # Get minimap observation (or black placeholder if disabled)
        if minimap_sensor is not None:
            minimap_obs, _ = minimap_sensor.get_obs()
            minimap_rgb = cv2.resize(
                minimap_obs["rgb"].numpy().astype(np.uint8),
                (224, 224),
            )
        else:
            minimap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        # Stack cameras: [left_wrist, right_wrist] on left, head in middle, viewer on right
        frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb, minimap_rgb]), head_rgb, viewer_rgb])
        write_video(
            np.expand_dims(frame, 0),
            video_writer=video_writer,
            batch_size=1,
            mode="rgb",
        )

    # Other helpful user info
    og.log.info("Running demo.")
    og.log.info("Press ESC to quit")
    if record_video:
        og.log.info("Recording video... Press ESC when done to save.")

    # Loop control until user quits
    max_steps = -1
    step = 0

    # Capture initial frame before any actions
    if record_video:
        initial_obs, _ = env.get_obs()
        initial_obs = flatten_obs_dict(initial_obs)
        write_video_frame(initial_obs)

    try:
        while step != max_steps:
            action = action_generator.get_teleop_action()
            obs, _, _, _, _ = env.step(action=action)

            # Update minimap display
            if minimap_sensor is not None:
                minimap_sensor.update_display()

            # Write video frame if recording
            if record_video:
                obs = flatten_obs_dict(obs)
                write_video_frame(obs)

            step += 1
    except Exception as e:
        og.log.info(f"Loop exited: {e}")
    finally:
        # Close video writer if not already saved via 'Z' key
        # TODO: og shuts down before the video writer is closed, so we need to save the video manually.
        save_video()

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True, help="Name of the task/activity")
    parser.add_argument("--activity_definition_id", type=int, default=0, help="Activity definition ID (default: 0)")
    parser.add_argument("--activity_instance_id", type=int, default=0, help="Activity instance ID (default: 0)")
    parser.add_argument("--quickstart", action="store_true", help="Use default controller settings")
    parser.add_argument("--record_video", action="store_true", help="Enable video recording")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save video")
    parser.add_argument("--enable_camera_teleop", action="store_true", help="Enable camera teleoperation")
    parser.add_argument(
        "--highlight_tro", action="store_true", help="Highlight task-relevant objects (uses cyan color)"
    )
    parser.add_argument("--enable_minimap", action="store_true", help="Enable minimap display")

    args = parser.parse_args()
    main(
        args.task_name,
        args.activity_definition_id,
        args.activity_instance_id,
        args.quickstart,
        args.record_video,
        args.video_path,
        args.enable_camera_teleop,
        args.highlight_tro,
        args.enable_minimap,
    )
