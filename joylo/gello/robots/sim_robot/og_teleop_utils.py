import os
import yaml
import torch as th
import numpy as np
import time

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.prims import VisualGeomPrim
from omnigibson.prims.material_prim import OmniPBRMaterialPrim
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.usd_utils import create_primitive_mesh, absolute_prim_path_to_scene_relative
from omnigibson.utils.ui_utils import dock_window
from omnigibson.utils import transform_utils as T
from omnigibson.sensors import VisionSensor
from omnigibson.objects.usd_object import USDObject
from omnigibson.robots.r1 import R1
from omnigibson.robots.r1pro import R1Pro
from bddl.activity import Conditions

from gello.robots.sim_robot.og_teleop_cfg import *


from bddl.activity import Conditions
from bddl.object_taxonomy import ObjectTaxonomy
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--activity", type=str, required=True)

ot = ObjectTaxonomy()


def get_task_relevant_room_types(activity_name):
    activity_conditions = Conditions(
        activity_name,
        0,
        simulator_name="omnigibson",
        predefined_problem=None,
    )
    init_conds = activity_conditions.parsed_initial_conditions
    room_types = set()
    for init_cond in init_conds:
        if len(init_cond) == 3:
            if "inroom" == init_cond[0]:
                room_types.add(init_cond[2])

    return list(room_types)


def augment_rooms(relevant_rooms, scene_model, task_name):
    """
    Augment the list of relevant rooms by adding dependent rooms that need to be loaded together.
    
    Args:
        relevant_rooms: List of room types that are initially relevant
        scene_model: The scene model being used
        task_name: Name of the task being used
        
    Returns:
        Augmented list of room types including all dependencies
    """

    
    # Get dependencies for current scene
    scene_dependencies = ROOM_DEPENDENCIES[scene_model]
    
    # Create a copy of the original list to avoid modifying it during iteration
    augmented_rooms = relevant_rooms.copy()
    
    # Check each relevant room for dependencies
    for room in relevant_rooms:
        if room in scene_dependencies:
            # Add dependent rooms if they're not already in the list
            for dependent_room in scene_dependencies[room]:
                if dependent_room not in augmented_rooms:
                    augmented_rooms.append(dependent_room)

    # Additionally add any task-specific rooms
    augmented_rooms += TASK_SPECIFIC_EXTRA_ROOMS.get(task_name, dict()).get(scene_model, [])
    # Remove redundancies
    augmented_rooms = list(set(augmented_rooms))
    
    return augmented_rooms


def infer_trunk_translate_from_torso_qpos(qpos):
    """
    Convert from torso joint positions to trunk translate value
    
    Args:
        qpos (torch.Tensor): Torso joint positions
        
    Returns:
        float: Trunk translate value
    """
    if qpos[0] > R1_DOWNWARD_TORSO_JOINT_POS[0]:
        # This is the interpolation between downward and ground
        translate = 1 + (qpos[0] - R1_DOWNWARD_TORSO_JOINT_POS[0]) / (
                    R1_GROUND_TORSO_JOINT_POS[0] - R1_DOWNWARD_TORSO_JOINT_POS[0])
    else:
        # This is the interpolation between upright and downward
        translate = (qpos[0] - R1_UPRIGHT_TORSO_JOINT_POS[0]) / (
                    R1_DOWNWARD_TORSO_JOINT_POS[0] - R1_UPRIGHT_TORSO_JOINT_POS[0])

    return translate.item()


def infer_torso_qpos_from_trunk_translate(translate):
    """
    Convert from trunk translate value to torso joint positions
    
    Args:
        translate (float): Trunk translate value between 0.0 and 2.0
        
    Returns:
        torch.Tensor: Torso joint positions
    """
    translate = min(max(translate, 0.0), 2.0)

    # Interpolate between the three pre-determined joint positions
    if translate <= 1.0:
        # Interpolate between upright and down positions
        interpolation_factor = translate
        interpolated_trunk_pos = (1 - interpolation_factor) * R1_UPRIGHT_TORSO_JOINT_POS + \
                                 interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
    else:
        # Interpolate between down and ground positions
        interpolation_factor = translate - 1.0
        interpolated_trunk_pos = (1 - interpolation_factor) * R1_DOWNWARD_TORSO_JOINT_POS + \
                                 interpolation_factor * R1_GROUND_TORSO_JOINT_POS

    return interpolated_trunk_pos


def print_color(*args, color=None, attrs=(), **kwargs):
    """
    Print text with color in the terminal
    
    Args:
        *args: Arguments to print
        color (str): Color name (red, green, blue, etc.)
        attrs (tuple): Additional attributes (bold, underline, etc.)
        **kwargs: Keyword arguments for print
    """
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def get_camera_config(name, relative_prim_path, position, orientation, resolution, modalities=[]):
    """
    Generate a camera configuration dictionary
    
    Args:
        name (str): Camera name
        relative_prim_path (str): Relative path to camera in the scene
        position (List[float]): Camera position [x, y, z]
        orientation (List[float]): Camera orientation [x, y, z, w]
        resolution (List[int]): Camera resolution [height, width]
        modalities (List[str]): List of modalities for the camera
        
    Returns:
        dict: Camera configuration dictionary
    """
    return {
        "sensor_type": "VisionSensor",
        "name": name,
        "relative_prim_path": relative_prim_path,
        "modalities": modalities,
        "sensor_kwargs": {
            "viewport_name": "Viewport",
            "image_height": resolution[0],
            "image_width": resolution[1],
        },
        "position": position,
        "orientation": orientation,
        "pose_frame": "parent",
        "include_in_obs": False,
    }


def create_and_dock_viewport(parent_window, position, ratio, camera_path):
    """
    Create and configure a viewport window.
    
    Args:
        parent_window: Parent window to dock this viewport to
        position: Docking position (LEFT, RIGHT, BOTTOM, etc.)
        ratio: Size ratio for the docked window
        camera_path: Path to the camera to set as active
        
    Returns:
        The created viewport window
    """
    viewport = lazy.omni.kit.viewport.utility.create_viewport_window()
    og.sim.render()
    
    dock_window(
        space=lazy.omni.ui.Workspace.get_window(parent_window),
        name=viewport.name,
        location=position,
        ratio=ratio,
    )
    og.sim.render()
    
    viewport.viewport_api.set_active_camera(camera_path)
    og.sim.render()
    
    return viewport


def setup_cameras(robot, external_sensors, resolution):
    """
    Set up all cameras for teleop visualization
    
    Args:
        robot: The robot object
        external_sensors: External camera sensors
        resolution: Camera resolution [height, width]
        
    Returns:
        tuple: (camera_paths, viewports)
    """
    viewports = {}
    
    if VIEWING_MODE == ViewingMode.MULTI_VIEW_1:
        viewport_left_shoulder = create_and_dock_viewport(
            "DockSpace", 
            lazy.omni.ui.DockPosition.LEFT,
            0.25,
            external_sensors["external_sensor1"].prim_path
        )
        viewport_left_wrist = create_and_dock_viewport(
            viewport_left_shoulder.name,
            lazy.omni.ui.DockPosition.BOTTOM,
            0.5,
            f"{robot.links[WRIST_CAMERA_LINK_NAME[robot.__class__.__name__]['left']].prim_path}/Camera"
        )
        viewport_right_shoulder = create_and_dock_viewport(
            "DockSpace",
            lazy.omni.ui.DockPosition.RIGHT,
            0.2,
            external_sensors["external_sensor2"].prim_path
        )
        viewport_right_wrist = create_and_dock_viewport(
            viewport_right_shoulder.name,
            lazy.omni.ui.DockPosition.BOTTOM,
            0.5,
            f"{robot.links[WRIST_CAMERA_LINK_NAME[robot.__class__.__name__]['right']].prim_path}/Camera"
        )
        # Set resolution for all viewports
        for viewport in [viewport_left_shoulder, viewport_left_wrist, 
                        viewport_right_shoulder, viewport_right_wrist]:
            viewport.viewport_api.set_texture_resolution((256, 256))
            og.sim.render()
            
        viewports = {
            "left_shoulder": viewport_left_shoulder,
            "left_wrist": viewport_left_wrist,
            "right_shoulder": viewport_right_shoulder,
            "right_wrist": viewport_right_wrist
        }
            
        for _ in range(3):
            og.sim.render()

    # Setup main camera view
    eyes_cam_prim_path = f"{robot.links[HEAD_CAMERA_LINK_NAME[robot.__class__.__name__]].prim_path}/Camera"
    og.sim.viewer_camera.active_camera_path = eyes_cam_prim_path
    og.sim.viewer_camera.image_height = resolution[0]
    og.sim.viewer_camera.image_width = resolution[1]

    # Adjust wrist cameras for R1
    if isinstance(robot, R1) and not isinstance(robot, R1Pro):
        left_wrist_camera_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(
            prim_path=f"{robot.links[WRIST_CAMERA_LINK_NAME[robot.__class__.__name__]['left']].prim_path}/Camera"
        )
        right_wrist_camera_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(
            prim_path=f"{robot.links[WRIST_CAMERA_LINK_NAME[robot.__class__.__name__]['right']].prim_path}/Camera"
        )
        
        left_wrist_camera_prim.GetAttribute("xformOp:translate").Set(
            lazy.pxr.Gf.Vec3d(*R1_WRIST_CAMERA_LOCAL_POS.tolist())
        )
        right_wrist_camera_prim.GetAttribute("xformOp:translate").Set(
            lazy.pxr.Gf.Vec3d(*R1_WRIST_CAMERA_LOCAL_POS.tolist())
        )
        
        left_wrist_camera_prim.GetAttribute("xformOp:orient").Set(
            lazy.pxr.Gf.Quatd(*R1_WRIST_CAMERA_LOCAL_ORI[[3, 0, 1, 2]].tolist())
        ) # expects (w, x, y, z)
        right_wrist_camera_prim.GetAttribute("xformOp:orient").Set(
            lazy.pxr.Gf.Quatd(*R1_WRIST_CAMERA_LOCAL_ORI[[3, 0, 1, 2]].tolist())
        ) # expects (w, x, y, z)
    
    # Adjust head camera for R1Pro (TODO: fix this in assets)
    if isinstance(robot, R1Pro):
        head_camera_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=f"{robot.links[HEAD_CAMERA_LINK_NAME[robot.__class__.__name__]].prim_path}/Camera")
        head_camera_prim.GetAttribute("xformOp:translate").Set(
            lazy.pxr.Gf.Vec3d(*R1PRO_HEAD_CAMERA_LOCAL_POS.tolist())
        )
        head_camera_prim.GetAttribute("xformOp:orient").Set(
            lazy.pxr.Gf.Quatd(*R1PRO_HEAD_CAMERA_LOCAL_ORI[[3, 0, 1, 2]].tolist())
        )

    camera_paths = [
        eyes_cam_prim_path,
        external_sensors["external_sensor0"].prim_path,
    ]
    
    # Lock camera attributes
    LOCK_CAMERA_ATTR = "omni:kit:cameraLock"
    for cam_path in camera_paths:
        cam_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(cam_path)
        cam_prim.GetAttribute("horizontalAperture").Set(40.0)

        # Lock attributes afterwards as well to avoid external modification
        if cam_prim.HasAttribute(LOCK_CAMERA_ATTR):
            attr = cam_prim.GetAttribute(LOCK_CAMERA_ATTR)
        else:
            attr = cam_prim.CreateAttribute(LOCK_CAMERA_ATTR, lazy.pxr.Sdf.ValueTypeNames.Bool)
        attr.Set(True)

    # Disable all render products to save on speed
    for sensor in VisionSensor.SENSORS.values():
        sensor.render_product.hydra_texture.set_updates_enabled(False)
        
    return camera_paths, viewports

def setup_camera_blinking_visualizers(camera_paths, scene):
    """
    Set up blinking visualizers for cameras
    
    Args:
        camera_paths (list): List of camera paths
        scene: Scene object
        
    Returns:
        dict: Dictionary of camera blinking visualizers
    """
    vis_elements = {}
    
    if BLINK_WHEN_IN_CONTACT:
        for cam_path in camera_paths:
            # Create material for blinking visualizer
            mat_prim_path = f"{cam_path}/Looks/blink_vis_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
                name=f"{cam_path}:blink_vis_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = th.tensor([1.0, 0.0, 0.0])  # Red color
            
            # Create visual sphere for blinking effect
            vis_prim_path = f"{cam_path}/blink_vis_sphere"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cube",
                extents=[2.0, 1.0, 0.01]
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{cam_path}:blink_vis_sphere"
            )
            vis_geom.load(scene)
            
            # Attach the material to this prim
            vis_geom.material = mat
            vis_geom.set_position_orientation(
                position=th.tensor([0, 0.65, -0.2]), 
                orientation=th.tensor([0, 0, 0, 1.0]), 
                frame="parent"
            )
            
            vis_elements[cam_path] = vis_geom
            vis_geom.visible = False  # Initially hidden
    
    return vis_elements

def update_camera_blinking_visualizers(visualizers, active_camera_path, obs, blink_frequency):
    """
    Update camera blinking visualizers based on contact status
    
    Args:
        active_visualizer: The visual geometry primitive for the active camera
        obs: Observation dictionary containing contact information
        blink_frequency (float): Frequency of blinking in Hz
    """
    if not BLINK_WHEN_IN_CONTACT:
        return
    
    # Check if robot is in contact
    in_contact = obs.get("trunk_contact", False) or obs.get("base_contact", False)
    
    if in_contact:
        # Calculate blinking based on time
        current_time = time.time()
        blink_period = 1.0 / blink_frequency  # Time for one complete blink cycle
        
        # Use sine wave for smooth blinking
        blink_phase = (current_time % blink_period) / blink_period * 2 * np.pi
        visibility = (np.sin(blink_phase) + 1) / 2  # Normalize to 0-1
        for path, vis in visualizers.items():
            if path == active_camera_path:
                # Only update the active camera's visualizer
                vis.visible = visibility > 0.5
            else:
                vis.visible = False  # Hide other visualizers
    else:
        # Not in contact - hide the visualizer
        for vis in visualizers.values():
            vis.visible = False


def setup_robot_visualizers(robot, scene):
    """
    Set up visualization elements for teleop
    
    Args:
        robot: The robot object
        scene: The scene object
        
    Returns:
        dict: Dictionary of visualization elements
    """
    vis_elements = {
        "eef_cylinder_geoms": {},
        "vis_mats": {},
        "vertical_visualizers": {},
        "reachability_visualizers": {}
    }
    
    # Create materials for visualization cylinders
    for arm in robot.arm_names:
        vis_elements["vis_mats"][arm] = []
        for axis, color in zip(("x", "y", "z"), VIS_GEOM_COLORS[False]):
            mat_prim_path = f"{robot.prim_path}/Looks/vis_cylinder_{arm}_{axis}_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
                name=f"{robot.name}:vis_cylinder_{arm}_{axis}_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = color
            vis_elements["vis_mats"][arm].append(mat)

    # Create material for visual sphere
    mat_prim_path = f"{robot.prim_path}/Looks/vis_sphere_mat"
    sphere_mat = OmniPBRMaterialPrim(
        relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
        name=f"{robot.name}:vis_sphere_mat",
    )
    sphere_color = np.array([252, 173, 76]) / 255.0
    sphere_mat.load(scene)
    sphere_mat.diffuse_color_constant = th.as_tensor(sphere_color)
    vis_elements["sphere_mat"] = sphere_mat

    # Create material for vertical cylinder
    if USE_VERTICAL_VISUALIZERS:
        mat_prim_path = f"{robot.prim_path}/Looks/vis_vertical_mat"
        vert_mat = OmniPBRMaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
            name=f"{robot.name}:vis_vertical_mat",
        )
        vert_color = np.array([252, 226, 76]) / 255.0
        vert_mat.load(scene)
        vert_mat.diffuse_color_constant = th.as_tensor(vert_color)
        vis_elements["vert_mat"] = vert_mat

    # Extract visualization cylinder settings
    vis_geom_width = VIS_CYLINDER_CONFIG["width"]
    vis_geom_lengths = VIS_CYLINDER_CONFIG["lengths"]
    vis_geom_proportion_offsets = VIS_CYLINDER_CONFIG["proportion_offsets"]
    vis_geom_quat_offsets = VIS_CYLINDER_CONFIG["quat_offsets"]

    # Create visualization cylinders for each arm
    for arm in robot.arm_names:
        hand_link = robot.eef_links[arm]
        vis_elements["eef_cylinder_geoms"][arm] = []
        for axis, length, mat, prop_offset, quat_offset in zip(
            ("x", "y", "z"),
            vis_geom_lengths,
            vis_elements["vis_mats"][arm],
            vis_geom_proportion_offsets,
            vis_geom_quat_offsets,
        ):
            vis_prim_path = f"{hand_link.prim_path}/vis_cylinder_{axis}"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_cylinder_{axis}"
            )
            vis_geom.load(scene)

            # Attach a material to this prim
            vis_geom.material = mat

            vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, length])
            vis_geom.set_position_orientation(
                position=th.tensor([0, 0, length * prop_offset]), 
                orientation=quat_offset, 
                frame="parent"
            )
            vis_elements["eef_cylinder_geoms"][arm].append(vis_geom)

        # Add vis sphere around EEF for reachability
        if USE_VISUAL_SPHERES:
            vis_prim_path = f"{hand_link.prim_path}/vis_sphere"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Sphere",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_sphere"
            )
            vis_geom.load(scene)

            # Attach a material to this prim
            sphere_mat.bind(vis_geom.prim_path)

            vis_geom.scale = th.ones(3) * 0.15
            vis_geom.set_position_orientation(
                position=th.zeros(3), 
                orientation=th.tensor([0, 0, 0, 1.0]), 
                frame="parent"
            )

        # Add vertical cylinder at EEF
        if USE_VERTICAL_VISUALIZERS:
            vis_prim_path = f"{hand_link.prim_path}/vis_vertical"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_vertical"
            )
            
            vis_geom.load(scene)

            # Attach a material to this prim
            vis_elements["vert_mat"].bind(vis_geom.prim_path)

            vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, 2.0])
            vis_geom.set_position_orientation(
                position=th.zeros(3), 
                orientation=th.tensor([0, 0, 0, 1.0]), 
                frame="parent"
            )
            vis_elements["vertical_visualizers"][arm] = vis_geom

    # Create reachability visualizers
    if USE_REACHABILITY_VISUALIZERS:
        # Create a square formation in front of the robot as reachability signal
        torso_link = robot.links["torso_link4"]
        beam_width = REACHABILITY_VISUALIZER_CONFIG["beam_width"]
        square_distance = REACHABILITY_VISUALIZER_CONFIG["square_distance"]
        square_width = REACHABILITY_VISUALIZER_CONFIG["square_width"]
        square_height = REACHABILITY_VISUALIZER_CONFIG["square_height"]
        beam_color = REACHABILITY_VISUALIZER_CONFIG["beam_color"]

        # Create material for beams
        beam_mat_prim_path = f"{robot.prim_path}/Looks/square_beam_mat"
        beam_mat = OmniPBRMaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, beam_mat_prim_path),
            name=f"{robot.name}:square_beam_mat",
        )
        beam_mat.load(scene)
        beam_mat.diffuse_color_constant = th.as_tensor(beam_color)
        vis_elements["beam_mat"] = beam_mat

        edges = [
            # name, position, scale, orientation
            ["top", [square_distance, 0, 0.3], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
            ["bottom", [square_distance, 0, 0.0], [beam_width, beam_width, square_width], [0.0, th.pi/2, th.pi/2]],
            ["left", [square_distance, 0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]],
            ["right", [square_distance, -0.2, 0.15], [beam_width, beam_width, square_height], [0.0, 0.0, 0.0]]
        ]

        for name, position, scale, orientation in edges:
            edge_prim_path = f"{torso_link.prim_path}/square_edge_{name}"
            edge_prim = create_primitive_mesh(
                edge_prim_path,
                "Cylinder",
                extents=1.0
            )
            edge_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, edge_prim_path),
                name=f"{robot.name}:square_edge_{name}"
            )
            edge_geom.load(scene)
            beam_mat.bind(edge_geom.prim_path)
            edge_geom.scale = th.tensor(scale)
            edge_geom.set_position_orientation(
                position=th.tensor(position),
                orientation=T.euler2quat(th.tensor(orientation)),
                frame="parent"
            )
            vis_elements["reachability_visualizers"][name] = edge_geom
    
    return vis_elements


def setup_flashlights(robot):
    """
    Set up flashlights on the robot's end effectors
    
    Args:
        robot: The robot object
        
    Returns:
        dict: Dictionary of flashlight objects
    """
    flashlights = {}
    
    for arm in robot.arm_names:
        light_prim = getattr(lazy.pxr.UsdLux, "SphereLight").Define(
            og.sim.stage, 
            f"{robot.links[f'{arm}_eef_link'].prim_path}/flashlight"
        )
        light_prim.GetRadiusAttr().Set(0.01)
        light_prim.GetIntensityAttr().Set(FLASHLIGHT_INTENSITY)
        light_prim.LightAPI().GetNormalizeAttr().Set(True)
        
        light_prim.ClearXformOpOrder()
        translate_op = light_prim.AddTranslateOp()
        translate_op.Set(lazy.pxr.Gf.Vec3d(-0.01, 0, -0.05))
        light_prim.SetXformOpOrder([translate_op])
        
        flashlights[arm] = light_prim
    
    return flashlights


def setup_task_instruction_ui(task_name, env, instance_id=None):
    """
    Set up UI for displaying task instructions and goal status
    
    Args:
        task_name (str): Name of the task
        env: Environment object
        instance_id (int, optional): Instance ID to display in the top right corner
        
    Returns:
        tuple: (overlay_window, text_labels, instance_id_label, bddl_goal_conditions)
    """
    if task_name is None:
        return None, None, None, None
    
    bddl_goal_conditions = env.task.activity_natural_language_goal_conditions

    # Setup overlay window
    main_viewport = og.sim.viewer_camera._viewport
    main_viewport.dock_tab_bar_visible = False
    og.sim.render()
    
    overlay_window = lazy.omni.ui.Window(
        main_viewport.name,
        width=0,
        height=0,
        flags=lazy.omni.ui.WINDOW_FLAGS_NO_TITLE_BAR |
            lazy.omni.ui.WINDOW_FLAGS_NO_SCROLLBAR |
            lazy.omni.ui.WINDOW_FLAGS_NO_RESIZE
    )
    og.sim.render()

    text_labels = []
    instance_id_label = None

    with overlay_window.frame:
        with lazy.omni.ui.ZStack():
            # Bottom layer - transparent spacer
            lazy.omni.ui.Spacer()
            
            # Create a container for the instance ID in the top right corner
            if instance_id is not None:
                with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.RIGHT_TOP, spacing=0):
                    lazy.omni.ui.Spacer(height=UI_SETTINGS["top_margin"])  # Top margin
                    with lazy.omni.ui.HStack(height=20):
                        instance_id_label = lazy.omni.ui.Label(
                            f"Instance ID: {instance_id}",
                            alignment=lazy.omni.ui.Alignment.RIGHT_CENTER,
                            style={
                                "color": 0xFFFFFFFF,  # White color (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Right margin
            
            # Text container at top left
            with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.LEFT_TOP, spacing=0):
                lazy.omni.ui.Spacer(height=UI_SETTINGS["top_margin"])  # Top margin

                # Create labels for each goal condition
                for line in bddl_goal_conditions:
                    with lazy.omni.ui.HStack(height=20):
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Left margin
                        label = lazy.omni.ui.Label(
                            line,
                            alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                            style={
                                "color": UI_SETTINGS["goal_unsatisfied_color"],  # Red color (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )
                        text_labels.append(label)
    
    # Force render to update the overlay
    og.sim.render()
    
    return overlay_window, text_labels, instance_id_label, bddl_goal_conditions


def setup_status_display_ui(main_viewport):
    """Set up UI for displaying status messages in bottom right corner"""
    # Create overlay window
    overlay_window = lazy.omni.ui.Window(
        main_viewport.name,
        width=0,
        height=0,
        flags=lazy.omni.ui.WINDOW_FLAGS_NO_TITLE_BAR |
              lazy.omni.ui.WINDOW_FLAGS_NO_SCROLLBAR |
              lazy.omni.ui.WINDOW_FLAGS_NO_RESIZE
    )
    og.sim.render()
    
    # Create UI container structure
    with overlay_window.frame:
        with lazy.omni.ui.ZStack():
            lazy.omni.ui.Spacer()
            vstack = lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.RIGHT_BOTTOM, spacing=0)
            with vstack:
                lazy.omni.ui.Spacer()
                hstack = lazy.omni.ui.HStack()
                with hstack:
                    lazy.omni.ui.Spacer()
                    # Create a single placeholder label - we'll update its text later
                    label = lazy.omni.ui.Label(
                        "",  # Empty initially
                        alignment=lazy.omni.ui.Alignment.RIGHT_CENTER,
                        visible=False,
                        style={
                            "font_size": STATUS_DISPLAY_SETTINGS["font_size"],
                            "margin": 0,
                            "padding": 0
                        }
                    )
                    lazy.omni.ui.Spacer(width=STATUS_DISPLAY_SETTINGS["right_margin"])
                lazy.omni.ui.Spacer(height=STATUS_DISPLAY_SETTINGS["bottom_margin"])
    
    og.sim.render()
    return overlay_window, label

def update_status_display(status_window, label, event_queue, current_time):
    """Update status display with the most recent active message"""
    # Filter out expired events
    filtered_queue = []
    for event_type, message, timestamp in event_queue:
        duration = STATUS_DISPLAY_SETTINGS.get("persistent_duration", 0.5) \
                  if event_type in STATUS_DISPLAY_SETTINGS["persistent_states"] \
                  else STATUS_DISPLAY_SETTINGS["event_duration"]
                  
        if current_time - timestamp < duration:
            filtered_queue.append((event_type, message, timestamp))
    
    # Get the most recent message if available
    if filtered_queue:
        event_type, message, _ = filtered_queue[-1]
        # Only update if text changed
        if not label.visible or label.text != message:
            label.text = message
            label.visible = True
            label.style = {
                "color": STATUS_DISPLAY_SETTINGS["event_colors"].get(event_type, 0xFFFFFFFF),
                "font_size": STATUS_DISPLAY_SETTINGS["font_size"],
                "margin": 0,
                "padding": 0
            }
    else:
        # Hide label when no messages
        label.visible = False
    
    return filtered_queue

def add_status_event(event_queue, event_type, message, persistent=False):
    """Add a status event to the display queue, avoiding duplicates for persistent events"""
    # For persistent events, check if it's already in the queue
    if persistent:
        for i, (e_type, _, _) in enumerate(event_queue):
            if e_type == event_type:
                # Just update timestamp to keep it alive
                event_queue[i] = (e_type, message, time.time())
                return
    
    # Add new event if not found or not persistent
    event_queue.append((event_type, message, time.time()))

def setup_object_beacons(task_relevant_objects, scene):
    """
    Set up visual beacons for task-relevant objects
    
    Args:
        task_relevant_objects (list): List of task-relevant objects
        scene: Scene object
        
    Returns:
        dict: Dictionary of object beacons
    """
    if not task_relevant_objects:
        return {}
    
    # Generate random colors for object highlighting
    random_colors = lazy.omni.replicator.core.random_colours(N=len(task_relevant_objects))[:, :3].tolist()
    object_highlight_colors = [[r/255, g/255, b/255] for r, g, b in random_colors]
    
    object_beacons = {}
    
    for obj, color in zip(task_relevant_objects, object_highlight_colors):
        obj.set_highlight_properties(color=color)
        
        # Create material for beacon
        mat_prim_path = f"{obj.prim_path}/Looks/beacon_cylinder_mat"
        mat = OmniPBRMaterialPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
            name=f"{obj.name}:beacon_cylinder_mat",
        )
        mat.load(scene)
        mat.diffuse_color_constant = th.tensor(color)

        # Create visual beacon
        vis_prim_path = f"{obj.prim_path}/beacon_cylinder"
        vis_prim = create_primitive_mesh(
            vis_prim_path,
            "Cylinder",
            extents=1.0
        )
        beacon = VisualGeomPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
            name=f"{obj.name}:beacon_cylinder"
        )
        beacon.load(scene)
        beacon.material = mat
        beacon.scale = th.tensor([0.01, 0.01, BEACON_LENGTH])
        beacon_pos = obj.aabb_center + th.tensor([0.0, 0.0, BEACON_LENGTH/2.0])
        beacon.set_position_orientation(
            position=beacon_pos, 
            orientation=T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
        )

        object_beacons[obj] = beacon
        beacon.visible = False
        
    return object_beacons

def setup_task_visualizers(task_relevant_objects, scene):
    task_visualizers = {}
    
    # Extract frame visualization settings
    vis_geom_width = ATTACHMENT_FRAME_CONFIG["width"]
    vis_geom_lengths = ATTACHMENT_FRAME_CONFIG["lengths"]
    vis_geom_quat_offsets = ATTACHMENT_FRAME_CONFIG["quat_offsets"]
    vis_geom_colors = ATTACHMENT_FRAME_CONFIG["colors"]
    
    for obj in task_relevant_objects:
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type == "attachment":
                # Create frame visualizer for the attachment link
                frame_visualizers = []
                
                # Create materials for each axis
                axis_materials = []
                for axis, color in zip(("x", "y", "z"), vis_geom_colors):
                    mat = OmniPBRMaterialPrim(
                        relative_prim_path=absolute_prim_path_to_scene_relative(scene, f"{link.prim_path}/attachment_frame_mat_{axis}"),
                        name=f"{obj.name}:attachment_frame_mat_{axis}",
                    )
                    mat.load(scene)
                    mat.diffuse_color_constant = color
                    axis_materials.append(mat)
                
                # Create cylinder for each axis (X, Y, Z)
                for axis, length, mat, quat_offset in zip(
                    ("x", "y", "z"),
                    vis_geom_lengths,
                    axis_materials,
                    vis_geom_quat_offsets,
                ):
                    vis_prim_path = f"{link.prim_path}/attachment_frame_{axis}"
                    vis_prim = create_primitive_mesh(
                        vis_prim_path,
                        "Cylinder",
                        extents=1.0
                    )
                    visualizer = VisualGeomPrim(
                        relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                        name=f"{obj.name}:attachment_frame_{axis}"
                    )
                    visualizer.load(scene)
                    
                    # Attach material
                    visualizer.material = mat
                    
                    # Scale the cylinder and normalize with link scale
                    visualizer.scale = th.tensor([vis_geom_width, vis_geom_width, length]) / link.scale
                    
                    # Set position and orientation relative to the attachment link
                    visualizer.set_position_orientation(
                        position=th.tensor([0, 0, 0]), 
                        orientation=quat_offset, 
                        frame="parent"
                    )
                    
                    visualizer.visible = False  # Initially hidden
                    frame_visualizers.append(visualizer)
                
                # Store all three axis visualizers for this attachment
                task_visualizers[obj] = frame_visualizers
    
    return task_visualizers

def setup_ghost_robot(scene, task_cfg=None):
    """
    Set up a ghost robot for visualization
    
    Args:
        scene: Scene object
        task_cfg: Dictionary of task configuration (optional)
        
    Returns:
        object: Ghost robot object
    """    
    # NOTE: Add ghost robot, but don't register it
    ghost = USDObject(
        name="ghost", 
        usd_path=os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{ROBOT_TYPE.lower()}/usd/{ROBOT_TYPE.lower()}.usda"), 
        visual_only=True, 
        position=(task_cfg is not None and task_cfg["robot_start_position"]) or [0.0, 0.0, 0.0],
        orientation=(task_cfg is not None and task_cfg["robot_start_orientation"]) or [0.0, 0.0, 0.0, 1.0],
    )
    scene.add_object(ghost, register=False)
    
    # Set ghost color
    for mat in ghost.materials:
        mat.diffuse_texture = ""
        mat.diffuse_color_constant = th.tensor([0.8, 0.0, 0.0], dtype=th.float32)
    
    # Hide all links initially
    for link in ghost.links.values():
        link.visible = False
        
    return ghost


def optimize_sim_settings(vr_mode=False):
    """Apply optimized simulation settings for better performance"""
    settings = lazy.carb.settings.get_settings()

    if not vr_mode:
        # Use asynchronous rendering for faster performance
        # NOTE: This gets reset EVERY TIME the sim stops / plays!!
        # For some reason, need to turn on, then take one render step, then turn off, and then back on in order to
        # avoid viewport freezing...not sure why
        settings.set_bool("/app/asyncRendering", True)
        og.sim.render()
        settings.set_bool("/app/asyncRendering", False)
        settings.set_bool("/app/asyncRendering", True)
        settings.set_bool("/app/asyncRenderingLowLatency", True)

        # Must ALWAYS be set after sim plays because omni overrides these values
        settings.set("/app/runLoops/main/rateLimitEnabled", False)
        settings.set("/app/runLoops/main/rateLimitUseBusyLoop", False)

        # Use asynchronous rendering for faster performance (repeat to ensure it takes effect)
        settings.set_bool("/app/asyncRendering", True)
        settings.set_bool("/app/asyncRenderingLowLatency", True)
        settings.set_bool("/app/asyncRendering", False)
        settings.set_bool("/app/asyncRenderingLowLatency", False)
        settings.set_bool("/app/asyncRendering", True)
        settings.set_bool("/app/asyncRenderingLowLatency", True)

        # Additional RTX settings
        settings.set_bool("/rtx-transient/dlssg/enabled", True)

        # Disable fractional cutout opacity for speed
        # Alternatively, turn this on so that we can use semi-translucent visualizers
        lazy.carb.settings.get_settings().set_bool("/rtx/raytracing/fractionalCutoutOpacity", False)

    # Does this improve things?
    # See https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/latest/TIME_STEPPING.html#synchronizing-wall-clock-time-and-simulation-time
    # Obtain the main timeline object
    timeline = lazy.omni.timeline.get_timeline_interface()

    # Configure Kit to not wait for wall clock time to catch up between updates
    # This setting is effective only with Fixed time stepping
    timeline.set_play_every_frame(True)

    # The following setting has the exact same effect as set_play_every_frame
    settings.set("/app/player/useFastMode", True)
    settings.set("/app/show_developer_preference_section", True)
    settings.set("/app/player/useFixedTimeStepping", True)

    for run_loop in ["present", "main", "rendering_0"]:
        settings.set(f"/app/runLoops/{run_loop}/rateLimitEnabled", False)
        settings.set(f"/app/runLoops/{run_loop}/rateLimitFrequency", 120)
        settings.set(f"/app/runLoops/{run_loop}/rateLimitUseBusyLoop", False)
    settings.set("/exts/omni.kit.renderer.core/present/enabled", True)
    settings.set("/app/vsync", True)

def setup_ghost_robot_info(ghost, robot):
    if isinstance(robot, R1Pro):
        robot_arm_dof = 7
    elif isinstance(robot, R1):
        robot_arm_dof = 6
    else:
        raise ValueError(f"Unknown robot type: {type(robot)}")
    
    # Aggregate joint indices for the ghost robot
    joint_keys_list = list(ghost.joints.keys())
    arm_action_idxs = []
    arm_joint_idxs = []
    finger_joint_idxs = []
    for arm in robot.arm_names:
        for i in range(robot_arm_dof):
            arm_joint_idxs.append(joint_keys_list.index(f"{arm}_arm_joint{i+1}"))
            arm_action_idxs.append(robot.arm_action_idx[arm][i])
        for i in range(2):
            finger_name = f"{arm}_gripper_finger_joint{i+1}"
            if not ghost.joints[finger_name].is_mimic_joint:
                finger_joint_idxs.append(joint_keys_list.index(finger_name))
    
    torso_joint_idxs = []
    for i in range(4):
        torso_joint_idxs.append(joint_keys_list.index(f"torso_joint{i+1}"))
                    
    ghost_info = {
        "arm_action_idxs": th.tensor(arm_action_idxs, dtype=th.int),
        "arm_joint_idxs": th.tensor(arm_joint_idxs, dtype=th.int),
        "finger_joint_idxs": th.tensor(finger_joint_idxs, dtype=th.int),
        "torso_joint_idxs": th.tensor(torso_joint_idxs, dtype=th.int),
        "lower_limit": ghost.joint_lower_limits,
        "upper_limit": ghost.joint_upper_limits,
        "left_links": [link for link_name, link in ghost.links.items() if link_name.startswith("left")],
        "right_links": [link for link_name, link in ghost.links.items() if link_name.startswith("right")],
    }
    
    return ghost_info


def update_ghost_robot(ghost, robot, action, ghost_appear_counter, ghost_info):
    """
    Update the ghost robot visualization based on current robot state and action
    
    Args:
        ghost: Ghost robot object
        robot: Robot object
        action: Current action being applied
        ghost_appear_counter: Counter for ghost appearance timing
        ghost_info: Dictionary of cached ghost information
        
    Returns:
        dict: Updated ghost_appear_counter
    """
    ghost.set_position_orientation(
        position=robot.get_position_orientation(frame="world")[0],
        orientation=robot.get_position_orientation(frame="world")[1],
    )
    
    robot_qpos = robot.get_joint_positions(normalized=False)
    ghost_qpos = robot_qpos.clone()
    ghost_qpos[ghost_info["arm_joint_idxs"]] = action[ghost_info["arm_action_idxs"]]
    ghost_qpos = ghost_qpos.clip(ghost_info["lower_limit"], ghost_info["upper_limit"])
    
    update = False
    for arm in robot.arm_names:
        # make arm visible if some joint difference is larger than the threshold
        if th.max(th.abs(
            robot_qpos[robot.arm_control_idx[arm]] - action[robot.arm_action_idx[arm]]
        )) > GHOST_APPEAR_THRESHOLD:
            ghost_appear_counter[arm] += 1
            update = True
            if ghost_appear_counter[arm] >= GHOST_APPEAR_TIME:
                for link in ghost_info[f"{arm}_links"]:
                    link.visible = True
        else:
            ghost_appear_counter[arm] = 0
            for link in ghost_info[f"{arm}_links"]:
                link.visible = False
    
    if update:
        # Concatenate arm and torso indices and set joint positions
        update_indices = th.cat([ghost_info["torso_joint_idxs"], ghost_info["arm_joint_idxs"], ghost_info["finger_joint_idxs"]])
        ghost.set_joint_positions(ghost_qpos[update_indices], indices=update_indices, normalized=False, drive=False)
    
    return ghost_appear_counter

def update_instance_id_label(instance_id_label, instance_id):
    """
    Update the instance ID label in the UI
    
    Args:
        instance_id_label: Label object for displaying instance ID
        instance_id: New instance ID to display
        
    Returns:
        None
    """
    if instance_id_label is not None and instance_id is not None:
        instance_id_label.text = f"Instance ID: {instance_id}"


def update_goal_status(text_labels, goal_status, prev_goal_status, env, recording_path=None, event_queue=None):
    """
    Update the UI based on goal status changes
    
    Args:
        text_labels: List of UI text labels
        goal_status: Current goal status
        prev_goal_status: Previous goal status
        env: Environment object
        recording_path: Path to save recordings (optional)
        
    Returns:
        dict: Updated previous goal status
    """
    if text_labels is None:
        return prev_goal_status

    # Check if status has changed
    status_changed = (set(goal_status['satisfied']) != set(prev_goal_status['satisfied']) or
                    set(goal_status['unsatisfied']) != set(prev_goal_status['unsatisfied']))

    if status_changed:
        # Update satisfied goals - make them green
        for idx in goal_status['satisfied']:
            if 0 <= idx < len(text_labels):
                current_style = text_labels[idx].style
                current_style.update({"color": UI_SETTINGS["goal_satisfied_color"]})  # Green (ABGR)
                text_labels[idx].set_style(current_style)

        # Update unsatisfied goals - make them red
        for idx in goal_status['unsatisfied']:
            if 0 <= idx < len(text_labels):
                current_style = text_labels[idx].style
                current_style.update({"color": UI_SETTINGS["goal_unsatisfied_color"]})  # Red (ABGR)
                text_labels[idx].set_style(current_style)

        # Return the updated status
        return goal_status.copy()
    
    return prev_goal_status


def update_in_hand_status(robot, vis_mats, prev_in_hand_status):
    """
    Update the visualization based on whether objects are in hand
    
    Args:
        robot: Robot object
        vis_mats: Visualization materials dictionary
        prev_in_hand_status: Previous in-hand status
        
    Returns:
        dict: Updated in-hand status
    """
    updated_status = prev_in_hand_status.copy()
    
    # Update the in-hand status of the robot's arms
    for arm in robot.arm_names:
        in_hand = len(robot._find_gripper_raycast_collisions(arm)) != 0
        if in_hand != prev_in_hand_status[arm]:
            updated_status[arm] = in_hand
            for idx, mat in enumerate(vis_mats[arm]):
                mat.diffuse_color_constant = VIS_GEOM_COLORS[in_hand][idx]
    
    return updated_status


def update_grasp_status(robot, eef_cylinder_geoms, prev_grasp_status):
    """
    Update the visualization based on whether robot is grasping
    
    Args:
        robot: Robot object
        eef_cylinder_geoms: End effector cylinder geometries
        prev_grasp_status: Previous grasp status
        
    Returns:
        dict: Updated grasp status
    """
    updated_status = prev_grasp_status.copy()
    
    for arm in robot.arm_names:
        is_grasping = robot.is_grasping(arm) > 0
        if is_grasping != prev_grasp_status[arm]:
            updated_status[arm] = is_grasping
            for cylinder in eef_cylinder_geoms[arm]:
                cylinder.visible = not is_grasping
    
    return updated_status


def update_reachability_visualizers(reachability_visualizers, joint_cmd, prev_base_motion):
    """
    Update the reachability visualizers based on base motion
    
    Args:
        reachability_visualizers: Reachability visualizer objects
        joint_cmd: Joint command dictionary
        prev_base_motion: Previous base motion state
        
    Returns:
        bool: Updated base motion state
    """
    if not USE_REACHABILITY_VISUALIZERS or not reachability_visualizers:
        return prev_base_motion

    # Show visualizers only when there's nonzero base motion
    has_base_motion = th.any(th.abs(joint_cmd["base"]) > 1e-3)
    
    if has_base_motion != prev_base_motion:
        for edge in reachability_visualizers.values():
            edge.visible = has_base_motion
    
    return has_base_motion


def update_checkpoint(env, frame_counter, recording_path=None, event_queue=None):
    """
    Update checkpoint based on periodic timer
    
    Args:
        env: Environment object
        frame_counter: Current frame counter
        recording_path: Path to save recordings (optional)
        
    Returns:
        int: Updated frame counter
    """
    if not AUTO_CHECKPOINTING:
        return frame_counter
    
    updated_counter = frame_counter + 1
    
    if updated_counter % STEPS_TO_AUTO_CHECKPOINT == 0:
        if recording_path is not None:
            env.update_checkpoint()
            print("Auto recorded checkpoint due to periodic save!")
            if event_queue:
                add_status_event(event_queue, "checkpoint", "Checkpoint Recorded due to periodic save")
        updated_counter = 0
    
    return updated_counter


def load_available_tasks():
    """
    Load available tasks from configuration file
    
    Returns:
        dict: Dictionary of available tasks
    """
    # Get directory of current file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    task_cfg_path = os.path.join(dir_path, '..', '..', '..', 'sampled_task', 'available_tasks.yaml')
    
    try:
        with open(task_cfg_path, 'r') as file:
            available_tasks = yaml.safe_load(file)
        return available_tasks
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading available tasks: {e}")
        return {}


def generate_basic_environment_config(task_name=None, task_cfg=None):
    """
    Generate a basic environment configuration
    
    Args:
        task_name (str): Name of the task (optional)
        task_cfg: Dictionary of task config (optional)
        
    Returns:
        dict: Environment configuration
    """
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [
                get_camera_config(
                    name="external_sensor0", 
                    relative_prim_path=f"/controllable__{ROBOT_TYPE.lower()}__{ROBOT_NAME}/base_link/external_sensor0", 
                    position=EXTERNAL_CAMERA_CONFIGS["external_sensor0"]["position"], 
                    orientation=EXTERNAL_CAMERA_CONFIGS["external_sensor0"]["orientation"], 
                    resolution=RESOLUTION
                ),
            ],
        },
    }
    
    if VIEWING_MODE == ViewingMode.MULTI_VIEW_1:
        cfg["env"]["external_sensors"].append(
            get_camera_config(
                name="external_sensor1", 
                relative_prim_path=f"/controllable__{ROBOT_TYPE.lower()}__{ROBOT_NAME}/base_link/external_sensor1", 
                position=EXTERNAL_CAMERA_CONFIGS["external_sensor1"]["position"], 
                orientation=EXTERNAL_CAMERA_CONFIGS["external_sensor1"]["orientation"], 
                resolution=RESOLUTION
            )
        )
        cfg["env"]["external_sensors"].append(
            get_camera_config(
                name="external_sensor2", 
                relative_prim_path=f"/controllable__{ROBOT_TYPE.lower()}__{ROBOT_NAME}/base_link/external_sensor2", 
                position=EXTERNAL_CAMERA_CONFIGS["external_sensor2"]["position"], 
                orientation=EXTERNAL_CAMERA_CONFIGS["external_sensor2"]["orientation"], 
                resolution=RESOLUTION
            )
        )

    if task_name is not None and task_cfg is not None:
        # Load the environment for a particular task
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": task_cfg["scene_model"],
            "load_room_types": None,
            "load_room_instances": task_cfg.get("load_room_instances", None),
            "include_robots": False,
        }

        cfg["task"] = {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {
                "max_steps": 50000,
            },
            "reward_config": {
                "r_potential": 1.0,
            },
            "include_obs": False,
        }
    elif FULL_SCENE:
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    else:
        # Simple scene with a table
        x_offset = 0.5
        cfg["scene"] = {"type": "Scene"}
        cfg["objects"] = [
            {
                "type": "PrimitiveObject",
                "name": "table",
                "primitive_type": "Cube",
                "fixed_base": True,
                "scale": [0.5, 0.5, 0.3],
                "position": [0.75 + x_offset, 0, 0.65],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "name": "table2",
                "primitive_type": "Cube",
                "fixed_base": True,
                "scale": [0.5, 0.5, 0.3],
                "position": [0.0, 0.95, 0.65],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "rgba": [0.0, 1.0, 1.0, 1.0],
            },
            {
                "type": "PrimitiveObject",
                "name": "table3",
                "primitive_type": "Cube",
                "fixed_base": True,
                "scale": [0.5, 0.5, 0.3],
                "position": [-1.0, 0.0, 0.25],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "rgba": [1.0, 1.0, 0.0, 1.0],
            }
        ]

        if USE_CLOTH:
            obj_cfgs = [{
                "type": "DatasetObject",
                "name": "obj",
                "category": "dishtowel",
                "model": "dtfspn",
                "prim_type": "CLOTH",
                "scale": [2.0, 2.0, 2.0],
                "position": [0.65 + x_offset, 0, 0.95],
                "orientation": [0.0, 0.0, 0, 1.0],
                "abilities": {"cloth": {}},
            }]
        elif USE_FLUID:
            obj_cfgs = [
                {
                    "type": "DatasetObject",
                    "name": "obj",
                    "category": "coffee_cup",
                    "model": "ykuftq",
                    "scale": [1.5] * 3,
                    "position": [0.65 + x_offset, -0.15, 0.85],
                    "orientation": [0.0, 0.0, 0, 1.0],
                },
                {
                    "type": "DatasetObject",
                    "name": "obj1",
                    "category": "coffee_cup",
                    "model": "xjdyon",
                    "scale": [1.1] * 3,
                    "position": [0.65 + x_offset, 0.15, 0.84],
                    "orientation": [0.0, 0.0, 0, 1.0],
                },
            ]
        elif USE_ARTICULATED:
            obj_cfgs = [{
                "type": "DatasetObject",
                "name": "obj",
                "category": "freezer",
                "model": "aayduy",
                "scale": [0.9, 0.9, 0.9],
                "position": [0.65 + x_offset, 0, 0.95],
                "orientation": [0.0, 0.0, 0, 1.0],
            },
            {
                "type": "DatasetObject",
                "name": "obj2",
                "category": "fridge",
                "model": "dxwbae",
                "scale": [0.9, 0.9, 0.9],
                "position": [5.0, 0, 1.0],
                "orientation": [0.0, 0.0, 0, 1.0],
            },
            {
                "type": "DatasetObject",
                "name": "obj3",
                "category": "wardrobe",
                "model": "bhyopq",
                "scale": [0.9, 0.9, 0.9],
                "position": [10.0, 0, 1.0],
                "orientation": [0.0, 0.0, 0, 1.0],
            },
            ]
        else:
            obj_cfgs = [{
                "type": "DatasetObject",
                "name": "obj",
                "category": "crock_pot",
                "model": "xdahvv",
                "scale": [0.9, 0.9, 0.9],
                "position": [0.65 + x_offset, 0, 0.95],
                "orientation": [0.0, 0.0, 0, 1.0],
            }]
        cfg["objects"] += obj_cfgs
    
    return cfg


def generate_robot_config(task_name=None, task_cfg=None):
    """
    Generate robot configuration
    
    Args:
        task_name: Name of the task (optional)
        task_cfg: Dictionary of task config (optional)
        
    Returns:
        dict: Robot configuration
    """
    # Create a copy of the controller config to avoid modifying the original
    controller_config = {k: v.copy() for k, v in R1_CONTROLLER_CONFIG.items()}
    
    robot_config = {
        "type": ROBOT_TYPE,
        "name": ROBOT_NAME,
        "action_normalize": False,
        "controller_config": controller_config,
        "self_collisions": True,
        "obs_modalities": [],
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "grasping_mode": "assisted",
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                },
            },
        },
    }
    
    # Override position and orientation for tasks
    if task_name is not None and task_cfg is not None:
        robot_config["position"] = task_cfg["robot_start_position"]
        robot_config["orientation"] = task_cfg["robot_start_orientation"]
    
    # Add reset joint positions
    joint_pos = ROBOT_RESET_JOINT_POS[ROBOT_TYPE].clone()
    
    # NOTE: Fingers MUST start open, or else generated AG spheres will be spawned incorrectly
    joint_pos[-4:] = 0.05
    
    # Update trunk qpos as well
    joint_pos[6:10] = infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE)
    
    robot_config["reset_joint_pos"] = joint_pos
    
    return robot_config


def apply_omnigibson_macros():
    """Apply global OmniGibson settings"""
    for key, value in OMNIGIBSON_MACROS.items():
        setattr(gm, key, value)

class SignalChangeDetector:
    def __init__(self, debounce_time=0.5):
        """
        A very simple signal change detector for stable signals that only change infrequently.
        
        Args:
            debounce_time: Minimum time (seconds) between detected changes
        """
        self.current_state = None
        self.last_change_time = 0
        self.debounce_time = debounce_time
    
    def reset(self):
        """Reset the detector to initial state"""
        self.current_state = None
        self.last_change_time = 0
        
    def process_sample(self, sample):
        """
        Process a new sample and detect if a change occurred.
        
        Args:
            sample: The signal value (1 or -1)
            
        Returns:
            bool: True if a change was detected, False otherwise
        """
        # Catch first valid sample
        if self.current_state is None:
            self.current_state = sample
            return False
        
        # Check if enough time has passed since last change
        current_time = time.time()
        if current_time - self.last_change_time < self.debounce_time:
            return False
        
        # Check for state change
        if sample != self.current_state:
            self.current_state = sample
            self.last_change_time = current_time
            return True
            
        return False