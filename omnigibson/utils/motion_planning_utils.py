import logging
import random

from transforms3d import euler

from omnigibson.robots.manipulation_robot import IsGraspingState
from omnigibson.utils.sim_utils import check_collision
from omni.isaac.core.utils.prims import get_prim_at_path

log = logging.getLogger(__name__)

import time

import numpy as np
from collections import OrderedDict

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson import app, assets_path
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.sensors.scan_sensor import ScanSensor
from omnigibson.scenes.static_traversable_scene import StaticTraversableScene
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
import omnigibson.utils.transform_utils as T
import omnigibson.macros as m
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.transform_utils import l2_distance, euler2quat
from omnigibson.utils.sim_utils import land_object, test_valid_pose

import lula

SEARCHED = []
# Setting this higher unfortunately causes things to become impossible to pick up (they touch their hosts)
BODY_MAX_DISTANCE = 0.05
HAND_MAX_DISTANCE = 0

import pybullet as p
from omnigibson.external.motion.motion_planners.rrt_connect import birrt
from omnigibson.external.pybullet_tools.utils import (
    PI,
    circular_difference,
    direct_path,
    get_aabb,
    get_base_values,
    get_joint_names,
    get_joint_positions,
    get_joints,
    get_max_limits,
    get_min_limits,
    get_movable_joints,
    get_sample_fn,
    is_collision_free,
    joints_from_names,
    link_from_name,
    movable_from_joints,
    pairwise_collision,
    plan_base_motion_2d,
    plan_joint_motion,
    set_base_values_with_z,
    set_joint_positions,
    set_pose,
    get_link_position_from_name,
    get_link_name,
    base_values_from_pose,
    pairwise_link_collision,
    control_joints,
    get_base_values,
    get_joint_positions,
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    is_collision_free,
    joints_from_names,
    link_from_name,
    plan_base_motion_2d,
    plan_joint_motion,
    set_base_values_with_z,
    set_joint_positions,
    set_pose,
    get_self_link_pairs,
    get_body_name,
    get_link_name,
)


class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    """

    def __init__(
        self,
        robot_description_path,
        robot_urdf_path,
        eef_name,
        default_joint_pos,
        headless=False,
    ):
        # Create robot description, kinematics, and config
        if headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        self.robot_body_id = p.loadURDF(robot_urdf_path)
        p.resetBasePositionAndOrientation(self.robot_body_id, [0, 0, 0], [0, 0, 0, 1])
        p.setGravity(0, 0, 0)
        p.stepSimulation()

    def solve(
        self,
        target_pos,
        target_quat,
        initial_joint_pos=None,
    ):
        print("Unused")


class MotionPlanner:
    """
    Motion planner object that supports both base and arm motion
    """

    def __init__(
        self,
        env=None,
        base_mp_algo="birrt",
        arm_mp_algo="birrt",
        optimize_iter=0,
        check_selfcollisions_in_path=True,
        check_collisions_with_env_in_path=True,
        full_observability_2d_planning=False,
        collision_with_pb_2d_planning=False,
        visualize_2d_planning=False,
        visualize_2d_result=False,
        mode=''
    ):
        """
        Get planning related parameters.
        """
        self.mode = mode
        self.robot_footprint_radius = 0.3
        self.arm_interaction_length = 0.1
        self.arm_ik_threshold = 0.1 #0.05
        self.floor_num = 0
        self.robot_idn = 0

        # Store internal variables
        self.env = env
        self.robot = self.env.robots[self.robot_idn]

        # Possibly load IK solvers if we're using a manipulation robot
        # If not None, maps arm name to ik lula solver
        self.ik_solver = None
        self.ik_control_idx = None
        if isinstance(self.robot, ManipulationRobot):
            self.ik_solver = OrderedDict()
            self.ik_control_idx = OrderedDict()

            if self.robot.model_name == "Fetch":
                control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[self.robot.default_arm]])
                self.ik_solver[self.robot.default_arm] = IKSolver(
                    robot_description_path=f"{assets_path}/models/fetch/fetch_descriptor.yaml",
                    robot_urdf_path=f"{assets_path}/models/fetch/fetch.urdf",
                    eef_name="gripper_link",
                    default_joint_pos=self.robot.default_joint_pos[control_idx],
                )
                self.ik_control_idx[self.robot.default_arm] = control_idx
            elif self.robot.model_name == "Tiago":
                if self.mode == 'baseline':
                    left_control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx["left"]])
                    print('left_control_idx: {}'.format(left_control_idx))
                    print('************************ PyBullet Enabled **************************')
                    self.ik_solver["left"] = IKSolver(
                        robot_description_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford_left_arm_descriptor.yaml",
                        robot_urdf_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford.urdf",
                        eef_name="gripper_left_grasping_frame",
                        default_joint_pos=self.robot.default_joint_pos[left_control_idx],
                    )
                    self.ik_control_idx["left"] = left_control_idx
                    right_control_idx = self.robot.arm_control_idx["right"]

                    self.ik_solver["right"] = self.ik_solver["left"]
                    self.ik_control_idx["right"] = right_control_idx
                else:
                    left_control_idx = np.concatenate(
                        [self.robot.trunk_control_idx, self.robot.arm_control_idx["left"]])
                    self.ik_solver["left"] = IKSolver(
                        robot_description_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford_left_arm_descriptor.yaml",
                        robot_urdf_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford.urdf",
                        eef_name="gripper_left_grasping_frame",
                        default_joint_pos=self.robot.default_joint_pos[left_control_idx],
                    )
                    self.ik_control_idx["left"] = left_control_idx
                    right_control_idx = self.robot.arm_control_idx["right"]
                    self.ik_solver["right"] = IKSolver(
                        robot_description_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford_right_arm_fixed_trunk_descriptor.yaml",
                        robot_urdf_path=f"{assets_path}/models/tiago/tiago_dual_omnidirectional_stanford.urdf",
                        eef_name="gripper_right_grasping_frame",
                        default_joint_pos=self.robot.default_joint_pos[right_control_idx],
                    )
                    self.ik_control_idx["right"] = right_control_idx
            else:
                raise ValueError("Invalid robot for generating IK solver. Must be either Fetch or Tiago")

        # We ignore all robot self-collisions
        for link in self.robot.links.values():
            self.env.add_ignore_robot_self_collision(robot_idn=self.robot_idn, link=link)

        self.floor_obj = list(self.env.scene.object_registry("category", "floors"))[self.floor_num]
        self.trav_map = self.env.scene.trav_map

        # In order to plan we either need an occupancy grid from the robot or full observability
        assert "occupancy_grid" in self.robot.obs_modalities or full_observability_2d_planning
        self.scan_sensor = None
        # Find the scan sensor from the robot's sensors
        for sensor in self.robot.sensors.values():
            if isinstance(sensor, ScanSensor):
                self.scan_sensor = sensor

        # Types of 2D planning
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=TRUE -> We teleport the robot to locations and check for collisions
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=FALSE -> We use the global occupancy map from the scene
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=FALSE -> We use the occupancy_grid from the lidar sensor
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=TRUE -> [not suported yet]
        self.full_observability_2d_planning = full_observability_2d_planning
        self.collision_with_pb_2d_planning = collision_with_pb_2d_planning
        assert not ((not self.full_observability_2d_planning) and self.collision_with_pb_2d_planning)

        if self.full_observability_2d_planning:
            assert len(self.trav_map.floor_map) == 1  # We assume there is only one floor (not true for Gibson scenes)
            self.map_2d = np.array(self.trav_map.floor_map[self.floor_num])
            self.map_2d = np.array((self.map_2d == 255)).astype(np.float32)
            self.per_pixel_resolution = self.trav_map.trav_map_resolution
            assert np.array(self.map_2d).shape[0] == np.array(self.map_2d).shape[1]
            self.grid_resolution = self.map_2d.shape[0]
            self.occupancy_range = self.grid_resolution * self.per_pixel_resolution
            self.robot_footprint_radius_in_map = int(np.ceil(self.robot_footprint_radius / self.per_pixel_resolution))
        else:
            self.grid_resolution = self.scan_sensor.occupancy_grid_resolution
            self.occupancy_range = self.scan_sensor.occupancy_grid_resolution
            self.robot_footprint_radius_in_map = self.scan_sensor.occupancy_grid_inner_radius

        self.base_mp_algo = base_mp_algo
        self.arm_mp_algo = arm_mp_algo
        # If we plan in the map, we do not need to check rotations: a location is in collision (or not) independently
        # of the orientation. If we use pybullet, we may find some cases where the base orientation changes the
        # collision value for the same location between True/False
        if not self.collision_with_pb_2d_planning:
            self.base_mp_resolutions = np.array([0.05, 0.05, 2 * np.pi])
        else:
            self.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        self.optimize_iter = optimize_iter
        self.initial_height = self.env.env_config["initial_pos_z_offset"]
        self.check_selfcollisions_in_path = check_selfcollisions_in_path
        self.check_collisions_with_env_in_path = check_collisions_with_env_in_path
        self.robot_type = self.robot.model_name

        self.marker = None
        self.marker_direction = None

        if not gm.HEADLESS:
            self.marker = PrimitiveObject(
                prim_path="/World/mp_vis_marker",
                primitive_type="Sphere",
                name="mp_vis_marker",
                rgba=[0, 0, 1, 1],
                radius=0.04,
                visual_only=True,
            )
            self.marker_direction = PrimitiveObject(
                prim_path="/World/mp_vis_marker_direction",
                primitive_type="Cylinder",
                name="mp_vis_marker_direction",
                rgba=[0, 0, 1, 1],
                radius=0.01,
                height=0.2,
                visual_only=True,
            )
            if self.mode == 'baseline':
                og.sim.import_object(self.marker, register=False, auto_initialize=True)
                og.sim.import_object(self.marker_direction, register=False, auto_initialize=True)
                # Need to render first before setting visibility. Otherwise, segfault will happen.
                og.sim.render()
                self.marker.visible = False
                self.marker_direction.visible = False
            else:
                self.env.simulator.import_object(self.marker, register=False, auto_initialize=True)
                self.env.simulator.import_object(self.marker_direction, register=False, auto_initialize=True)

        self.visualize_2d_planning = visualize_2d_planning
        self.visualize_2d_result = visualize_2d_result

        self.mp_obstacles = []
        self.mp_obstacles = set(self.mp_obstacles)
        self.enable_simulator_sync = True
        self.enable_simulator_step = False

    def simulator_step(self):
        """Step the simulator and sync the simulator to renderer"""
        og.sim.step(render=True)

    def plan_base_motion(self, goal, plan_full_base_motion=True, objects_to_ignore=None):
        """
        Plan base motion given a base goal location and orientation

        Args:
            goal (3-array): base goal location (x, y, theta) in global coordinates
            plan_full_base_motion (bool): compute only feasibility of the goal location and return it as only point in the
                path
            objects_to_ignore (None or list of BaseObject): If specified, should be object(s) to ignore in the
                collision checking (only when actively using collision checking). This is useful to plan base motions
                while grasping objects

        Returns:
            None or list: path or None if no path can be found
        """

        if self.marker is not None:
            self.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        log.debug("Motion planning base goal: {}".format(goal))

        # Save state to reload
        state = og.sim.dump_state(serialized=False)
        x, y, theta = goal

        # Check only feasibility of the last location
        if not plan_full_base_motion:
            # Ignore any objects being currently held
            objects_to_ignore = [] if objects_to_ignore is None else objects_to_ignore
            objects_to_ignore.append(self.floor_obj)
            if isinstance(self.robot, ManipulationRobot):
                obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm]
                if obj_in_hand is not None:
                    objects_to_ignore.append(obj_in_hand)

            for obj in objects_to_ignore:
                self.env.add_ignore_robot_object_collision(robot_idn=self.robot_idn, obj=obj)

            collision_free = test_valid_pose(
                obj=self.robot,
                pos=np.array((x, y, self.trav_map.floor_heights[self.floor_num])),
                quat=euler2quat(np.array([0, 0, theta])),
                z_offset=self.initial_height,
            )

            for obj in objects_to_ignore:
                self.env.remove_ignore_robot_object_collision(robot_idn=self.robot_idn, obj=obj)

            if collision_free:
                path = [goal]
            else:
                log.warning("Goal in collision")
                path = None
        else:
            robot_obs = self.env.get_obs()[self.robot.name]
            map_2d = robot_obs[f"{self.scan_sensor}_occupancy_grid"] if not self.full_observability_2d_planning else self.map_2d

            if not self.full_observability_2d_planning:
                yaw = self.robot.get_rpy()[2]
                half_occupancy_range = self.occupancy_range / 2.0
                robot_position_xy = self.robot.get_position()[:2]
                corners = [
                    robot_position_xy + (euler2mat([0., 0., yaw]) @ np.append(local_corner, 1.0))[:2]
                    for local_corner in [
                        np.array([half_occupancy_range, half_occupancy_range]),
                        np.array([half_occupancy_range, -half_occupancy_range]),
                        np.array([-half_occupancy_range, half_occupancy_range]),
                        np.array([-half_occupancy_range, -half_occupancy_range]),
                    ]
                ]
            else:
                top_left = self.trav_map.map_to_world(np.array([0, 0]))
                bottom_right = self.trav_map.map_to_world(np.array(self.map_2d.shape) - np.array([1, 1]))
                corners = [top_left, bottom_right]

            obj_prim_paths_to_ignore = set() if objects_to_ignore is None else \
                {link_prim_path for obj in objects_to_ignore for link_prim_path in obj.link_prim_paths}

            # Define obstacles
            obstacles = self.mp_obstacles - set(self.floor_obj.link_prim_paths) - obj_prim_paths_to_ignore if \
                self.collision_with_pb_2d_planning else set()

            robot_body_id = list(self.ik_solver.values())[0].robot_body_id

            path = plan_base_motion_2d(
                robot_body_id,
                [x, y, theta],
                (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
                map_2d=map_2d,
                occupancy_range=self.occupancy_range,
                grid_resolution=self.grid_resolution,
                # If we use the global map, it has been eroded: we do not need to use the full size of the robot, a 1 px
                # robot would be enough
                robot_footprint_radius_in_map=[self.robot_footprint_radius_in_map, 1][
                    self.full_observability_2d_planning
                ],
                resolutions=self.base_mp_resolutions,
                # Add all objects in the scene as obstacles except the robot itself and the floor
                obstacles=obstacles,
                algorithm=self.base_mp_algo,
                optimize_iter=self.optimize_iter,
                visualize_planning=self.visualize_2d_planning,
                visualize_result=self.visualize_2d_result,
                metric2map=[None, self.trav_map.world_to_map][self.full_observability_2d_planning],
                flip_vertically=self.full_observability_2d_planning,
                use_pb_for_collisions=self.collision_with_pb_2d_planning,
                robot=self.robot,
                simulator=og.sim,
            )

        if path is not None and len(path) > 0:
            log.debug("Path found!")
        else:
            log.debug("Path NOT found!")

        # Restore original state
        og.sim.load_state(state=state, serialized=False)
        return path

    def visualize_base_path(self, path, keep_last_location=True):
        """
        Dry run base motion plan by setting the base positions without physics simulation

        :param path: base waypoints or None if no plan can be found
        """

        if path is not None:
            # If we are not keeping the last location, se save the state to reload it after the visualization
            if not keep_last_location:
                initial_state = og.sim.dump_state(serialized=False)

            grasping_object = self.robot.is_grasping() == IsGraspingState.TRUE
            grasped_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]

            if grasping_object:
                gripper_pos = self.robot.get_eef_position(arm="default")
                gripper_orn = self.robot.get_eef_orientation(arm="default")
                obj_pos, obj_orn = grasped_obj.get_position_orientation()
                grasp_pose = T.relative_pose_transform(gripper_pos, gripper_orn, obj_pos, obj_orn)

            if not gm.HEADLESS:
                for way_point in path:
                    robot_position, robot_orn = self.robot.get_position_orientation()
                    robot_position[0] = way_point[0]
                    robot_position[1] = way_point[1]
                    robot_orn = T.euler2quat([0, 0, way_point[2]])
                    self.robot.set_position_orientation(robot_position, robot_orn)
                    if grasping_object:
                        gripper_pos = self.robot.get_eef_position(arm="default")
                        gripper_orn = self.robot.get_eef_orientation(arm="default")
                        object_pose = T.pose_transform(grasp_pose[0], grasp_pose[1], gripper_pos, gripper_orn)
                    og.sim.step()
                    time.sleep(0.01)
            else:
                robot_position, robot_orn = self.robot.get_position_orientation()
                robot_position[0] = path[-1][0]
                robot_position[1] = path[-1][1]
                robot_orn = T.euler2quat([0, 0, path[-1][2]])
                self.robot.set_position_orientation(robot_position, robot_orn)
                if grasping_object:
                    gripper_pos = self.robot.get_eef_position(arm="default")
                    gripper_orn = self.robot.get_eef_orientation(arm="default")
                    object_pose = T.pose_transform(grasp_pose[0], grasp_pose[1], gripper_pos, gripper_orn)

            if not keep_last_location:
                log.info("Not keeping the last state, only visualizing the path and restoring at the end")
                og.sim.load_state(state=initial_state, serialized=False)

    def get_ik_parameters(self, arm="default"):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """

        arm = self.robot.default_arm
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        if self.robot_type == "Fetch":
            arm_joint_pb_ids = np.array(
                joints_from_names(self.ik_solver[arm].robot_body_id, self.robot.arm_joint_names[self.robot.default_arm])
            )
            max_limits_arm = get_max_limits(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids)
            max_limits = [0.5, 0.5] + [max_limits_arm[0]] + [0.5, 0.5] + list(max_limits_arm[1:]) + [0.05, 0.05]
            min_limits_arm = get_min_limits(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids)
            min_limits = [-0.5, -0.5] + [min_limits_arm[0]] + [-0.5, -0.5] + list(min_limits_arm[1:]) + [0.0, 0.0]
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            current_position = get_joint_positions(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids)
            rest_position = [0.0, 0.0] + [current_position[0]] + [0.0, 0.0] + list(current_position[1:]) + [0.01,
                                                                                                            0.01]
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]
        elif self.robot_type == "Tiago":
            max_limits = get_max_limits(self.ik_solver[arm].robot_body_id, get_movable_joints(self.ik_solver[arm].robot_body_id))
            min_limits = get_min_limits(self.ik_solver[arm].robot_body_id, get_movable_joints(self.ik_solver[arm].robot_body_id))
            current_position = get_joint_positions(self.ik_solver[arm].robot_body_id, get_movable_joints(self.ik_solver[arm].robot_body_id))
            rest_position = list(current_position)
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_damping = [0.1 for _ in joint_range]
        elif self.robot_type == "BehaviorRobot":
            log.warning("Not implemented!")
        else:
            log.warning("Robot type is not compatible with IK for motion planning")
            raise ValueError

        return max_limits, min_limits, rest_position, joint_range, joint_damping

    def get_joint_pose_for_ee_pose_with_ik(
        self,
        ee_position,
        ee_orientation=None,
        arm=None,
        check_selfcollisions=False,  # True,
        check_collisions_with_env=True,
        randomize_initial_pose=True,  # True,
        obj_name=None,
    ):
        """
        Attempt to find arm_joint_pose that satisfies ee_position (and possibly, ee_orientation)
        If failed, return None
        NOTE: This function must remain SAFE, i.e., the state of the simulator before and after calling this function
        should be the same, no matter if the function succeeds or fails

        :param ee_position: desired position of the end-effector [x, y, z] in the world frame
        :param ee_orientation: desired orientation of the end-effector [x,y,z,w] in the world frame
        :param arm: string of the name of the arm to use. Use default arm if None
        :param check_selfcollisions: whether we check for selfcollisions (robot-robot) in the solution or not
        :param check_collisions_with_env: whether we check for collisions robot-environment in the solution or not
        :return: arm joint_pose
        """
        log.debug("IK query for end-effector pose ({}, {}) with arm {}".format(ee_position, ee_orientation, arm))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to get IK for the default arm: {}".format(arm))
        if self.mode != 'baseline':
            if self.robot_type == "BehaviorRobot":
                # TODO
                if arm == "left_hand":
                    position_arm_shoulder_in_bf = np.array(
                        [0, 0, 0]
                    )  # TODO: get the location we set the max hand distance from
                elif arm == "right_hand":
                    position_arm_shoulder_in_bf = np.array(
                        [0, 0, 0]
                    )  # TODO: get the location we set the max hand distance from
                body_pos, body_orn = self.robot.get_position_orientation()
                position_arm_shoulder_in_wf, _ = p.multiplyTransforms(
                    body_pos, body_orn, position_arm_shoulder_in_bf, [0, 0, 0, 1]
                )
                if T.l2_distance(ee_position, position_arm_shoulder_in_wf) > 0.7:  # TODO: get max distance
                    return None
                else:
                    if ee_orientation is not None:
                        return np.concatenate((ee_position, p.getEulerFromQuaternion(ee_orientation)))
                    else:
                        current_orientation = np.array(self.robot.get_eef_orientation(arm=arm))
                        current_orientation_rpy = p.getEulerFromQuaternion(current_orientation)
                        return np.concatenate((ee_position, np.asarray(current_orientation_rpy)))

            ik_start = time.time()
            rest_position = self.get_ik_parameters()

            n_attempt = 0
            max_attempt = 75

            state = og.sim.dump_state(serialized=False)

        base_pos, base_quat = self.robot.get_position_orientation()
        base_pos_col_check = base_pos + np.array([0,0,self.initial_height])
        control_idx = self.ik_control_idx[arm]
        n_control_idx = len(control_idx)
        current_joint_pos = self.robot.get_joint_positions()
        jnt_range = (self.robot.joint_upper_limits - self.robot.joint_lower_limits)[control_idx]
        jnt_low = self.robot.joint_lower_limits[control_idx]
        jnt_middle = ((self.robot.joint_upper_limits + self.robot.joint_lower_limits) / 2.0)[control_idx]

        ee_orientation = np.array([0, 0, 0, 1]) if ee_orientation is None else ee_orientation
        if obj_name in [
            "microwave.n.02_1-cleaning_microwave_oven",
        ]:
            ee_orientation = np.array([0.1830127, -0.1830127, 0.6830127, 0.6830127])

        log.debug(f"Original IK query at {ee_position}, {ee_orientation} in world frame")
        # There are 2 options:
        # a) we move the pybullet robot to the current robot location and query a desired ee location in world frame
        # b) we keep the robot at the origin and we use the desired ee location in robot base link
        # We are going with a)
        robot_position, robot_orientation_q = self.robot.get_position_orientation()
        base_pose = base_values_from_pose((robot_position, robot_orientation_q))
        set_base_values_with_z(self.ik_solver[arm].robot_body_id, base_pose, z=0)
        p.stepSimulation()

        arm_joint_pb_ids = np.array(joints_from_names(self.ik_solver[arm].robot_body_id, self.robot.arm_joint_names[arm]))
        sample_fn = get_sample_fn(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids)

        initial_pb_state = p.saveState()

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 100

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:

            if randomize_initial_pose:
                # Start the iterative IK from a different random initial joint pose
                set_joint_positions(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids, sample_fn())

            kwargs = dict()
            if ee_orientation is not None:
                kwargs["targetOrientation"] = ee_orientation

            ee_link_id = link_from_name(self.ik_solver[arm].robot_body_id, self.robot.eef_link_names[arm])
            joint_pose = p.calculateInverseKinematics(
                self.ik_solver[arm].robot_body_id,
                ee_link_id,
                targetPosition=ee_position,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                maxNumIterations=100,
                **kwargs,
            )
            # Pybullet returns the joint poses for the entire body. Get only for the relevant arm
            arm_movable_joint_pb_idx = movable_from_joints(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids)
            joint_pose = np.array(joint_pose)[arm_movable_joint_pb_idx]
            set_joint_positions(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids, joint_pose)

            ee_current_position = get_link_position_from_name(self.ik_solver[arm].robot_body_id, self.robot.eef_link_names[arm]) #self.robot.get_eef_position(arm=arm)  #TODO:
            dist = l2_distance(ee_current_position, ee_position)
            if dist > self.arm_ik_threshold:
                # print("IK solution is not close enough to the desired pose. Distance: {}, ee_current_position: {}, "
                #       "self.ik_solver[arm].robot_body_id: {}, self.robot.eef_link_names[arm]: {}".format(dist,
                #      ee_current_position, self.ik_solver[arm].robot_body_id, self.robot.eef_link_names[arm]))
                n_attempt += 1
                continue

            if check_selfcollisions:    # This can be done with pybullet, as it contains the robot
                p.stepSimulation()
                # simulator_step will slightly move the robot base and the objects
                set_base_values_with_z(self.ik_solver[arm].robot_body_id, base_pose, z=0)

                # Only the robot is in pybullet. If there are collisions, are self-collisions
                selfcollision_free = True
                disabled_self_colliding_pairs = {
                    (link_from_name(self.ik_solver[arm].robot_body_id, link1),
                     link_from_name(self.ik_solver[arm].robot_body_id, link2))
                    for (link1, link2) in self.robot.disabled_collision_pairs
                }
                check_link_pairs = get_self_link_pairs(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids, disabled_self_colliding_pairs)
                for link1, link2 in check_link_pairs:
                    if pairwise_link_collision(self.ik_solver[arm].robot_body_id, link1, self.ik_solver[arm].robot_body_id, link2):
                        selfcollision_free = False
                        break

                # gripper should not have any self-collision
                ee_link_id = link_from_name(self.ik_solver[arm].robot_body_id, self.robot.eef_link_names[arm])
                ee_selfcollision_free = is_collision_free(
                    body_a=self.ik_solver[arm].robot_body_id,
                    link_a_list=[ee_link_id],
                    body_b=self.ik_solver[arm].robot_body_id,
                )
                if not selfcollision_free or not ee_selfcollision_free:
                    n_attempt += 1
                    log.debug("IK solution brings the robot to self collision")
                    continue

            if not self.mode == 'baseline':

                # Restore state
                og.sim.load_state(state=state, serialized=False)
                # app.update()
            else:
                p.restoreState(initial_pb_state)
                p.removeState(initial_pb_state)
            log.debug("IK Solver found a valid configuration")
            return joint_pose

        if not self.mode == 'baseline':
            og.sim.load_state(state=state, serialized=False)
        else:
            p.restoreState(initial_pb_state)
            p.removeState(initial_pb_state)
        log.debug("IK Solver failed to find a configuration")
        return None

    def plan_arm_motion_to_joint_pose(
        self, arm_joint_pose, initial_arm_configuration=None, arm=None, disable_collision_hand_links=False
    ):
        """
        Plan a joint space collision-free trajectory from the current or the given initial configuration to the given
        arm_joint_pose configuration and return the computed path
        If failed, reset the arm to its original configuration and return None

        :param arm_joint_pose: final arm joint configuration to try to reach in collision-free space
        :param initial_arm_configuration: initial joint configuration to initialize the arm. Use the current
        configuration, if it is `None`
        :param arm: arm to use for planning. Use the default arm if `None`
        :param disable_collision_hand_links: if True, include Fetch hand and finger collisions while motion planning
        :return: arm trajectory or None if no plan can be found
        """
        log.warning("Planning path in joint space to {}".format(arm_joint_pose))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to planning a joint space trajectory with the default arm: {}".format(arm))

        disabled_self_colliding_pairs = {}
        if self.check_selfcollisions_in_path:
            if self.robot_type != "BehaviorRobot":  # For any other robot, remove the collisions from the list
                disabled_self_colliding_pairs = {
                    (link_from_name(self.ik_solver[arm].robot_body_id, link1), link_from_name(self.ik_solver[arm].robot_body_id, link2))
                    for (link1, link2) in self.robot.disabled_collision_pairs
                }

        if self.check_collisions_with_env_in_path:
            mp_obstacles = self.mp_obstacles
        else:
            mp_obstacles = []

        state = og.sim.dump_state(serialized=False)

        if initial_arm_configuration is not None:
            log.warning(
                "Start the planning trajectory from a given joint configuration is not implemented yet."
                "Should it use the utils function or the iG function to move the arm? What indices?"
            )
            exit(-1)
        else:
            # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
            self.robot.tuck()
            self.simulator_step()

        disabled_colliding_links = []
        if disable_collision_hand_links:
            disabled_colliding_links = [self.robot.eef_links[arm].link_id] + [
                finger.link_id for finger in self.robot.finger_links[arm]
            ]

        if self.robot_type != "BehaviorRobot":
            arm_joint_pb_ids = np.array(joints_from_names(self.ik_solver[arm].robot_body_id, self.robot.arm_joint_names[arm]))
            arm_path = plan_joint_motion(
                body=self.ik_solver[arm].robot_body_id,
                joints=arm_joint_pb_ids,
                end_conf=arm_joint_pose,
                disabled_colliding_links=disabled_colliding_links,
                check_self_collisions=self.check_selfcollisions_in_path,
                disabled_self_colliding_pairs=disabled_self_colliding_pairs,
                obstacles=mp_obstacles,
                algorithm=self.arm_mp_algo,
                robot=self.robot,
                simulator=og.sim,
                og_joint_ids=np.concatenate((self.robot.trunk_control_idx, self.robot.arm_control_idx[arm])),
            )
        else:
            arm_path = plan_hand_motion_br(
                self.robot,
                arm,
                arm_joint_pose,
                disabled_colliding_links=disabled_colliding_links,
                check_self_collisions=check_self_collisions,
                disabled_self_colliding_pairs=disabled_self_colliding_pairs,
                obstacles=mp_obstacles,
                algorithm=self.arm_mp_algo,
            )

        og.sim.load_state(state=state, serialized=False)

        if arm_path is not None and len(arm_path) > 0:
            log.warning("Path found!")
        else:
            log.warning("Path NOT found!")

        return arm_path

    def plan_ee_motion_to_cartesian_pose(
        self, ee_position, ee_orientation=None, initial_arm_configuration=None, arm=None, set_marker=True
    ):
        """
        Attempt to reach a Cartesian 6D pose (position and possibly orientation) with the end effector of one arm.

        :param ee_position: desired position to reach with the end-effector [x, y, z] in the world frame
        :param ee_orientation: desired orientation of the end-effector [x,y,z,w] in the world frame
        :param initial_arm_configuration: initial joint configuration to initialize the arm. Use the current
        configuration, if it is `None`
        :param arm: arm to use for planning. Use the default arm if `None`
        :param set_marker: whether we set the visual marker at the given goal
        :return: arm trajectory or None if no plan can be found
        """
        log.warning("Planning arm motion to end-effector pose ({}, {})".format(ee_position, ee_orientation))
        if self.marker is not None and set_marker:
            self.set_marker_position_direction(ee_position, [0, 0, 1])

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to plan EE to Cartesian pose with the default arm: {}".format(arm))

        # Solve the IK problem to set the arm at the desired position
        joint_pose = self.get_joint_pose_for_ee_pose_with_ik(ee_position, ee_orientation=ee_orientation, arm=arm)

        if joint_pose is not None:
            path = self.plan_arm_motion_to_joint_pose(
                joint_pose, initial_arm_configuration=initial_arm_configuration, arm=arm
            )
            if path is not None and len(path) > 0:
                log.warning("Planning succeeded: found path in joint space to Cartesian goal")
            else:
                log.warning("Planning failed: no collision free path to Cartesian goal")
            return path
        else:
            log.warning("Planning failed: goal position may be non-reachable")
            return None

    def plan_ee_straight_line_motion(
        self,
        initial_arm_pose,
        line_initial_point,
        line_direction,
        ee_orn=None,
        line_length=0.2,
        steps=50,
        arm=None,
    ):
        log.warning(
            "Planning straight line motion from {} along {} a distance of {}".format(
                line_initial_point, line_direction, line_length
            )
        )
        log.warning("Initial joint configuration {}".format(initial_arm_pose))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Planning straight line for the default arm: {}".format(arm))

        if line_length == 0.0:
            log.warning("Requested line of length 0. Returning a path with only one configuration: initial_arm_pose")
            return [initial_arm_pose]

        state = og.sim.dump_state(serialized=False)

        # Start planning from the given pose
        if self.robot_type != "BehaviorRobot":
            joint_pos = self.robot.get_joint_positions()
            joint_pos[self.ik_control_idx[arm]] = initial_arm_pose

            arm_joint_pb_ids = np.array(joints_from_names(self.ik_solver[arm].robot_body_id, self.robot.arm_joint_names[arm]))
            set_joint_positions(self.ik_solver[arm].robot_body_id, arm_joint_pb_ids, initial_arm_pose)
        else:
            # TODO
            self.robot.set_eef_position_orientation(
                initial_arm_pose[:3], p.getQuaternionFromEuler(initial_arm_pose[3:]), arm
            )
            if self.enable_simulator_sync:
                self.simulator_sync()

        line_segment = np.array(line_direction) * (line_length)

        line_path = []

        for i in range(steps):
            start_push_goal = time.time()
            push_goal = line_initial_point + line_segment * (i + 1) / float(steps)
            # Solve the IK problem to set the arm at the desired position
            joint_pose = self.get_joint_pose_for_ee_pose_with_ik(
                push_goal, ee_orientation=ee_orn, arm=arm, check_selfcollisions=True, check_collisions_with_env=False, randomize_initial_pose=False
            )
            start_restore = time.time()
            if joint_pose is None:
                og.sim.load_state(state=state, serialized=False)
                log.warning("Failed to retrieve IK solution for EE line path. Failure.")
                return None

            line_path.append(joint_pose)
            end_restore = time.time()

        og.sim.load_state(state=state, serialized=False)

        return line_path


    def plan_ee_pick(
        self,
        grasping_location,
        grasping_direction=np.array([0.0, 0.0, -1.0]),
        pre_grasping_distance=0.1,
        grasping_steps=50,
        plan_full_pre_grasp_motion=False,
        arm=None,
    ):
        """
        Plans a full grasping trajectory.
        The trajectory includes a pre-grasping motion in free space (contact-free) to a location in front of the grasping
        location, and a grasping motion (with collisions, to grasp) in a straight line in Cartesian space along the
        grasping direction

        :param grasping_location: 3D point to push at
        :param grasping_direction: direction to approach the object to grasp. Default: [0,0,-1] -> top-down
        :param pre_grasping_distance: distance in front of the grasping point along the grasping direction to plan a motion
            in free space before the approach to grasp.
        :param grasping_steps: steps to compute IK along a straight line for the second phase of the grasp (interaction)
        :param arm: which arm to use for multi-arm agents. `None` to use the default arm
        :return: tuple of (pre_grasp_path, grasp_interaction_path) with the joint space trajectory to follow in the two
        phases of the interaction. Both will be None if the planner fails to find a path
        """
        log.debug(
            "Planning end-effector grasping action at point {} with direction {}".format(
                grasping_location, grasping_direction
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(grasping_location, grasping_direction)

        if arm is None:
            arm = self.robot.default_arm
            log.debug("Grasping with the default arm: {}".format(arm))

        pre_grasping_location = grasping_location - pre_grasping_distance * grasping_direction
        log.debug(
            "It will plan a motion to the location {}, {} m in front of the grasping location"
            "".format(pre_grasping_distance, pre_grasping_location)
        )

        # Compute orientation of the hand to align the fingers to the grasping direction and the palm up to the ceiling
        # Assuming fingers point in +x, palm points in +z in eef
        # In world frame, +z points up
        desired_x_dir_normalized = np.array(grasping_direction) / np.linalg.norm(np.array(grasping_direction))
        z_dir_in_wf = np.array([0, 0, 1.0])
        desired_y_dir = -np.cross(desired_x_dir_normalized, z_dir_in_wf)

        if np.linalg.norm(desired_y_dir) < 0.05:
            log.debug("Approaching grasping location top-down")
            desired_y_dir_normalized = np.array([0.0, 1.0, 0.0])
        else:
            desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
        desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
        rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
        quatt = T.mat2quat(rot_matrix)

        if plan_full_pre_grasp_motion:
            pre_grasp_path = self.plan_ee_motion_to_cartesian_pose(pre_grasping_location, ee_orientation=quatt, arm=arm)
        else:
            log.debug("Not planning the pre-grasp path, only checking feasibility of the last location.")
            last_pose = self.get_joint_pose_for_ee_pose_with_ik(pre_grasping_location, ee_orientation=quatt, arm=arm)
            pre_grasp_path = [last_pose] if last_pose is not None else []

        grasp_interaction_path = None

        if pre_grasp_path is None or len(pre_grasp_path) == 0:
            print("Planning failed: no path found to pre-grasping location")
        elif pre_grasping_distance == 0:
            print("Skipping computation of interaction path because the pre-grasping distance is zero")
            grasp_interaction_path = []
        else:
            grasp_interaction_path = self.plan_ee_straight_line_motion(
                pre_grasp_path[-1],
                grasping_location
                - grasping_direction * pre_grasping_distance,  # This is where the pre-grasping path should end
                grasping_direction,
                ee_orn=quatt,
                line_length=pre_grasping_distance,
                steps=grasping_steps,
                arm=arm,
            )
            if grasp_interaction_path is None:
                print("Planning failed: no path found to grasp the object")

        return pre_grasp_path, grasp_interaction_path

    def plan_ee_place(
        self,
        placing_location,
        placing_direction=np.array([-1.0, 0.0, 0.0]),
        placing_ee_orientation=np.array([0.0, 0.0, -1.0]),
        plan_full_pre_place_motion=False,
        pre_placing_distance=0.19,
        arm=None,
    ):
        """
        Plans a full grasping trajectory.
        The trajectory includes a pre-grasping motion in free space (contact-free) to a location in front of the grasping
        location, and a grasping motion (with collisions, to grasp) in a straight line in Cartesian space along the
        grasping direction

        :param grasping_location: 3D point to push at
        :param grasping_direction: direction to approach the object to grasp. Default: [0,0,-1] -> top-down
        :param pre_grasping_distance: distance in front of the grasping point along the grasping direction to plan a motion
            in free space before the approach to grasp.
        :param grasping_steps: steps to compute IK along a straight line for the second phase of the grasp (interaction)
        :param arm: which arm to use for multi-arm agents. `None` to use the default arm
        :return: tuple of (pre_grasp_path, grasp_interaction_path) with the joint space trajectory to follow in the two
        phases of the interaction. Both will be None if the planner fails to find a path
        """
        log.debug(
            "Planning end-effector grasping action at point {} with direction {}".format(
                placing_location, placing_direction
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(placing_location, placing_ee_orientation)

        if arm is None:
            arm = self.robot.default_arm
            log.debug("Placing with the default arm: {}".format(arm))

        log.debug(
            "It will plan a motion to the location {}, {} m above of the placing location"
            "".format(pre_placing_distance, placing_location)
        )

        # Compute orientation of the hand to align the fingers to the grasping direction and the palm up to the ceiling
        # Assuming fingers point in +x, palm points in +z in eef
        # In world frame, +z points up
        desired_x_dir_normalized = np.array(placing_ee_orientation) / np.linalg.norm(np.array(placing_ee_orientation))
        z_dir_in_wf = np.array([0, 0, 1.0])
        desired_y_dir = -np.cross(desired_x_dir_normalized, z_dir_in_wf)

        if np.linalg.norm(desired_y_dir) < 0.05:
            log.debug("Approaching placing location top-down")
            desired_y_dir_normalized = np.array([0.0, 1.0, 0.0])
        else:
            desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
        desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
        rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
        quatt = T.mat2quat(rot_matrix)

        if plan_full_pre_place_motion:
            pre_place_path = self.plan_ee_motion_to_cartesian_pose(placing_location, ee_orientation=quatt, arm=arm)
        else:
            log.debug("Not planning the pre-place path, only checking feasibility of the last location.")
            last_pose = self.get_joint_pose_for_ee_pose_with_ik(placing_location, ee_orientation=quatt, arm=arm)
            pre_place_path = [last_pose] if last_pose is not None else []

        print("Place pre Place path success")

        if pre_place_path is not None:
            if plan_full_pre_place_motion:
                pre_place_above_path = self.plan_ee_straight_line_motion(
                    pre_place_path[-1],
                    placing_location, # This is where the pre-above path should end
                    -1.0 * placing_direction,
                    ee_orn=quatt,
                    line_length=pre_placing_distance,
                    steps=10,
                    arm=arm,
                )
            else:
                last_pose_above = self.get_joint_pose_for_ee_pose_with_ik(placing_location - pre_placing_distance * placing_direction, ee_orientation=quatt, arm=arm)
                pre_place_above_path = [last_pose_above] if last_pose_above is not None else []
        else:
            pre_place_above_path = []

        return pre_place_path, pre_place_above_path


    def plan_ee_push(
        self,
        pushing_location,
        pushing_direction,
        pushing_ee_orientation=np.array([0.0, 0.0, -1.0]),
        inserting_distance=0.3,
        inserting_steps=20,
        pre_pushing_distance=0.25,
        pushing_steps=10,
        plan_full_pre_push_motion=False,
        arm=None,
    ):
        """
        Plans a full grasping trajectory.
        The trajectory includes a pre-grasping motion in free space (contact-free) to a location in front of the grasping
        location, and a grasping motion (with collisions, to grasp) in a straight line in Cartesian space along the
        grasping direction

        :param grasping_location: 3D point to push at
        :param grasping_direction: direction to approach the object to grasp. Default: [0,0,-1] -> top-down
        :param pre_grasping_distance: distance in front of the grasping point along the grasping direction to plan a motion
            in free space before the approach to grasp.
        :param grasping_steps: steps to compute IK along a straight line for the second phase of the grasp (interaction)
        :param arm: which arm to use for multi-arm agents. `None` to use the default arm
        :return: tuple of (pre_grasp_path, grasp_interaction_path) with the joint space trajectory to follow in the two
        phases of the interaction. Both will be None if the planner fails to find a path
        """
        log.debug(
            "Planning end-effector grasping action at point {} with direction {}".format(
                pushing_location, pushing_ee_orientation
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(pushing_location, pushing_ee_orientation)

        if arm is None:
            arm = self.robot.default_arm
            log.debug("Grasping with the default arm: {}".format(arm))

        log.debug(
            "It will plan a motion to the location {}, {} m above of the pushing location"
            "".format(pre_pushing_distance, pushing_location)
        )

        # Compute orientation of the hand to align the fingers to the grasping direction and the palm up to the ceiling
        # Assuming fingers point in +x, palm points in +z in eef
        # In world frame, +z points up
        desired_x_dir_normalized = np.array(pushing_ee_orientation) / np.linalg.norm(np.array(pushing_ee_orientation))
        z_dir_in_wf = np.array([0, 0, 1.0])
        desired_y_dir = -np.cross(desired_x_dir_normalized, z_dir_in_wf)

        if np.linalg.norm(desired_y_dir) < 0.05:
            log.debug("Approaching grasping location top-down")
            desired_y_dir_normalized = np.array([0.0, 1.0, 0.0])
        else:
            desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
        desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
        rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
        quatt = T.mat2quat(rot_matrix)

        if plan_full_pre_push_motion:
            pre_push_path = self.plan_ee_motion_to_cartesian_pose(pushing_location, ee_orientation=quatt, arm=arm)
        else:
            log.debug("Not planning the pre-grasp path, only checking feasibility of the last location.")
            last_pose = self.get_joint_pose_for_ee_pose_with_ik(pushing_location, ee_orientation=quatt, arm=arm)
            pre_push_path = [last_pose] if last_pose is not None else []

        if pre_push_path is not None and len(pre_push_path) > 1:
            pre_insert_path = self.plan_ee_straight_line_motion(
                pre_push_path[-1],
                pushing_location, # This is where the pre-above path should end
                -1.0 * pushing_direction,
                ee_orn=quatt,
                line_length=pre_pushing_distance,
                steps=inserting_steps,
                arm=arm,
            )

            insert_path = self.plan_ee_straight_line_motion(
                pre_push_path[-1],
                pushing_location - pre_pushing_distance * pushing_direction,
                # This is where the pre-inserting path should end
                pushing_ee_orientation,
                ee_orn=quatt,
                line_length=inserting_distance,  # insert more into the drawer
                steps=inserting_steps,
                arm=arm,
            )

            if insert_path is not None and len(insert_path) > 1:
                push_interaction_path = self.plan_ee_straight_line_motion(
                    insert_path[-1],
                    pushing_location + inserting_distance * pushing_ee_orientation - pre_pushing_distance * pushing_direction,
                    # This is where the pre-pushing path should end
                    pushing_direction,
                    ee_orn=quatt,
                    line_length=pre_pushing_distance * 0.23,
                    steps=pushing_steps,
                    arm=arm,
                )

                if push_interaction_path is None:
                    log.debug("Planning failed: no path found to push the object")
            else:
                pre_insert_path, insert_path, push_interaction_path = [], [], []
        else:
            pre_insert_path, insert_path, push_interaction_path = [], [], []

        return pre_push_path, pre_insert_path, insert_path, push_interaction_path

    def plan_ee_drop(
        self,
        dropping_location,
        ee_dropping_orn=None,
        dropping_distance=0.3,
        plan_full_pre_drop_motion=True,
        arm=None,
        obj_name=None,
    ):
        """
        Plans a full dropping trajectory.
        The trajectory includes a pre-dropping motion in free space (contact-free) to a location in front of the dropping
        location, and a dropping motion (with collisions, to interact) in a straight line in Cartesian space along the
        dropping direction

        :param dropping_location: 3D point to push at
        :param ee_dropping_orn: orientation of the end effector during dropping [x,y,z,w]. None if we do not constraint it
        :param dropping_distance: distance above the dropping point in the vertical direction to plan a motion
        in free space
        :param arm: which arm to use for multi-arm agents. `None` to use the default arm
        :return: tuple of (pre_drop_path, []) with the joint space trajectory to follow for the single phase of the
            interaction (second phase is empty). pre_drop_path will be None if the planner fails to find a path
        """
        # TODO: it can be better to align the end-effector with the fingers towards the floor
        log.warning(
            "Planning end-effector dropping action over point {} a distance {}".format(
                dropping_location, dropping_distance
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(dropping_location, np.array([0, 0, 1]))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("dropping with the default arm: {}".format(arm))

        pre_dropping_location = dropping_location + dropping_distance * np.array([0, 0, 1])
        log.warning("Predropping location {}".format(pre_dropping_location))

        if plan_full_pre_drop_motion:
            pre_drop_path = self.plan_ee_motion_to_cartesian_pose(
                pre_dropping_location, ee_orientation=ee_dropping_orn, arm=arm
            )
        else:
            log.warning("Not planning the pre-drop path, only checking feasibility of the last location.")
            last_pose = self.get_joint_pose_for_ee_pose_with_ik(
                pre_dropping_location, ee_orientation=ee_dropping_orn, arm=arm, obj_name=obj_name
            )
            pre_drop_path = [last_pose] if last_pose is not None else []

        if pre_drop_path is None or len(pre_drop_path) == 0:
            log.warning("Planning failed: no path found to pre-dropping location")

        return pre_drop_path, []  # There is no second part of the action for dropping


    def visualize_arm_path(self, arm_path, reverse_path=False, arm=None, grasped_obj=None, keep_last_location=True):
        """
        Dry run arm motion path by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no path can be found
        """
        if arm_path is None:
            log.warning("Arm path to visualize is empty. Returning")
            return

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Visualizing arm path for the default arm: {}".format(arm))

        if not keep_last_location:
            state = og.sim.dump_state(serialized=False)

        if grasped_obj is not None:
            if self.robot_type != "BehaviorRobot":
                # Get global EEF pose
                grasp_pose = (self.robot.get_eef_position(arm=arm), self.robot.get_eef_orientation(arm=arm))
            else:
                gripper_pos = self.robot.get_eef_position(arm)
                gripper_orn = self.robot.get_eef_orientation(arm)
                obj_pos, obj_orn = p.getBasePositionAndOrientation(grasped_obj_id)
                grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        execution_path = arm_path if not reverse_path else arm_path[::-1]
        execution_path = (
            execution_path if not gm.HEADLESS else [execution_path[-1]]
        )
        if self.robot_type != "BehaviorRobot":
            for joint_way_point in execution_path:
                self.robot.set_joint_positions(joint_way_point, indices=self.ik_control_idx[arm])

                # Teleport also the grasped object
                if grasped_obj is not None:
                    joint_prim_path = f"{self.robot.eef_links[arm].prim_path}/ag_constraint"
                    joint_prim = get_prim_at_path(joint_prim_path)

                    # Set the local pose of this joint
                    local_pos_0 = joint_prim.GetAttribute("physics:localPos0").Get()
                    local_orn_0 = joint_prim.GetAttribute("physics:localRot0").Get()
                    local_pos_1 = joint_prim.GetAttribute("physics:localPos1").Get()
                    local_orn_1 = joint_prim.GetAttribute("physics:localRot1").Get()

                    local_pos_0 = np.array(local_pos_0) * self.robot.scale
                    local_orn_0 = np.array([*local_orn_0.imaginary, local_orn_0.real])
                    local_pos_1 = np.array(local_pos_1) * grasped_obj.scale
                    local_orn_1 = np.array([*local_orn_1.imaginary, local_orn_1.real])

                    gripper_pos = self.robot.get_eef_position(arm)
                    gripper_orn = self.robot.get_eef_orientation(arm)
                    gripper_pose = T.pose2mat((gripper_pos, gripper_orn))
                    jnt_local_pose_0 = T.pose2mat((local_pos_0, local_orn_0))
                    inv_jnt_local_pose_1 = T.pose_inv(T.pose2mat((local_pos_1, local_orn_1)))
                    grasp_pose = gripper_pose @ jnt_local_pose_0 @ inv_jnt_local_pose_1
                    grasp_pos, grasp_orn = T.mat2pose(grasp_pose)

                    self.simulator_step()
                    grasped_obj.set_position_orientation(grasp_pos, grasp_orn)

                self.simulator_step()
                time.sleep(0.01)
        else:
            # TODO
            for (x, y, z, roll, pitch, yaw) in execution_path:
                self.robot.set_eef_position_orientation([x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]), arm)
                # Teleport also the grasped object
                if grasped_obj_id is not None:
                    gripper_pos = self.robot.get_eef_position(arm)
                    gripper_orn = self.robot.get_eef_orientation(arm)
                    object_pose = p.multiplyTransforms(gripper_pos, gripper_orn, grasp_pose[0], grasp_pose[1])
                    set_pose(grasped_obj_id, object_pose)
                time.sleep(0.1)
                if self.enable_simulator_sync:
                    self.simulator_sync()

        if not keep_last_location:
            og.sim.load_state(state=state, serialized=False)

    def set_marker_position(self, pos):
        """
        Set subgoal marker position

        :param pos: position
        """
        self.marker.set_position(pos)

    def set_marker_position_yaw(self, pos, yaw):
        """
        Set subgoal marker position and orientation

        :param pos: position
        :param yaw: yaw angle
        """
        quat = T.euler2quat((0, -np.pi / 2, yaw))
        self.marker.set_position(pos)
        self.marker_direction.set_position_orientation(pos, quat)

    def set_marker_position_direction(self, pos, direction):
        """
        Set subgoal marker position and orientation

        :param pos: position
        :param direction: direction vector
        """
        yaw = np.arctan2(direction[1], direction[0])
        self.set_marker_position_yaw(pos, yaw)

