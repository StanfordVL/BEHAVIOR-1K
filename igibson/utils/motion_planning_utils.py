import logging
import random

from transforms3d import euler

from igibson.robots.manipulation_robot import IsGraspingState

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

import time

import numpy as np
from collections import OrderedDict

# from igibson.external.motion.motion_planners.rrt_connect import birrt
# from igibson.external.pybullet_tools.utils import (
#     PI,
#     circular_difference,
#     direct_path,
#     get_aabb,
#     get_base_values,
#     get_joint_names,
#     get_joint_positions,
#     get_joints,
#     get_max_limits,
#     get_min_limits,
#     get_movable_joints,
#     get_sample_fn,
#     is_collision_free,
#     joints_from_names,
#     link_from_name,
#     movable_from_joints,
#     pairwise_collision,
#     plan_base_motion_2d,
#     plan_joint_motion,
#     set_base_values_with_z,
#     set_joint_positions,
#     set_pose,
# )

# from igibson.external.pybullet_tools.utils import (
#     control_joints,
#     get_base_values,
#     get_joint_positions,
#     get_max_limits,
#     get_min_limits,
#     get_sample_fn,
#     is_collision_free,
#     joints_from_names,
#     link_from_name,
#     plan_base_motion_2d,
#     plan_joint_motion,
#     set_base_values_with_z,
#     set_joint_positions,
#     set_pose,
# )

import igibson.macros as m
from igibson import app, assets_path
from igibson.objects.primitive_object import PrimitiveObject
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.sensors.scan_sensor import ScanSensor
from igibson.scenes.gibson_indoor_scene import StaticTraversableScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
import igibson.utils.transform_utils as T
from igibson.utils.utils import l2_distance, rotate_vector_2d

import lula

SEARCHED = []
# Setting this higher unfortunately causes things to become impossible to pick up (they touch their hosts)
BODY_MAX_DISTANCE = 0.05
HAND_MAX_DISTANCE = 0


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
    ):
        # Create robot description, kinematics, and config
        self.robot_description = lula.load_robot(robot_description_path, robot_urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.config = lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.default_joint_pos = default_joint_pos

    def solve(
        self,
        target_pos,
        target_quat,
        initial_joint_pos=None,
    ):
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        Args:
            target_pos (3-array): desired (x,y,z) local target cartesian position in robot's base coordinate frame
            target_quat (3-array): desired (x,y,z,w) local target quaternion orientation in robot's base coordinate frame
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.default_joint_pos

        Returns:
            None or n-array: Joint positions for reaching desired target_pos and target_quat, otherwise None if no
                solution was found
        """
        pos = np.array(target_pos, dtype=np.float64).reshape(3,1)
        rot = np.array(T.quat2mat(target_quat), dtype=np.float64)
        ik_target_pose = lula.Pose3(lula.Rotation3(rot), pos)

        # Set the cspace seed
        initial_joint_pos = self.default_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        self.config.cspace_seeds = [initial_joint_pos]

        # Compute target joint positions
        ik_results = lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        return np.array(ik_results.cspace_position)


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
        fine_motion_plan=True,
        full_observability_2d_planning=False,
        collision_with_pb_2d_planning=False,
        visualize_2d_planning=False,
        visualize_2d_result=False,
    ):
        """
        Get planning related parameters.
        """

        # TODO: Magic numbers -- change?
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

            # TODO: Make this less hard-coded once MJ's PR is merged in
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
                left_control_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx["left"]])
                self.ik_solver["left"] = IKSolver(
                    robot_description_path=f"{assets_path}/models/tiago/tiago_left_arm_descriptor.yaml",
                    robot_urdf_path=f"{assets_path}/models/tiago/tiago_dual.urdf",
                    eef_name="gripper_left_grasping_frame",
                    default_joint_pos=self.robot.default_joint_pos[left_control_idx],
                )
                self.ik_control_idx["left"] = left_control_idx
                right_control_idx = self.robot.arm_control_idx["right"]
                self.ik_solver["right"] = IKSolver(
                    robot_description_path=f"{assets_path}/models/tiago/tiago_right_arm_fixed_trunk_descriptor.yaml",
                    robot_urdf_path=f"{assets_path}/models/tiago/tiago_dual.urdf",
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
            # TODO: it may be better to unify and make that scene.floor_map uses OccupancyGridState values always
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
        self.fine_motion_plan = fine_motion_plan
        self.robot_type = self.robot.model_name

        self.marker = None
        self.marker_direction = None

        if not m.HEADLESS:
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
                primitive_type="Sphere",
                name="mp_vis_marker_direction",
                rgba=[0, 0, 1, 1],
                radius=0.01,
                length=0.2,
                visual_only=True,
            )
            self.env.simulator.import_object(self.marker, register=False, auto_initialize=True)
            self.env.simulator.import_object(self.marker_direction, register=False, auto_initialize=True)

        self.visualize_2d_planning = visualize_2d_planning
        self.visualize_2d_result = visualize_2d_result

        self.mp_obstacles = []
        if isinstance(self.env.scene, StaticTraversableScene):
            # TODO!
            if self.env.scene.mesh_body_id is not None:
                self.mp_obstacles.append(self.env.scene.mesh_body_id)
        elif isinstance(self.env.scene, InteractiveTraversableScene):
            # Iterate over all objects in the scene and grab their link prim paths
            for obj in self.env.scene.objects:
                self.mp_obstacles += obj.link_prim_paths
        self.mp_obstacles = set(self.mp_obstacles)
        self.enable_simulator_sync = True
        self.enable_simulator_step = False

    def simulator_step(self):
        """Step the simulator and sync the simulator to renderer"""
        self.env.simulator.step(render=True)

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
        state = self.env.dump_state(serialized=False)
        x, y, theta = goal
        print(f"goal: {x},{y},{theta}")

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

            collision_free = self.env.test_valid_position(
                obj=self.robot,
                pos=np.array((x, y, self.trav_map.floor_heights[self.floor_num])),
                ori=np.array([0, 0, theta]),
            )
            # collision_free = True

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
                    robot_position_xy + rotate_vector_2d(local_corner, -yaw)
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

            path = plan_base_motion_2d(
                self.robot_body_id,
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
            )

        if path is not None and len(path) > 0:
            log.debug("Path found!")
        else:
            log.debug("Path NOT found!")

        # Restore original state
        self.env.load_state(state=state, serialized=False)
        # app.update()

        return path

    def visualize_base_path(self, path, keep_last_location=True):
        """
        Dry run base motion plan by setting the base positions without physics simulation

        :param path: base waypoints or None if no plan can be found
        """

        if path is not None:
            # If we are not keeping the last location, se save the state to reload it after the visualization
            if not keep_last_location:
                initial_state = self.env.dump_state(serialized=False)

            grasping_object = self.robot.is_grasping() == IsGraspingState.TRUE
            grasped_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]

            if grasping_object:
                gripper_pos = self.robot.get_eef_position(arm="default")
                gripper_orn = self.robot.get_eef_orientation(arm="default")
                obj_pos, obj_orn = grasped_obj.get_position_orientation()
                grasp_pose = T.relative_pose_transform(gripper_pos, gripper_orn, obj_pos, obj_orn)

            if not m.HEADLESS:
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
                        grasped_obj.set_position_orientation(*object_pose)
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
                    grasped_obj.set_position_orientation(*object_pose)

            if not keep_last_location:
                log.info("Not keeping the last state, only visualizing the path and restoring at the end")
                self.env.load_state(state=initial_state, serialized=False)
                # app.update()

    def get_ik_parameters(self, arm="default"):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        arm = self.robot.default_arm if arm == "default" else arm
        rest_position = None
        if self.robot_type == "Fetch":
            rest_position = self.robot.get_joint_positions()[self.ik_control_idx[arm]]
        elif self.robot_type == "Tiago":
            rest_position = self.robot.get_joint_positions()[self.ik_control_idx[arm]]
        elif self.robot_type == "BehaviorRobot":
            log.warning("Not implemented!")
        else:
            log.warning("Robot type is not compatible with IK for motion planning")
            raise ValueError

        return rest_position

    def get_joint_pose_for_ee_pose_with_ik(
        self,
        ee_position,
        ee_orientation=None,
        arm=None,
        check_collisions=True,
        randomize_initial_pose=True,
        obj_name=None,
    ):
        """
        Attempt to find arm_joint_pose that satisfies ee_position (and possibly, ee_orientation)
        If failed, return None

        :param ee_position: desired position of the end-effector [x, y, z] in the world frame
        :param ee_orientation: desired orientation of the end-effector [x,y,z,w] in the world frame
        :param arm: string of the name of the arm to use. Use default arm if None
        :param check_collisions: whether we check for collisions in the solution or not
        :return: arm joint_pose
        """
        log.debug("IK query for end-effector pose ({}, {}) with arm {}".format(ee_position, ee_orientation, arm))

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Defaulting to get IK for the default arm: {}".format(arm))

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
            if l2_distance(ee_position, position_arm_shoulder_in_wf) > 0.7:  # TODO: get max distance
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

        # arm_joint_pb_ids = np.array(joints_from_names(self.robot_body_id, self.robot.arm_joint_names[arm]))
        # sample_fn = get_sample_fn(self.robot_body_id, arm_joint_pb_ids)
        # base_pose = get_base_values(self.robot_body_id)
        # initial_pb_state = p.saveState()

        # self.robot.keep_still()
        # self.simulator_step()
        # self.simulator_step()
        # self.simulator_step()

        # jnt_state = self.robot.get_joints_state()

        state = self.env.dump_state(serialized=False)
        # self.simulator_step()
        # self.simulator_step()
        # self.env.load_state(state=state, serialized=False)
        # self.simulator_step()
        # self.simulator_step()
        # for i in range(100):
        #     self.simulator_step()
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

        # Find the local transform
        ee_local_pos, ee_local_ori = T.relative_pose_transform(ee_position, ee_orientation, base_pos, base_quat)
        ee_local_ori = np.array([0,0,0,1])

        ee_local_pos = ee_local_pos

        print(f"attempting ik at: {ee_local_pos}, {ee_local_ori}")

        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:

            if randomize_initial_pose:
                # Start the iterative IK from a different random initial joint pose
                #sample = sample_fn()
                rest_position = np.random.rand(n_control_idx) * jnt_range + jnt_low

            # Calculate IK solution
            control_joint_pos = self.ik_solver[arm].solve(
                target_pos=ee_local_pos,
                target_quat=ee_local_ori,
                initial_joint_pos=rest_position,
            )

            # Set the joint positions
            current_joint_pos[control_idx] = control_joint_pos
            self.robot.set_joint_positions(current_joint_pos)
            # app.update()

            dist = l2_distance(self.robot.get_eef_position(arm=arm), ee_position)
            if dist > self.arm_ik_threshold:
                # input(f"Distance from pose: {dist}, max: {self.arm_ik_threshold}")
                log.warning("IK solution is not close enough to the desired pose. Distance: {}".format(dist))
                n_attempt += 1
                continue

            if check_collisions:
                # simulator_step will slightly move the robot base and the objects
                self.robot.set_position(base_pos_col_check)
                # self.reset_object_states()
                # TODO: have a principled way for stashing and resetting object states
                # arm should not have any collision
                collision_free = not self.env.check_collision(linksA=self.robot.arm_links[arm], step_sim=True)

                if not collision_free:
                    n_attempt += 1
                    log.warning("IK solution brings the arm into collision")
                    continue

                # gripper should not have any self-collision
                collision_free = not self.env.check_collision(
                    linksA=[self.robot.eef_links[arm]] + self.robot.finger_links[arm],
                    objsB=self.robot,
                    step_sim=False,
                )
                if not collision_free:
                    n_attempt += 1
                    log.warning("IK solution brings the gripper into collision")
                    continue

            # self.episode_metrics['arm_ik_time'] += time() - ik_start
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

            # Restore state
            self.env.load_state(state=state, serialized=False)
            # app.update()
            log.debug("IK Solver found a valid configuration")
            return control_joint_pos

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        self.env.load_state(state=state, serialized=False)
        # app.update()
        # self.episode_metrics['arm_ik_time'] += time() - ik_start
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
        if self.fine_motion_plan:
            check_self_collisions = True
            if self.robot_type != "BehaviorRobot":  # For any other robot, remove the collisions from the list
                disabled_self_colliding_pairs = {
                    (link_from_name(self.robot_body_id, link1), link_from_name(self.robot_body_id, link2))
                    for (link1, link2) in self.robot.disabled_collision_pairs
                }
            mp_obstacles = self.mp_obstacles
        else:
            check_self_collisions = False
            mp_obstacles = []

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        initial_pb_state = p.saveState()

        if initial_arm_configuration is not None:
            log.warning(
                "Start the planning trajectory from a given joint configuration is not implemented yet."
                "Should it use the utils function or the iG function to move the arm? What indices?"
            )
            exit(-1)
        else:
            # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
            self.robot.untuck()

        disabled_colliding_links = []
        if disable_collision_hand_links:
            disabled_colliding_links = [self.robot.eef_links[arm].link_id] + [
                finger.link_id for finger in self.robot.finger_links[arm]
            ]

        if self.robot_type != "BehaviorRobot":
            arm_joint_pb_ids = np.array(joints_from_names(self.robot_body_id, self.robot.arm_joint_names[arm]))
            arm_path = plan_joint_motion(
                self.robot_body_id,
                arm_joint_pb_ids,
                arm_joint_pose,
                disabled_colliding_links=disabled_colliding_links,
                check_self_collisions=check_self_collisions,
                disabled_self_colliding_pairs=disabled_self_colliding_pairs,
                obstacles=mp_obstacles,
                algorithm=self.arm_mp_algo,
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
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(initial_pb_state)
        p.removeState(initial_pb_state)

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
        log.warn("Planning arm motion to end-effector pose ({}, {})".format(ee_position, ee_orientation))
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
            log.warn("Planning straight line for the default arm: {}".format(arm))

        if line_length == 0.0:
            log.warning("Requested line of length 0. Returning a path with only one configuration: initial_arm_pose")
            return [initial_arm_pose]

        state = self.env.dump_state(serialized=False)

        # Start planning from the given pose
        if self.robot_type != "BehaviorRobot":
            joint_pos = self.robot.get_joint_positions()
            joint_pos[self.ik_control_idx[arm]] = initial_arm_pose
            self.robot.set_joint_positions(joint_pos)
            # app.update()
            # app.update()
            # self.simulator_sync()
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
            start_joint_pose = time.time()
            # print(i, 'start joint pose time {}'.format(start_joint_pose - start_push_goal))
            # Solve the IK problem to set the arm at the desired position
            joint_pose = self.get_joint_pose_for_ee_pose_with_ik(
                push_goal, ee_orientation=ee_orn, arm=arm, check_collisions=False, randomize_initial_pose=False
            )
            start_restore = time.time()
            # print('start restore {}'.format(start_restore-start_joint_pose))
            if joint_pose is None:
                self.env.load_state(state=state, serialized=False)
                # app.update()
                log.warning("Failed to retrieve IK solution for EE line path. Failure.")
                return None

            line_path.append(joint_pose)
            end_restore = time.time()
            # print('end restore {}'.format(end_restore - start_restore))
        self.env.load_state(state=state, serialized=False)
        # app.update()
        return line_path

    # def plan_ee_push(
    #     self,
    #     pushing_location,
    #     pushing_direction,
    #     ee_pushing_orn=None,
    #     pre_pushing_distance=0.1,
    #     pushing_distance=0.1,
    #     pushing_steps=50,
    #     plan_full_pre_push_motion=True,
    #     arm=None,
    # ):
    #     """
    #     Plans a full pushing trajectory.
    #     The trajectory includes a pre-pushing motion in free space (contact-free) to a location in front of the pushing
    #     location, and a pushing motion (with collisions, to interact) in a straight line in Cartesian space along the
    #     pushing direction
    #
    #     :param pushing_location: 3D point to push at
    #     :param pushing_direction: direction to push after reaching that point
    #     :param ee_pushing_orn: orientation of the end effector during pushing [x,y,z,w]. None if we do not constraint it
    #     :param pre_pushing_distance: distance in front of the pushing point along the pushing direction to plan a motion
    #         in free space before the interaction
    #     :param pushing_distance: distance after the pushing point along the pushing direction to move in the interaction
    #         push. The total motion in the second phase will be pre_pushing_distance + pushing_distance
    #     :param pushing_steps: steps to compute IK along a straight line for the second phase of the push (interaction)
    #     :param arm: which arm to use for multi-arm agents. `None` to use the default arm
    #     :return: tuple of (pre_push_path, push_interaction_path) with the joint space trajectory to follow in the two
    #         phases of the interaction. Both will be None if the planner fails to find a path
    #     """
    #     log.warning(
    #         "Planning end-effector pushing action at point {} with direction {}".format(
    #             pushing_location, pushing_direction
    #         )
    #     )
    #     if self.marker is not None:
    #         self.set_marker_position_direction(pushing_location, pushing_direction)
    #
    #     if arm is None:
    #         arm = self.robot.default_arm
    #         log.warning("Pushing with the default arm: {}".format(arm))
    #
    #     pre_pushing_location = pushing_location - pre_pushing_distance * pushing_direction
    #     log.warning(
    #         "It will plan a motion to the location {}, {} m in front of the pushing location"
    #         "".format(pre_pushing_location, pre_pushing_distance)
    #     )
    #
    #     if plan_full_pre_push_motion:
    #         pre_push_path = self.plan_ee_motion_to_cartesian_pose(
    #             pre_pushing_location, ee_orientation=ee_pushing_orn, arm=arm
    #         )
    #     else:
    #         log.warning("Not planning the pre-push path, only checking feasibility of the last location.")
    #         last_pose = self.get_joint_pose_for_ee_pose_with_ik(
    #             pre_pushing_location, ee_orientation=ee_pushing_orn, arm=arm
    #         )
    #         pre_push_path = [last_pose] if last_pose is not None else []
    #
    #     push_interaction_path = None
    #
    #     if pre_push_path is None or len(pre_push_path) == 0:
    #         log.warning("Planning failed: no path found to pre-pushing location")
    #     else:
    #         push_interaction_path = self.plan_ee_straight_line_motion(
    #             pre_push_path[-1],
    #             pushing_location
    #             - pushing_direction * pre_pushing_distance,  # This is where the pre-pushing path should end
    #             pushing_direction,
    #             ee_orn=ee_pushing_orn,
    #             line_length=pre_pushing_distance + pushing_distance,
    #             steps=pushing_steps,
    #             arm=arm,
    #         )
    #         if push_interaction_path is None:
    #             log.warn("Planning failed: no path found to push the object")
    #
    #     return pre_push_path, push_interaction_path

    # def plan_ee_pull(
    #     self,
    #     pulling_location,
    #     pulling_direction,
    #     ee_pulling_orn=None,
    #     pre_pulling_distance=0.1,
    #     pulling_distance=0.1,
    #     pulling_steps=50,
    #     plan_full_pre_pull_motion=True,
    #     arm=None,
    # ):
    #     """
    #     Plans a full pulling trajectory.
    #     The trajectory includes a pre-pulling motion in free space (contact-free) to a location in front of the pulling
    #     location, and a pulling motion (with collisions, to pull) in a straight line in Cartesian space along the
    #     pulling direction
    #
    #     :param pulling_location: 3D point to push at
    #     :param pulling_direction: direction to push after reaching that point
    #     :param pre_pulling_distance: distance in front of the pulling point in the opposite direction to the pulling
    #         direction to plan a motion in free space before the interaction pull.
    #     :param pulling_distance: distance after the pulling point along the pulling direction to move in the interaction
    #         push.
    #     :param pulling_steps: steps to compute IK
    #     :param arm: which arm to use for multi-arm agents. `None` to use the default arm
    #     :return: tuple of (pre_pull_path, pull_interaction_path1, pull_interaction_path2) with the joint space
    #         trajectory to follow in the three phases of the interaction. All will be None if the planner fails to find a
    #         path
    #     """
    #     log.warning(
    #         "Planning end-effector pulling action at point {} with direction {}".format(
    #             pulling_location, pulling_direction
    #         )
    #     )
    #     if self.marker is not None:
    #         self.set_marker_position_direction(pulling_location, pulling_direction)
    #
    #     if arm is None:
    #         arm = self.robot.default_arm
    #         log.warning("Pulling with the default arm: {}".format(arm))
    #
    #     pre_pulling_location = pulling_location + pre_pulling_distance * pulling_direction
    #     log.warning(
    #         "It will plan a motion to the location {}, {} m in front of the pulling location"
    #         "".format(pre_pulling_location, pre_pulling_distance)
    #     )
    #
    #     if ee_pulling_orn is None:
    #         # Compute orientation of the hand to align the fingers to the opposite of the pulling direction and the palm up
    #         # to the ceiling
    #         # Assuming fingers point in +x, palm points in +z in eef
    #         # In world frame, +z points up
    #         desired_x_dir_normalized = -np.array(pulling_direction) / np.linalg.norm(np.array(pulling_direction))
    #         z_dir_in_wf = np.array([0, 0, 1.0])
    #         desired_y_dir = -np.cross(desired_x_dir_normalized, z_dir_in_wf)
    #         desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
    #         desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
    #         rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
    #         quatt = quatXYZWFromRotMat(rot_matrix)
    #     else:
    #         log.warning("Planning pulling with end-effector orientation {}".format(ee_pulling_orn))
    #         quatt = ee_pulling_orn
    #
    #     if plan_full_pre_pull_motion:
    #         pre_pull_path = self.plan_ee_motion_to_cartesian_pose(pre_pulling_location, ee_orientation=quatt, arm=arm)
    #     else:
    #         log.warning("Not planning the pre-pull path, only checking feasibility of the last location.")
    #         last_pose = self.get_joint_pose_for_ee_pose_with_ik(pre_pulling_location, ee_orientation=quatt, arm=arm)
    #         pre_pull_path = [last_pose] if last_pose is not None else []
    #
    #     approach_interaction_path = None
    #     pull_interaction_path = None
    #
    #     if pre_pull_path is None or len(pre_pull_path) == 0:
    #         log.warning("Planning failed: no path found to pre-pulling location")
    #     else:
    #         approach_interaction_path = self.plan_ee_straight_line_motion(
    #             pre_pull_path[-1],  # This is where the pre-pulling path should end
    #             pre_pulling_location,
    #             -pulling_direction,
    #             ee_orn=quatt,
    #             line_length=pre_pulling_distance,
    #             steps=pulling_steps,
    #             arm=arm,
    #         )
    #         if approach_interaction_path is None:
    #             log.warn("Planning failed: no path found to approach the object to pull")
    #         else:
    #             pull_interaction_path = self.plan_ee_straight_line_motion(
    #                 approach_interaction_path[-1],
    #                 pre_pulling_location - pulling_direction * pre_pulling_distance,
    #                 pulling_direction,
    #                 ee_orn=quatt,
    #                 line_length=pulling_distance,
    #                 steps=pulling_steps,
    #                 arm=arm,
    #             )
    #
    #     return pre_pull_path, approach_interaction_path, pull_interaction_path
    # '''
    # def plan_arm_pull(self, hit_pos, hit_normal):
    #     """
    #     Attempt to reach a 3D position and prepare for a push later
    #
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     :return: arm trajectory or None if no plan can be found
    #     """
    #     log.debug("Planning arm push at point {} with direction {}".format(hit_pos, hit_normal))
    #
    #     hit_pos = hit_pos + (np.array(hit_normal) * -1.0) * (self.arm_interaction_length * 0.3)
    #
    #     # if self.marker is not None:
    #     #     self.set_marker_position_direction(hit_pos, hit_normal)
    #     #
    #     # # Solve the IK problem to set the arm at the desired position
    #     # joint_positions = self.get_joint_pose_for_ee_pose_with_ik(hit_pos)
    #
    #     if joint_positions is not None:
    #         # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         plan = self.plan_base_motion([hit_pos[0], hit_pos[1], 0.05])
    #         return plan
    #     else:
    #         log.debug("Planning failed: goal position may be non-reachable")
    #         return None
    #
    # def interact_pull(self, push_point, push_direction):
    #     """
    #     Move the arm starting from the push_point along the push_direction
    #     and physically simulate the interaction
    #
    #     :param push_point: 3D point to start pushing from
    #     :param push_direction: push direction
    #     """
    #     body_ids = self.robots.get_body_ids()
    #     assert len(body_ids) == 1, "Only single-body robots are supported."
    #     self.robot_id = body_ids[0]
    #     self.arm_default_joint_positions = (
    #         0.1,
    #         -1.41,
    #         1.517,
    #         0.82,
    #         2.2,
    #         2.96,
    #         -1.286,
    #         0.0,
    #     )
    #     self.arm_joint_names = [
    #         "torso_lift_joint",
    #         "shoulder_pan_joint",
    #         "shoulder_lift_joint",
    #         "upperarm_roll_joint",
    #         "elbow_flex_joint",
    #         "forearm_roll_joint",
    #         "wrist_flex_joint",
    #         "wrist_roll_joint",
    #     ]
    #     self.robot_joint_names = [
    #         "r_wheel_joint",
    #         "l_wheel_joint",
    #         "torso_lift_joint",
    #         "head_pan_joint",
    #         "head_tilt_joint",
    #         "shoulder_pan_joint",
    #         "shoulder_lift_joint",
    #         "upperarm_roll_joint",
    #         "elbow_flex_joint",
    #         "forearm_roll_joint",
    #         "wrist_flex_joint",
    #         "wrist_roll_joint",
    #         "r_gripper_finger_joint",
    #         "l_gripper_finger_joint",
    #     ]
    #     self.rest_joint_names = [
    #         "r_wheel_joint",
    #         "l_wheel_joint",
    #         "head_pan_joint",
    #         "head_tilt_joint",
    #         "r_gripper_finger_joint",
    #         "l_gripper_finger_joint",
    #     ]
    #
    #     self.arm_joint_ids = joints_from_names(
    #         self.robot_id,
    #         self.arm_joint_names,
    #     )
    #
    #     self.robot_arm_indices = [
    #         self.robot_joint_names.index(arm_joint_name) for arm_joint_name in self.arm_joint_names
    #     ]
    #
    #     self.rest_joint_ids = joints_from_names(
    #         self.robot_id,
    #         self.rest_joint_names,
    #     )
    #
    #     push_vector = np.array(push_direction) * self.arm_interaction_length
    #
    #     max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
    #     base_pose = get_base_values(self.robot_id)
    #
    #     # if self.sleep_flag:
    #     #     steps = 50
    #     # else:
    #     steps = 5
    #
    #     for i in range(steps):
    #         push_goal = np.array(push_point) + push_vector * (i + 1) / float(steps)
    #
    #         joint_positions = p.calculateInverseKinematics(
    #             self.robot_id,
    #             self.robot.eef_links[self.robot.default_arm].link_id,
    #             targetPosition=push_goal,
    #             # targetOrientation=(0, 0.707, 0, 0.707),
    #             lowerLimits=min_limits,
    #             upperLimits=max_limits,
    #             jointRanges=joint_range,
    #             restPoses=rest_position,
    #             jointDamping=joint_damping,
    #             # solver=p.IK_DLS,
    #             maxNumIterations=100,
    #         )
    #
    #         if self.robot_type == "Fetch":
    #             joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         control_joints(self.robot_id, self.arm_joint_ids, joint_positions)
    #         control_joints(self.robot_id, self.rest_joint_ids, np.array([0.0, 0.0, 0.0, 0.45, 0.5, 0.5]))
    #
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
    #
    #         # if self.robot_type == "Fetch":
    #         #     joint_positions = np.array(joint_positions)[self.robot_arm_indices]
    #
    #         achieved = self.robot.get_eef_position()
    #         self.simulator_step()
    #         set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
    #
    #         if self.mode == "gui_interactive" and self.sleep_flag:
    #             sleep(0.02)  # for visualization
    #
    # def execute_arm_pull(self, plan, hit_pos, hit_normal):
    #     """
    #     Execute arm push given arm trajectory
    #     Should be called after plan_arm_push()
    #
    #     :param plan: arm trajectory or None if no plan can be found
    #     :param hit_pos: 3D position to reach
    #     :param hit_normal: direction to push after reacehing that position
    #     """
    #     if plan is not None:
    #         log.debug("Teleporting arm along the trajectory. No physics simulation")
    #         self.visualize_base_path(plan)
    #         log.debug("Performing pushing actions")
    #         self.interact_pull(hit_pos, hit_normal)
    #         log.debug("Teleporting arm to the default configuration")
    #         self.visualize_base_path(plan[::-1])
    #         # set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
    #         self.simulator_sync()
    #         done, _ = self.env.task.get_termination(self.env)
    #         if self.print_log:
    #             print("pull done: ", done)
    # '''

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
        log.warning(
            "Planning end-effector grasping action at point {} with direction {}".format(
                grasping_location, grasping_direction
            )
        )
        if self.marker is not None:
            self.set_marker_position_direction(grasping_location, grasping_direction)

        if arm is None:
            arm = self.robot.default_arm
            log.warning("Grasping with the default arm: {}".format(arm))

        pre_grasping_location = grasping_location - pre_grasping_distance * grasping_direction
        log.warning(
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
            log.warning("Approaching grasping location top-down")
            desired_y_dir_normalized = np.array([0.0, 1.0, 0.0])
        else:
            desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
        desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
        rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
        quatt = T.mat2quat(rot_matrix)

        if plan_full_pre_grasp_motion:
            pre_grasp_path = self.plan_ee_motion_to_cartesian_pose(pre_grasping_location, ee_orientation=quatt, arm=arm)
        else:
            log.warning("Not planning the pre-grasp path, only checking feasibility of the last location.")
            last_pose = self.get_joint_pose_for_ee_pose_with_ik(pre_grasping_location, ee_orientation=quatt, arm=arm)
            pre_grasp_path = [last_pose] if last_pose is not None else []

        grasp_interaction_path = None

        if pre_grasp_path is None or len(pre_grasp_path) == 0:
            log.warning("Planning failed: no path found to pre-grasping location")
        elif pre_grasping_distance == 0:
            log.debug("Skipping computation of interaction path because the pre-grasping distance is zero")
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
                log.warn("Planning failed: no path found to grasp the object")

        return pre_grasp_path, grasp_interaction_path

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

    # def plan_ee_toggle(
    #     self,
    #     toggling_location,
    #     toggling_direction,
    #     pre_toggling_distance=0.1,
    #     toggling_steps=50,
    #     plan_full_pre_toggle_motion=True,
    #     arm=None,
    # ):
    #     """
    #     Plans a full toggling trajectory.
    #     The trajectory includes a pre-toggling motion in free space (contact-free) to a location in front of the toggling
    #     location, and a toggling motion (with collisions, to toggle) in a straight line in Cartesian space along the
    #     toggling direction
    #
    #     :param toggling_location: 3D point to push at
    #     :param toggling_direction: direction to push after reaching that point
    #     :param pre_toggling_distance: distance in front of the toggling point along the toggling direction to plan a motion
    #         in free space before the interaction toggle. The total motion in the second phase will be pre_toggling_distance + toggling_distance
    #     :param toggling_steps: steps to compute IK along a straight line for the second phase of the toggle (interaction)
    #     :param arm: which arm to use for multi-arm agents. `None` to use the default arm
    #     :return: tuple of (pre_toggle_path, toggle_interaction_path) with the joint space trajectory to follow in the two
    #     phases of the interaction. Both will be None if the planner fails to find a path
    #     """
    #     log.warning(
    #         "Planning end-effector toggling action at point {} with direction {}".format(
    #             toggling_location, toggling_direction
    #         )
    #     )
    #     if self.marker is not None:
    #         self.set_marker_position_direction(toggling_location, toggling_direction)
    #
    #     if arm is None:
    #         arm = self.robot.default_arm
    #         log.warning("toggling with the default arm: {}".format(arm))
    #
    #     pre_toggling_location = toggling_location - pre_toggling_distance * toggling_direction
    #     log.warning(
    #         "It will plan a motion to the location {}, {} m in front of the toggling location"
    #         "".format(pre_toggling_distance, pre_toggling_location)
    #     )
    #     """
    #     # Compute orientation of the hand to align the fingers to the toggling direction and the palm up to the ceiling
    #     # Assuming fingers point in +x, palm points in +z in eef
    #     # In world frame, +z points up
    #     desired_x_dir_normalized = np.array(toggling_direction) / np.linalg.norm(np.array(toggling_direction))
    #     z_dir_in_wf = np.array([0, 0, 1.0])
    #     desired_y_dir = -np.cross(desired_x_dir_normalized, z_dir_in_wf)
    #     desired_y_dir_normalized = desired_y_dir / np.linalg.norm(desired_y_dir)
    #     desired_z_dir_normalized = np.cross(desired_x_dir_normalized, desired_y_dir_normalized)
    #     rot_matrix = np.column_stack((desired_x_dir_normalized, desired_y_dir_normalized, desired_z_dir_normalized))
    #     quatt = quatXYZWFromRotMat(rot_matrix)
    #     """
    #     quatt = (0, 0.7071068, 0, 0.7071068)
    #     if plan_full_pre_toggle_motion:
    #         pre_toggle_path = self.plan_ee_motion_to_cartesian_pose(
    #             pre_toggling_location, ee_orientation=quatt, arm=arm
    #         )
    #     else:
    #         log.warning("Not planning the pre-toggle path, only checking feasibility of the last location.")
    #         last_pose = self.get_joint_pose_for_ee_pose_with_ik(pre_toggling_location, ee_orientation=quatt, arm=arm)
    #         pre_toggle_path = [last_pose] if last_pose is not None else []
    #
    #     toggle_interaction_path = None
    #
    #     if pre_toggle_path is None or len(pre_toggle_path) == 0:
    #         log.warning("Planning failed: no path found to pre-toggling location")
    #     else:
    #         toggle_interaction_path = self.plan_ee_straight_line_motion(
    #             pre_toggle_path[-1],
    #             toggling_location
    #             - toggling_direction * pre_toggling_distance,  # This is where the pre-toggling path should end
    #             toggling_direction,
    #             ee_orn=quatt,
    #             line_length=pre_toggling_distance,
    #             steps=toggling_steps,
    #             arm=arm,
    #         )
    #         if toggle_interaction_path is None:
    #             log.warn("Planning failed: no path found to toggle the object")
    #
    #     return pre_toggle_path, toggle_interaction_path

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
            log.warn("Visualizing arm path for the default arm: {}".format(arm))

        if not keep_last_location:
            state = self.env.dump_state(serialized=False)

        if grasped_obj is not None:
            if self.robot_type != "BehaviorRobot":
                # Get global EEF pose
                grasp_pose = (self.robot.get_eef_position(arm=arm), self.robot.get_eef_orientation(arm=arm))
            else:
                # TODO
                gripper_pos = self.robot.get_eef_position(arm)
                gripper_orn = self.robot.get_eef_orientation(arm)
                obj_pos, obj_orn = p.getBasePositionAndOrientation(grasped_obj_id)
                grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        # base_pose = get_base_values(self.robot_body_id)
        execution_path = arm_path if not reverse_path else reversed(arm_path)
        execution_path = (
            execution_path if not m.HEADLESS else [execution_path[-1]]
        )
        if self.robot_type != "BehaviorRobot":
            for joint_way_point in execution_path:
                self.robot.set_joint_positions(joint_way_point, indices=self.ik_control_idx[arm])

                # Teleport also the grasped object
                if grasped_obj is not None:
                    grasp_pose = (self.robot.get_eef_position(arm=arm), self.robot.get_eef_orientation(arm=arm))
                    grasped_obj.set_position_orientation(*grasp_pose)

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
            self.env.load_state(state=state, serialized=False)

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


# def plan_hand_motion_br(
#     robot,
#     arm,
#     end_conf,
#     obstacles=[],
#     attachments=[],
#     direct_path=False,
#     max_distance=HAND_MAX_DISTANCE,
#     iterations=50,
#     restarts=2,
#     shortening=0,
#     algorithm="birrt",
#     disabled_colliding_links=[],
#     check_self_collisions=True,
#     disabled_self_colliding_pairs=set(),
#     step_resolutions=(0.03, 0.03, 0.03, 0.2, 0.2, 0.2),
# ):
#     if algorithm != "birrt":
#         print("We only allow birrt with the BehaviorRobot")
#         exit(-1)
#
#     # Define the sampling domain.
#     cur_pos = np.array(robot.get_position())
#     target_pos = np.array(end_conf[:3])
#     both_pos = np.array([cur_pos, target_pos])
#     HAND_SAMPLING_DOMAIN_PADDING = 1  # Allow 1m of freedom around the sampling range.
#     min_pos = np.min(both_pos, axis=0) - HAND_SAMPLING_DOMAIN_PADDING
#     max_pos = np.max(both_pos, axis=0) + HAND_SAMPLING_DOMAIN_PADDING
#
#     hand_limits = (min_pos, max_pos)
#
#     obj_in_hand = robot._ag_obj_in_hand[arm]
#
#     hand_distance_fn, sample_fn, extend_fn, collision_fn = get_brobot_hand_planning_fns(
#         robot, arm, hand_limits, obj_in_hand, obstacles, step_resolutions, max_distance=max_distance
#     )
#
#     # Get the initial configuration of the selected hand
#     pos, orn = robot.eef_links[arm].get_position_orientation()
#     rpy = p.getEulerFromQuaternion(orn)
#     start_conf = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]
#
#     if collision_fn(start_conf):
#         log.warning("Warning: initial configuration is in collision. Impossible to find a path.")
#         return None
#     else:
#         log.debug("Initial conf is collision free")
#
#     if collision_fn(end_conf):
#         log.warning("Warning: end configuration is in collision. Impossible to find a path.")
#         return None
#     else:
#         log.debug("Final conf is collision free")
#
#     if direct_path:
#         log.warning("Planning direct path")
#         rpy = p.getEulerFromQuaternion(end_conf[3:])
#         end_conf = [end_conf[0], end_conf[1], end_conf[2], rpy[0], rpy[1], rpy[2]]
#         return direct_path(start_conf, end_conf, extend_fn, collision_fn)
#
#     # TODO: Deal with obstacles and attachments
#
#     path = birrt(
#         start_conf,
#         end_conf,
#         hand_distance_fn,
#         sample_fn,
#         extend_fn,
#         collision_fn,
#         iterations=iterations,
#         restarts=restarts,
#     )
#
#     if path is None:
#         return None
#     return shorten_path(path, extend_fn, collision_fn, shortening)


# def get_brobot_hand_planning_fns(
#     robot, arm, hand_limits, obj_in_hand, obstacles, step_resolutions, max_distance=HAND_MAX_DISTANCE
# ):
#     """
#     Define the functions necessary to do motion planning with a floating hand for the BehaviorRobot:
#     distance function, sampling function, extend function and collision checking function
#
#     """
#
#     def hand_difference_fn(q2, q1):
#         # Metric distance in location
#         dx, dy, dz = np.array(q2[:3]) - np.array(q1[:3])
#         # Per-element distance in orientation
#         # TODO: the orientation distance should be the angle in the axis-angle representation of the difference of
#         # orientations between the poses (1 value)
#         droll = circular_difference(q2[3], q1[3])
#         dpitch = circular_difference(q2[4], q1[4])
#         dyaw = circular_difference(q2[5], q1[5])
#
#         return np.array((dx, dy, dz, droll, dpitch, dyaw))
#
#     def hand_distance_fn(q1, q2, weights=(1, 1, 1, 5, 5, 5)):
#         """
#         Function to calculate the distance between two poses of a hand
#         :param q1: Initial pose of the hand
#         :param q2: Goal pose of the hand
#         :param weights: Weights of each of the six elements of the pose difference (3 position and 3 orientation)
#         """
#         difference = hand_difference_fn(q1, q2)
#         return np.sqrt(np.dot(np.array(weights), difference * difference))
#
#     def sample_fn():
#         """
#         Sample a random hand configuration (6D pose) within limits
#         """
#         x, y, z = np.random.uniform(*hand_limits)
#         r, p, yaw = np.random.uniform((-PI, -PI, -PI), (PI, PI, PI))
#         return (x, y, z, r, p, yaw)
#
#     def extend_fn(q1, q2):
#         """
#         Extend function for sampling-based planning
#         It interpolates between two 6D pose configurations linearly
#         :param q1: initial configuration
#         :param q2: final configuration
#         """
#         # TODO: Use scipy's slerp
#         steps = np.abs(np.divide(hand_difference_fn(q2, q1), step_resolutions))
#         n = int(np.max(steps)) + 1
#
#         for i in range(n):
#             delta = hand_difference_fn(q2, q1)
#             delta_ahora = ((i + 1) / float(n)) * np.array(delta)
#             q = delta_ahora + np.array(q1)
#             q = tuple(q)
#             yield q
#
#     non_hand_non_oih_obstacles = {
#         obs
#         for obs in obstacles
#         if ((obj_in_hand is None or obs not in obj_in_hand.get_body_ids()) and (obs != robot.eef_links[arm].body_id))
#     }
#
#     def collision_fn(pose3d):
#         quat = pose3d[3:] if len(pose3d[3:]) == 4 else p.getQuaternionFromEuler(pose3d[3:])
#         pose3dq = (pose3d[:3], quat)
#         robot.set_eef_position_orientation(pose3d[:3], quat, arm)
#         close_objects = set(
#             x[0]
#             for x in p.getOverlappingObjects(*get_aabb(robot.eef_links[arm].body_id))
#             if x[0] != robot.eef_links[arm].body_id
#         )
#         close_obstacles = close_objects & non_hand_non_oih_obstacles
#         collisions = [
#             (obs, pairwise_collision(robot.eef_links[arm].body_id, obs, max_distance=max_distance))
#             for obs in close_obstacles
#         ]
#         colliding_bids = [obs for obs, col in collisions if col]
#         if colliding_bids:
#             log.debug("Hand collision with objects: ", colliding_bids)
#         collision = bool(colliding_bids)
#
#         if obj_in_hand is not None:
#             # Generalize more.
#             [oih_bid] = obj_in_hand.get_body_ids()  # Generalize.
#             oih_close_objects = set(x[0] for x in p.getOverlappingObjects(*get_aabb(oih_bid)))
#             oih_close_obstacles = (oih_close_objects & non_hand_non_oih_obstacles) | close_obstacles
#             obj_collisions = [
#                 (obs, pairwise_collision(oih_bid, obs, max_distance=max_distance)) for obs in oih_close_obstacles
#             ]
#             obj_colliding_bids = [obs for obs, col in obj_collisions if col]
#             if obj_colliding_bids:
#                 log.debug("Held object collision with objects: ", obj_colliding_bids)
#             collision = collision or bool(obj_colliding_bids)
#
#         return collision
#
#     return hand_distance_fn, sample_fn, extend_fn, collision_fn


# def shorten_path(path, extend, collision, iterations=50):
#     shortened_path = path
#     for _ in range(iterations):
#         if len(shortened_path) <= 2:
#             return shortened_path
#         i = random.randint(0, len(shortened_path) - 1)
#         j = random.randint(0, len(shortened_path) - 1)
#         if abs(i - j) <= 1:
#             continue
#         if j < i:
#             i, j = j, i
#         shortcut = list(extend(shortened_path[i], shortened_path[j]))
#         if all(not collision(q) for q in shortcut):
#             shortened_path = shortened_path[: i + 1] + shortened_path[j:]
#     return shortened_path
