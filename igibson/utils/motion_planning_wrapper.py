import logging
from time import sleep, time

import numpy as np

from transforms3d import euler

from igibson.external.pybullet_tools.utils import (
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
)
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.utils.utils import l2_distance, quatToXYZW, restoreState, rotate_vector_2d


class MotionPlanningWrapper(object):
    """
    Motion planner wrapper that supports both base and arm motion
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
        self.env = env
        assert "occupancy_grid" in self.env.output
        # get planning related parameters from env
        body_ids = self.env.robots[0].get_body_ids()
        assert len(body_ids) == 1, "Only single-body robots are supported."
        self.robot_id = body_ids[0]

        # Types of 2D planning
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=TRUE -> We teleport the robot to locations and check for collisions
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=FALSE -> We use the global occupancy map from the scene
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=FALSE -> We use the occupancy_grid from the lidar sensor
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=TRUE -> [not suported yet]
        self.full_observability_2d_planning = full_observability_2d_planning
        self.collision_with_pb_2d_planning = collision_with_pb_2d_planning
        assert not ((not self.full_observability_2d_planning) and self.collision_with_pb_2d_planning)

        self.robot_footprint_radius = self.env.sensors["scan_occ"].robot_footprint_radius
        if self.full_observability_2d_planning:
            # TODO: it may be better to unify and make that scene.floor_map uses OccupancyGridState values always
            assert len(self.env.scene.floor_map) == 1  # We assume there is only one floor (not true for Gibson scenes)
            self.map_2d = np.array(self.env.scene.floor_map[0])
            self.map_2d = np.array((self.map_2d == 255)).astype(np.float32)
            self.per_pixel_resolution = self.env.scene.trav_map_resolution
            assert np.array(self.map_2d).shape[0] == np.array(self.map_2d).shape[1]
            self.grid_resolution = self.map_2d.shape[0]
            self.occupancy_range = self.grid_resolution * self.per_pixel_resolution
            self.robot_footprint_radius_in_map = int(np.ceil(self.robot_footprint_radius / self.per_pixel_resolution))
        else:
            self.grid_resolution = self.env.grid_resolution
            self.occupancy_range = self.env.sensors["scan_occ"].occupancy_range
            self.robot_footprint_radius_in_map = self.env.sensors["scan_occ"].robot_footprint_radius_in_map

        self.robot = self.env.robots[0]
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
        self.mode = self.env.mode
        self.initial_height = self.env.initial_pos_z_offset
        self.fine_motion_plan = fine_motion_plan
        self.robot_type = self.robot.model_name

        if self.env.simulator.viewer is not None:
            self.env.simulator.viewer.setup_motion_planner(self)

        if self.robot_type in ["Fetch"]:
            self.setup_arm_mp()

        self.arm_interaction_length = 0.2

        self.marker = None
        self.marker_direction = None

        if self.mode in ["gui_non_interactive", "gui_interactive"]:
            self.marker = VisualMarker(radius=0.04, rgba_color=[0, 0, 1, 1])
            self.marker_direction = VisualMarker(
                visual_shape=p.GEOM_CAPSULE,
                radius=0.01,
                length=0.2,
                initial_offset=[0, 0, -0.1],
                rgba_color=[0, 0, 1, 1],
            )
            self.env.simulator.import_object(self.marker)
            self.env.simulator.import_object(self.marker_direction)

        self.visualize_2d_planning = visualize_2d_planning
        self.visualize_2d_result = visualize_2d_result

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
        quat = quatToXYZW(seq="wxyz", orn=euler.euler2quat(0, -np.pi / 2, yaw))
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

    def setup_arm_mp(self):
        """
        Set up arm motion planner
        """
        if self.robot_type == "Fetch":
            self.arm_default_joint_positions = (
                0.1,
                -1.41,
                1.517,
                0.82,
                2.2,
                2.96,
                -1.286,
                0.0,
            )
            self.arm_joint_names = [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]
            self.robot_joint_names = [
                "r_wheel_joint",
                "l_wheel_joint",
                "torso_lift_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "r_gripper_finger_joint",
                "l_gripper_finger_joint",
            ]
            self.arm_joint_ids = joints_from_names(
                self.robot_id,
                self.arm_joint_names,
            )
            self.robot_arm_indices = [
                self.robot_joint_names.index(arm_joint_name) for arm_joint_name in self.arm_joint_names
            ]

        self.arm_ik_threshold = 0.05

        self.mp_obstacles = []
        if type(self.env.scene) == StaticIndoorScene:
            if self.env.scene.mesh_body_id is not None:
                self.mp_obstacles.append(self.env.scene.mesh_body_id)
        elif type(self.env.scene) == InteractiveIndoorScene:
            self.mp_obstacles.extend(self.env.scene.get_body_ids())
            # Since the refactoring, the robot is another object in the scene
            # We need to remove it to not check twice for self collisions
            self.mp_obstacles.remove(self.robot_id)

    def plan_base_motion(self, goal):
        """
        Plan base motion given a base subgoal

        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        if self.marker is not None:
            self.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        logging.info("Motion planning base goal: {}".format(goal))

        state = self.env.get_state()
        x, y, theta = goal

        map_2d = state["occupancy_grid"] if not self.full_observability_2d_planning else self.map_2d

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
            top_left = self.env.scene.map_to_world(np.array([0, 0]))
            bottom_right = self.env.scene.map_to_world(np.array(self.map_2d.shape) - np.array([1, 1]))
            corners = [top_left, bottom_right]

        if self.collision_with_pb_2d_planning:
            obstacles = [
                body_id
                for body_id in self.env.scene.get_body_ids()
                if body_id not in self.robot.get_body_ids()
                and body_id != self.env.scene.objects_by_category["floors"][0].get_body_ids()[0]
            ]
        else:
            obstacles = []

        path = plan_base_motion_2d(
            self.robot_id,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            map_2d=map_2d,
            occupancy_range=self.occupancy_range,
            grid_resolution=self.grid_resolution,
            # If we use the global map, it has been eroded: we do not need to use the full size of the robot, a 1 px
            # robot would be enough
            robot_footprint_radius_in_map=[self.robot_footprint_radius_in_map, 1][self.full_observability_2d_planning],
            resolutions=self.base_mp_resolutions,
            # Add all objects in the scene as obstacles except the robot itself and the floor
            obstacles=obstacles,
            algorithm=self.base_mp_algo,
            optimize_iter=self.optimize_iter,
            visualize_planning=self.visualize_2d_planning,
            visualize_result=self.visualize_2d_result,
            metric2map=[None, self.env.scene.world_to_map][self.full_observability_2d_planning],
            flip_vertically=self.full_observability_2d_planning,
            use_pb_for_collisions=self.collision_with_pb_2d_planning,
        )

        if path is not None and len(path) > 0:
            logging.info("Path found!")
        else:
            logging.info("Path NOT found!")

        return path

    def simulator_sync(self):
        """Sync the simulator to renderer"""
        self.env.simulator.sync()

    def simulator_step(self):
        """Step the simulator and sync the simulator to renderer"""
        self.env.simulator.step()
        self.simulator_sync()

    def dry_run_base_plan(self, path):
        """
        Dry run base motion plan by setting the base positions without physics simulation

        :param path: base waypoints or None if no plan can be found
        """
        if path is not None:
            if self.mode in ["gui_non_interactive", "gui_interactive"]:
                for way_point in path:
                    set_base_values_with_z(
                        self.robot_id, [way_point[0], way_point[1], way_point[2]], z=self.initial_height
                    )
                    self.simulator_sync()
                    # sleep(0.005) # for animation
            else:
                set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        if self.robot_type == "Fetch":
            max_limits_arm = get_max_limits(self.robot_id, self.arm_joint_ids)
            max_limits = [0.5, 0.5] + [max_limits_arm[0]] + [0.5, 0.5] + list(max_limits_arm[1:]) + [0.05, 0.05]
            min_limits_arm = get_min_limits(self.robot_id, self.arm_joint_ids)
            min_limits = [-0.5, -0.5] + [min_limits_arm[0]] + [-0.5, -0.5] + list(min_limits_arm[1:]) + [0.0, 0.0]
            # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
            min_limits[2] += 0.02
            current_position = get_joint_positions(self.robot_id, self.arm_joint_ids)
            rest_position = [0.0, 0.0] + [current_position[0]] + [0.0, 0.0] + list(current_position[1:]) + [0.01, 0.01]
            joint_range = list(np.array(max_limits) - np.array(min_limits))
            joint_range = [item + 1 for item in joint_range]
            joint_damping = [0.1 for _ in joint_range]

        return (max_limits, min_limits, rest_position, joint_range, joint_damping)

    def get_arm_joint_positions(self, arm_ik_goal):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None

        :param arm_ik_goal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        logging.info("IK query for EE position {}".format(arm_ik_goal))
        ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)
        state_id = p.saveState()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:

            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robot.eef_links[self.robot.default_arm].link_id,
                targetPosition=arm_ik_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                # solver=p.IK_DLS,
                maxNumIterations=100,
            )

            if self.robot_type == "Fetch":
                arm_joint_positions = np.array(arm_joint_positions)[self.robot_arm_indices]

            set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(self.robot.get_eef_position(), arm_ik_goal)
            # print('dist', dist)
            if dist > self.arm_ik_threshold:
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
            # self.reset_object_states()
            # TODO: have a princpled way for stashing and resetting object states

            # arm should not have any collision
            collision_free = is_collision_free(body_a=self.robot_id, link_a_list=self.arm_joint_ids)

            if not collision_free:
                n_attempt += 1
                # print('arm has collision')
                continue

            # gripper should not have any self-collision
            collision_free = is_collision_free(
                body_a=self.robot_id,
                link_a_list=[self.robot.eef_links[self.robot.default_arm].link_id],
                body_b=self.robot_id,
            )
            if not collision_free:
                n_attempt += 1
                logging.info("Gripper in collision")
                continue

            # self.episode_metrics['arm_ik_time'] += time() - ik_start
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            restoreState(state_id)
            p.removeState(state_id)
            logging.info("IK Solver found a valid configuration")
            return arm_joint_positions

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(state_id)
        p.removeState(state_id)
        # self.episode_metrics['arm_ik_time'] += time() - ik_start
        logging.info("IK Solver failed to find a configuration")
        return None

    def plan_arm_motion(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :return: arm trajectory or None if no plan can be found
        """
        logging.info("Planning path in joint space to {}".format(arm_joint_positions))
        disabled_collisions = {}
        if self.robot_type == "Fetch":
            disabled_collisions = {
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "torso_fixed_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "shoulder_lift_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "upperarm_roll_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "forearm_roll_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "elbow_flex_link")),
            }

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles
        else:
            self_collisions = False
            mp_obstacles = []

        plan_arm_start = time()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()

        allow_collision_links = []
        if self.robot_type == "Fetch":
            allow_collision_links = [self.robot.eef_links[self.robot.default_arm].link_id] + [
                finger.link_id for finger in self.robot.finger_links[self.robot.default_arm]
            ]
        arm_path = plan_joint_motion(
            self.robot_id,
            self.arm_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=mp_obstacles,
            algorithm=self.arm_mp_algo,
            allow_collision_links=allow_collision_links,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(state_id)
        p.removeState(state_id)

        if arm_path is not None and len(arm_path) > 0:
            logging.info("Path found!")
        else:
            logging.info("Path NOT found!")

        return arm_path

    def dry_run_arm_plan(self, arm_path):
        """
        Dry run arm motion plan by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no plan can be found
        """
        base_pose = get_base_values(self.robot_id)
        if arm_path is not None:
            if self.mode in ["gui_non_interactive", "gui_interactive"]:
                for joint_way_point in arm_path:
                    set_joint_positions(self.robot_id, self.arm_joint_ids, joint_way_point)
                    set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
                    self.simulator_sync()
                    # sleep(0.02)  # animation
            else:
                set_joint_positions(self.robot_id, self.arm_joint_ids, arm_path[-1])
        else:
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)

    def plan_arm_push(self, hit_pos, hit_normal):
        """
        Attempt to reach a 3D position and prepare for a push later

        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        :return: arm trajectory or None if no plan can be found
        """
        logging.info("Planning arm push at point {} with direction {}".format(hit_pos, hit_normal))
        if self.marker is not None:
            self.set_marker_position_direction(hit_pos, hit_normal)

        # Solve the IK problem to set the arm at the desired position
        joint_positions = self.get_arm_joint_positions(hit_pos)

        if joint_positions is not None:
            # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            self.simulator_sync()
            plan = self.plan_arm_motion(joint_positions)
            return plan
        else:
            logging.info("Planning failed: goal position may be non-reachable")
            return None

    def interact(self, push_point, push_direction):
        """
        Move the arm starting from the push_point along the push_direction
        and physically simulate the interaction

        :param push_point: 3D point to start pushing from
        :param push_direction: push direction
        """
        push_vector = np.array(push_direction) * self.arm_interaction_length

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
        base_pose = get_base_values(self.robot_id)

        steps = 50
        for i in range(steps):
            push_goal = np.array(push_point) + push_vector * (i + 1) / float(steps)

            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robot.eef_links[self.robot.default_arm].link_id,
                targetPosition=push_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                # solver=p.IK_DLS,
                maxNumIterations=100,
            )

            if self.robot_type == "Fetch":
                joint_positions = np.array(joint_positions)[self.robot_arm_indices]

            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)

            # set_joint_positions(self.robot_id, self.arm_joint_ids, joint_positions)
            achieved = self.robot.get_eef_position()
            self.simulator_step()
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)

            if self.mode == "gui_interactive":
                sleep(0.02)  # for visualization

    def execute_arm_push(self, plan, hit_pos, hit_normal):
        """
        Execute arm push given arm trajectory
        Should be called after plan_arm_push()

        :param plan: arm trajectory or None if no plan can be found
        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        """
        if plan is not None:
            logging.info("Teleporting arm along the trajectory. No physics simulation")
            self.dry_run_arm_plan(plan)
            logging.info("Performing pushing actions")
            self.interact(hit_pos, hit_normal)
            logging.info("Teleporting arm to the default configuration")
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            self.simulator_sync()
