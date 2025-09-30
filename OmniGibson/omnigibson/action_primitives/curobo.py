import math
from enum import Enum
from typing import Dict, Any, Optional

import torch as th  # MUST come before importing omni!!!

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.utils.constants import JointType
from omnigibson.utils.python_utils import multi_dim_linspace


# Gives 1 - 5% better speedup, according to https://github.com/NVlabs/curobo/discussions/245#discussioncomment-9265692
th.backends.cudnn.benchmark = True
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT = 5.0  # meters
m.HOLONOMIC_BASE_REVOLUTE_JOINT_LIMIT = math.pi * 2  # radians

m.DEFAULT_COLLISION_ACTIVATION_DISTANCE = 0.005
m.DEFAULT_ATTACHED_OBJECT_SCALE = 0.8


class CuRoboEmbodimentSelection(str, Enum):
    BASE = "base"
    ARM = "arm"
    DEFAULT = "default"


def create_world_mesh_collision(tensor_args, obb_cache_size=10, mesh_cache_size=2048, max_distance=0.05):
    """
    Creates a CuRobo WorldMeshCollision to use for collision checking

    Args:
        tensor_args (TensorDeviceType): Tensor device information
        obb_cache_size (int): Cache size for number of oriented bounding boxes supported in the collision world
        mesh_cache_size (int): Cache size for number of meshes supported in the collision world
        max_distance (float): maximum distance when checking collisions (see curobo source code)

    Returns:
        MeshCollisionWorld: collision world used to check against for collisions
    """
    world_cfg = lazy.curobo.geom.sdf.world.WorldCollisionConfig.load_from_dict(
        dict(
            cache={"obb": obb_cache_size, "mesh": mesh_cache_size},
            n_envs=1,
            checker_type=lazy.curobo.geom.sdf.world.CollisionCheckerType.MESH,
            max_distance=max_distance,
        ),
        tensor_args=tensor_args,
    )

    # To update, run world_coll_checker.load_collision_model(obstacles)
    return lazy.curobo.geom.sdf.utils.create_collision_checker(world_cfg)


class CuRoboMotionGenerator:
    """
    Class for motion generator using CuRobo backend
    """

    def __init__(
        self,
        robot,
        robot_cfg_path=None,
        robot_usd_path=None,
        device="cuda:0",
        motion_cfg_kwargs=None,
        batch_size=2,
        use_cuda_graph=True,
        debug=False,
        use_default_embodiment_only=False,
        collision_activation_distance=m.DEFAULT_COLLISION_ACTIVATION_DISTANCE,
    ):
        """
        Args:
            robot (BaseRobot): Robot for which to generate motion plans
            robot_cfg_path (None or str): If specified, the path to the robot configuration to use. If None, will
                try to use a pre-configured one directly from curobo based on the robot class of @robot
            robot_usd_path (None or str): If specified, the path to the robot USD file to use. If None, will
                try to use a pre-configured one directly from curobo based on the robot class of @robot
            device (str): Which device to use for curobo
            motion_cfg_kwargs (None or dict): If specified, keyward arguments to pass to
                MotionGenConfig.load_from_robot_config(...)
            batch_size (int): Size of batches for computing trajectories. This must be FIXED
            use_cuda_graph (bool): Whether to use CUDA graph for motion generation or not
            debug (bool): Whether to debug generation or not, setting this True will set use_cuda_graph to False implicitly
            use_default_embodiment_only (bool): Whether to use only the default embodiment for the robot or not
            collision_activation_distance (float): Distance threshold at which a collision with the world is detected.
                Increasing this value will make the motion planner more conservative in its planning with respect
                to the underlying sphere representation of the robot. Note that this does not affect self-collisions detection.
        """
        # Only support one scene for now -- verify that this is the case
        assert len(og.sim.scenes) == 1

        # Store internal variables
        self._tensor_args = lazy.curobo.types.base.TensorDeviceType(device=th.device(device))
        self.debug = debug
        self.robot = robot
        self.robot_joint_names = list(robot.joints.keys())
        self.batch_size = batch_size

        # Load robot config and usd paths and make sure paths point correctly
        robot_cfg_path_dict = robot.curobo_path if robot_cfg_path is None else robot_cfg_path
        if not isinstance(robot_cfg_path_dict, dict):
            robot_cfg_path_dict = {CuRoboEmbodimentSelection.DEFAULT: robot_cfg_path_dict}
        if use_default_embodiment_only:
            robot_cfg_path_dict = {
                CuRoboEmbodimentSelection.DEFAULT: robot_cfg_path_dict[CuRoboEmbodimentSelection.DEFAULT]
            }
        robot_usd_path = robot.usd_path if robot_usd_path is None else robot_usd_path

        # This will be shared across all MotionGen instances
        world_coll_checker = create_world_mesh_collision(
            self._tensor_args, obb_cache_size=10, mesh_cache_size=2048, max_distance=0.05
        )

        usd_help = lazy.curobo.util.usd_helper.UsdHelper()
        usd_help.stage = og.sim.stage
        self.usd_help = usd_help

        self.mg = dict()
        self.ee_link = dict()
        self.additional_links = dict()
        self.base_link = dict()

        # Grab mapping from robot joint name to index
        reset_qpos = self.robot.reset_joint_pos
        joint_idx_mapping = {joint.joint_name: i for i, joint in enumerate(self.robot.joints.values())}
        for emb_sel, robot_cfg_path in robot_cfg_path_dict.items():
            content_path = lazy.curobo.types.file_path.ContentPath(
                robot_config_absolute_path=robot_cfg_path, robot_usd_absolute_path=robot_usd_path
            )
            robot_cfg_dict = lazy.curobo.cuda_robot_model.util.load_robot_yaml(content_path)["robot_cfg"]
            robot_cfg_dict["kinematics"]["use_usd_kinematics"] = True

            # Automatically populate the locked joints and retract config from the robot values
            for joint_name, lock_val in robot_cfg_dict["kinematics"]["lock_joints"].items():
                if lock_val is None:
                    joint_idx = joint_idx_mapping[joint_name]
                    robot_cfg_dict["kinematics"]["lock_joints"][joint_name] = reset_qpos[joint_idx]
            if robot_cfg_dict["kinematics"]["cspace"]["retract_config"] is None:
                robot_cfg_dict["kinematics"]["cspace"]["retract_config"] = [
                    reset_qpos[joint_idx_mapping[joint_name]]
                    for joint_name in robot_cfg_dict["kinematics"]["cspace"]["joint_names"]
                ]

            self.ee_link[emb_sel] = robot_cfg_dict["kinematics"]["ee_link"]
            # RobotConfig.from_dict will append ee_link to link_names, so we make a copy here.
            self.additional_links[emb_sel] = robot_cfg_dict["kinematics"]["link_names"].copy()
            self.base_link[emb_sel] = robot_cfg_dict["kinematics"]["base_link"]

            robot_cfg_obj = lazy.curobo.types.robot.RobotConfig.from_dict(robot_cfg_dict, self._tensor_args)

            if isinstance(robot, HolonomicBaseRobot):
                self.update_joint_limits(robot_cfg_obj, emb_sel)

            motion_kwargs = dict(
                trajopt_tsteps=32,
                collision_checker_type=lazy.curobo.geom.sdf.world.CollisionCheckerType.MESH,
                use_cuda_graph=use_cuda_graph,
                num_ik_seeds=128,
                num_batch_ik_seeds=128,
                num_batch_trajopt_seeds=1,
                num_trajopt_noisy_seeds=1,
                ik_opt_iters=100,
                optimize_dt=True,
                num_trajopt_seeds=4,
                num_graph_seeds=4,
                interpolation_dt=og.sim.get_sim_step_dt(),
                collision_activation_distance=collision_activation_distance,
                self_collision_check=True,
                maximum_trajectory_dt=None,
                fixed_iters_trajopt=True,
                finetune_trajopt_iters=100,
                finetune_dt_scale=1.05,
            )
            if motion_cfg_kwargs is not None:
                motion_kwargs.update(motion_cfg_kwargs)

            motion_gen_config = lazy.curobo.wrap.reacher.motion_gen.MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_cfg_obj,
                world_model=None,
                world_coll_checker=world_coll_checker,
                tensor_args=self._tensor_args,
                store_trajopt_debug=self.debug,
                **motion_kwargs,
            )
            self.mg[emb_sel] = lazy.curobo.wrap.reacher.motion_gen.MotionGen(motion_gen_config)

        for mg in self.mg.values():
            mg.warmup(enable_graph=False, warmup_js_trajopt=False, batch=batch_size, warmup_joint_delta=0.0)

            # Make sure all cuda graphs have been warmed up
            for solver in [mg.ik_solver, mg.trajopt_solver, mg.finetune_trajopt_solver]:
                if solver.solver.use_cuda_graph_metrics:
                    assert solver.solver.safety_rollout._metrics_cuda_graph_init
                    if isinstance(solver, lazy.curobo.wrap.reacher.trajopt.TrajOptSolver):
                        assert solver.interpolate_rollout._metrics_cuda_graph_init
                for opt in solver.solver.optimizers:
                    if opt.use_cuda_graph:
                        assert opt.cu_opt_init

    def update_joint_limits(self, robot_cfg_obj, emb_sel):
        joint_limits = robot_cfg_obj.kinematics.kinematics_config.joint_limits
        for joint_name in self.robot.base_joint_names:
            if joint_name in joint_limits.joint_names:
                joint_idx = joint_limits.joint_names.index(joint_name)
                # Manually specify joint limits for the base_footprint_x/y/rz
                if self.robot.joints[joint_name].joint_type == JointType.JOINT_PRISMATIC:
                    joint_limits.position[0][joint_idx] = -m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT
                else:
                    # Needs to be -2pi to 2pi, instead of -pi to pi, otherwise the planning success rate is much lower
                    joint_limits.position[0][joint_idx] = -m.HOLONOMIC_BASE_REVOLUTE_JOINT_LIMIT

                joint_limits.position[1][joint_idx] = -joint_limits.position[0][joint_idx]

    def save_visualization(self, q, file_path, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        # Update obstacles
        self.update_obstacles()

        # Get robot collision spheres
        cu_js = lazy.curobo.types.state.JointState(
            position=self.tensor_args.to_device(q),
            joint_names=self.robot_joint_names,
        ).get_ordered_joint_state(self.mg[emb_sel].kinematics.joint_names)
        sph = self.mg[emb_sel].kinematics.get_robot_as_spheres(cu_js.position)
        robot_world = lazy.curobo.geom.types.WorldConfig(sphere=sph[0])

        # Combine all obstacles into a single mesh
        mesh_world = self.mg[emb_sel].world_model.get_mesh_world(merge_meshes=True)
        robot_world.add_obstacle(mesh_world.mesh[0])
        robot_world.save_world_as_mesh(file_path)

    def update_obstacles(self, ignore_objects=None):
        """
        Updates internal world collision cache representation based on sim state

        Args:
            ignore_objects (None or list of DatasetObject): If specified, objects that should
                be ignored when updating obstacles
        """
        obstacles = {"cuboid": None, "sphere": None, "mesh": [], "cylinder": None, "capsule": None}
        robot_transform = T.pose_inv(T.pose2mat(self.robot.root_link.get_position_orientation()))

        if og.sim.floor_plane is not None:
            prim = og.sim.floor_plane.prim.GetChildren()[0]
            m = lazy.curobo.util.usd_helper.get_mesh_attrs(
                prim, cache=self.usd_help._xform_cache, transform=robot_transform.numpy()
            )
            obstacles["mesh"].append(m)

        for obj in self.robot.scene.objects:
            if obj == self.robot:
                continue
            if obj.visual_only:
                continue
            if ignore_objects is not None and obj in ignore_objects:
                continue
            for link in obj.links.values():
                for collision_mesh in link.collision_meshes.values():
                    assert (
                        collision_mesh.geom_type == "Mesh"
                    ), f"collision_mesh {collision_mesh.prim_path} is not a mesh, but a {collision_mesh.geom_type}"
                    obj_pose = T.pose2mat(collision_mesh.get_position_orientation())
                    pose = robot_transform @ obj_pose
                    pos, orn = T.mat2pose(pose)
                    # xyzw -> wxyz
                    orn = orn[[3, 0, 1, 2]]
                    m = lazy.curobo.geom.types.Mesh(
                        name=collision_mesh.prim_path,
                        pose=th.cat([pos, orn]).tolist(),
                        vertices=collision_mesh.points.numpy(),
                        faces=collision_mesh.faces.numpy(),
                        scale=collision_mesh.get_world_scale().numpy(),
                    )
                    obstacles["mesh"].append(m)

        world = lazy.curobo.geom.types.WorldConfig(**obstacles)
        world = world.get_collision_check_world()
        self.mg[CuRoboEmbodimentSelection.DEFAULT].update_world(world)

    def check_collisions(
        self,
        q,
        initial_joint_pos=None,
        self_collision_check=True,
        skip_obstacle_update=False,
        attached_obj=None,
        attached_obj_scale=None,
    ):
        """
        Checks collisions between the sphere representation of the robot and the rest of the current scene

        Args:
            q (th.tensor): (N, D)-shaped tensor, representing N-total different joint configurations to check
                collisions against the world
            initial_joint_pos (None or th.tensor): If specified, the initial joint positions to set the locked joints.
                Default is the current joint positions of the robot
            self_collision_check (bool): Whether to check self-collisions or not
            skip_obstacle_update (bool): Whether to skip updating the obstacles in the world collision checker
            attached_obj (None or Dict[str, BaseObject]): If specified, a dictionary where the keys are the end-effector
                link names and the values are the corresponding BaseObject instances to attach to that link
            attached_obj_scale (None or Dict[str, float]): If specified, a dictionary where the keys are the end-effector
                link names and the values are the corresponding scale to apply to the attached object

        Returns:
            th.tensor: (N,)-shaped tensor, where each value is True if in collision, else False
        """
        # check_collisions only makes sense for the default embodiment where all the joints are actuated
        emb_sel = CuRoboEmbodimentSelection.DEFAULT

        # Update obstacles
        if not skip_obstacle_update:
            self.update_obstacles()

        q_pos = self.robot.get_joint_positions() if initial_joint_pos is None else initial_joint_pos
        q_pos = q_pos.unsqueeze(0)
        cu_joint_state = lazy.curobo.types.state.JointState(
            position=self._tensor_args.to_device(q_pos),
            joint_names=self.robot_joint_names,
        )

        # Update the locked joints with the current joint positions
        self.update_locked_joints(cu_joint_state, emb_sel)

        # Compute kinematics to get corresponding sphere representation
        cu_js = lazy.curobo.types.state.JointState(
            position=self.tensor_args.to_device(q),
            joint_names=self.robot_joint_names,
        ).get_ordered_joint_state(self.mg[emb_sel].kinematics.joint_names)

        # Attach objects if specified
        attached_info = self._attach_objects_to_robot(
            attached_obj=attached_obj,
            attached_obj_scale=attached_obj_scale,
            cu_js_batch=cu_js,
            emb_sel=emb_sel,
        )

        robot_spheres = self.mg[emb_sel].compute_kinematics(cu_js).robot_spheres
        # (N_samples, n_spheres, 4) --> (N_samples, 1, n_spheres, 4)
        robot_spheres = robot_spheres.unsqueeze(dim=1)

        with th.no_grad():
            collision_dist = (
                self.mg[emb_sel].rollout_fn.primitive_collision_constraint.forward(robot_spheres).squeeze(1)
            )
            collision_results = collision_dist > 0.0
            if self_collision_check:
                self_collision_dist = (
                    self.mg[emb_sel].rollout_fn.robot_self_collision_constraint.forward(robot_spheres).squeeze(1)
                )
                self_collision_results = self_collision_dist > 0.0
                collision_results = collision_results | self_collision_results

        # Detach objects before returning
        self._detach_objects_from_robot(attached_info, emb_sel)

        # Return results
        return collision_results  # shape (B,)

    def update_locked_joints(self, cu_joint_state, emb_sel):
        """
        Updates the locked joints and fixed transforms for the given embodiment selection
        This is needed to update curobo robot model about the current joint positions from Isaac.

        Args:
            cu_joint_state (JointState): JointState object representing the current joint positions
            emb_sel (CuRoboEmbodimentSelection): Which embodiment selection to use for updating locked joints
        """
        kc = self.mg[emb_sel].kinematics.kinematics_config
        # Update the lock joint state position
        kc.lock_jointstate.position = cu_joint_state.get_ordered_joint_state(kc.lock_jointstate.joint_names).position[0]
        # Update all the fixed transforms between the parent links and the child links of these joints
        for i, joint_name in enumerate(kc.lock_jointstate.joint_names):
            joint = self.robot.joints[joint_name]
            joint_pos = kc.lock_jointstate.position[i]
            child_link_name = joint.body1.split("/")[-1]

            # Compute the fixed transform between the parent link and the child link
            # Note that we cannot directly query the parent and child link poses from OG
            # because the cu_joint_state might not represent the current joint position in OG

            jf_to_cf_pose = joint.local_position_1, joint.local_orientation_1
            # Compute the transform from child frame to joint frame
            cf_to_jf_pose = T.invert_pose_transform(*jf_to_cf_pose)

            # Compute the transform from the joint frame to the joint frame moved by the joint position
            if joint.joint_type == JointType.JOINT_FIXED:
                jf_to_jf_moved_pos = th.zeros(3)
                jf_to_jf_moved_quat = th.tensor([0.0, 0.0, 0.0, 1.0])
            elif joint.joint_type == JointType.JOINT_PRISMATIC:
                jf_to_jf_moved_pos = th.tensor([0.0, 0.0, 0.0])
                jf_to_jf_moved_pos[["X", "Y", "Z"].index(joint.axis)] = joint_pos
                jf_to_jf_moved_quat = th.tensor([0.0, 0.0, 0.0, 1.0])
            elif joint.joint_type == JointType.JOINT_REVOLUTE:
                jf_to_jf_moved_pos = th.zeros(3)
                axis = th.zeros(3)
                axis[["X", "Y", "Z"].index(joint.axis)] = 1.0
                jf_to_jf_moved_quat = T.axisangle2quat(axis * joint_pos.cpu())
            else:
                raise NotImplementedError(f"Joint type {joint.joint_type} not supported")

            # Compute the transform from the child frame to the joint frame moved by the joint position
            cf_to_jf_moved_pose = T.pose_transform(jf_to_jf_moved_pos, jf_to_jf_moved_quat, *cf_to_jf_pose)

            # Compute the transform from the joint frame moved by the joint position to the parent frame
            jf_moved_to_pf_pose = joint.local_position_0, joint.local_orientation_0

            # Compute the transform from the child frame to the parent frame
            cf_to_pf_pose = T.pose_transform(*jf_moved_to_pf_pose, *cf_to_jf_moved_pose)
            cf_to_pf_pose = T.pose2mat(cf_to_pf_pose)

            link_idx = kc.link_name_to_idx_map[child_link_name]
            kc.fixed_transforms[link_idx] = cf_to_pf_pose

    def solve_ik_batch(
        self,
        start_state: Any,
        goal_pose: Any,
        plan_config: Any,
        link_poses: Optional[Any] = None,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    ):
        """Find IK solutions to reach a batch of goal poses from a batch of start joint states.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.ik_
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            IKResult: Result of IK solution. Check :attr:`IKResult.success`
                attribute to check which indices of the batch were successful.
            bool: Whether the IK solution was successful for the batch.
            JointState: Joint state of the robot at the goal pose.
        """
        solve_state = self.mg[emb_sel]._get_solve_state(
            lazy.curobo.wrap.reacher.types.ReacherSolveType.BATCH, plan_config, goal_pose, start_state
        )
        result = self.mg[emb_sel]._solve_ik_from_solve_state(
            goal_pose,
            solve_state,
            start_state,
            plan_config.use_nn_ik_seed,
            plan_config.partial_ik_opt,
            link_poses,
        )
        # If any of the IK seeds is successful
        success = result.success.any(dim=1)
        # Set non-successful error to infinity
        result.error[~result.success].fill_(float("inf"))
        # Get the index of the minimum error
        min_error_idx = result.error.argmin(dim=1)
        # Get the joint state with the minimum error
        joint_state = result.js_solution[range(result.js_solution.shape[0]), min_error_idx]
        joint_state = [joint_state[i] for i in range(joint_state.shape[0])]
        return result, success, joint_state

    def plan_batch(
        self,
        start_state: Any,
        goal_pose: Any,
        plan_config: Any,
        link_poses: Optional[Any] = None,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    ):
        """Plan a batch of trajectories from a batch of start joint states to a batch of goal poses.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of IK solution. Check :attr:`MotionGenResult.success`
                attribute to check which indices of the batch were successful.
            bool: Whether the IK solution was successful for the batch.
            JointState: Joint state of the robot at the goal pose.
        """
        result = self.mg[emb_sel].plan_batch(start_state, goal_pose, plan_config, link_poses=link_poses)
        success = result.success
        if result.interpolated_plan is None:
            joint_state = [None] * goal_pose.batch
        else:
            joint_state = result.get_paths()

        return result, success, joint_state

    def compute_trajectories(
        self,
        target_pos,
        target_quat,
        initial_joint_pos=None,
        is_local=False,
        max_attempts=5,
        timeout=2.0,
        ik_fail_return=5,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=None,
        attached_obj=None,
        attached_obj_scale=None,
        motion_constraint=None,
        skip_obstacle_update=False,
        ik_only=False,
        ik_world_collision_check=True,
        emb_sel=CuRoboEmbodimentSelection.DEFAULT,
    ):
        """
        Computes the robot joint trajectory to reach the desired @target_pos and @target_quat

        Args:
            target_pos (Dict[str, th.Tensor] or th.Tensor): The torch tensor shape is either (3,) or (N, 3)
                where each entry is an individual (x,y,z) position to reach with the default end-effector link specified
                @self.ee_link[emb_sel]. If a dictionary is given, the keys should be the end-effector links and
                the values should be the corresponding (N, 3) tensors
            target_quat (Dict[str, th.Tensor] or th.Tensor): The torch tensor shape is either (4,) or (N, 4)
                where each entry is an individual (x,y,z,w) quaternion to reach with the default end-effector link specified
                @self.ee_link[emb_sel]. If a dictionary is given, the keys should be the end-effector links and
                the values should be the corresponding (N, 4) tensors
            initial_joint_pos (None or th.Tensor): If specified, the initial joint positions to start the trajectory.
                Default is the current joint positions of the robot
            is_local (bool): Whether @target_pos and @target_quat are specified in the robot's local frame or the world
                global frame
            max_attempts (int): Maximum number of attempts for trying to compute a valid trajectory
            timeout (float): Maximum time in seconds allowed to solve the motion generation problem
            ik_fail_return (None or int): Number of IK attempts allowed before returning a failure. Set this to a
                low value (5) to save compute time when an unreachable goal is given
            enable_finetune_trajopt (bool): Whether to enable timing reparameterization for a smoother trajectory
            finetune_attempts (int): Number of attempts to run finetuning trajectory optimization. Every attempt will
                increase the `MotionGenPlanConfig.finetune_dt_scale` by `MotionGenPlanConfig.finetune_dt_decay` as a
                path couldn't be found with the previous smaller dt
            return_full_result (bool): Whether to return a list of raw MotionGenResult object(s) or a 2-tuple of
                (success, results); the default is the latter
            success_ratio (None or float): If set, specifies the fraction of successes necessary given self.batch_size.
                If None, will automatically be the smallest ratio (1 / self.batch_size), i.e: any nonzero number of
                successes
            attached_obj (None or Dict[str, BaseObject]): If specified, a dictionary where the keys are the end-effector
                link names and the values are the corresponding BaseObject instances to attach to that link
            attached_obj_scale (None or Dict[str, float]): If specified, a dictionary where the keys are the end-effector
                link names and the values are the corresponding scale to apply to the attached object
            motion_constraint (None or List[float]): If specified, the motion constraint vector is a 6D vector controlling
                end-effector movement (angular first, linear next): [qx, qy, qz, x, y, z]. Setting any component to 1.0
                locks that axis, forcing the planner to reach the target using only the remaining unlocked axes.
                Details can be found here: https://curobo.org/advanced_examples/3_constrained_planning.html
            skip_obstacle_update (bool): Whether to skip updating the obstacles in the world collision checker
            ik_only (bool): Whether to only run the IK solver and not the trajectory optimization
            ik_world_collision_check (bool): Whether to check for collisions in the world when running the IK solver for ik_only mode
            emb_sel (CuRoboEmbodimentSelection): Which embodiment selection to use for computing trajectories
        Returns:
            2-tuple or list of MotionGenResult: If @return_full_result is True, will return a list of raw MotionGenResult
                object(s) computed from internal batch trajectory computations. If it is False, will return 2-tuple
                (success, results), where success is a (N,)-shaped boolean tensor representing whether each requested
                target pos / quat successfully generated a motion plan, and results is a (N,)-shaped array of
                corresponding JointState objects.
        """
        # Previously, this would silently fail so we explicitly check for out-of-range joint limits here
        # This may be fixed in a recent version of CuRobo? See https://github.com/NVlabs/curobo/discussions/288
        # relevant_joint_positions_normalized = (
        #     lazy.curobo.types.state.JointState(
        #         position=self.tensor_args.to_device(self.robot.get_joint_positions(normalized=True)),
        #         joint_names=self.robot_joint_names,
        #     )
        #     .get_ordered_joint_state(self.mg[emb_sel].kinematics.joint_names)
        #     .position
        # )

        # if not th.all(th.abs(relevant_joint_positions_normalized) < 0.99):
        #     print("Robot is near joint limits! No trajectory will be computed")
        #     return None, None if not return_full_result else None

        if not skip_obstacle_update:
            self.update_obstacles()

        # If target_pos and target_quat are torch tensors, it's assumed that they correspond to the default ee_link
        if isinstance(target_pos, th.Tensor):
            target_pos = {self.ee_link[emb_sel]: target_pos}
        if isinstance(target_quat, th.Tensor):
            target_quat = {self.ee_link[emb_sel]: target_quat}

        assert target_pos.keys() == target_quat.keys(), "Expected target_pos and target_quat to have the same keys!"

        # Make sure tensor shapes are (N, 3) and (N, 4)
        target_pos = {k: v if len(v.shape) == 2 else v.unsqueeze(0) for k, v in target_pos.items()}
        target_quat = {k: v if len(v.shape) == 2 else v.unsqueeze(0) for k, v in target_quat.items()}

        for link_name in target_pos.keys():
            target_pos_link = target_pos[link_name]
            target_quat_link = target_quat[link_name]
            if not is_local:
                # Convert target pose to base link *in the eyes of curobo*.
                # For stationary arms (e.g. Franka), it is @robot.root_link / @robot.base_footprint_link_name ("base_link")
                # For holonomic robots (e.g. Tiago, R1), it is @robot.root_link ("base_footprint_x"), not @robot.base_footprint_link_name ("base_link")
                curobo_base_link_name = self.base_link[emb_sel]
                robot_pos, robot_quat = self.robot.links[curobo_base_link_name].get_position_orientation()
                target_pose = th.zeros((target_pos_link.shape[0], 4, 4))
                target_pose[:, 3, 3] = 1.0
                target_pose[:, :3, :3] = T.quat2mat(target_quat_link)
                target_pose[:, :3, 3] = target_pos_link
                inv_robot_pose = T.pose_inv(T.pose2mat((robot_pos, robot_quat)))
                target_pose = inv_robot_pose.view(1, 4, 4) @ target_pose
                target_pos_link = target_pose[:, :3, 3]
                target_quat_link = T.mat2quat(target_pose[:, :3, :3])

            # Map xyzw -> wxyz quat
            target_quat_link = target_quat_link[:, [3, 0, 1, 2]]

            # Make sure tensors are on device and contiguous
            target_pos_link = self._tensor_args.to_device(target_pos_link).contiguous()
            target_quat_link = self._tensor_args.to_device(target_quat_link).contiguous()

            target_pos[link_name] = target_pos_link
            target_quat[link_name] = target_quat_link

        # Define the plan config
        plan_cfg = lazy.curobo.wrap.reacher.motion_gen.MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=max_attempts,
            timeout=timeout,
            enable_graph_attempt=None,
            ik_fail_return=ik_fail_return,
            enable_finetune_trajopt=enable_finetune_trajopt,
            finetune_attempts=finetune_attempts,
            success_ratio=1.0 / self.batch_size if success_ratio is None else success_ratio,
        )

        # Add the pose cost metric
        if self.ee_link[emb_sel] in target_pos and motion_constraint is not None:
            plan_cfg.pose_cost_metric = lazy.curobo.wrap.reacher.motion_gen.PoseCostMetric(
                hold_partial_pose=True, hold_vec_weight=self._tensor_args.to_device(motion_constraint)
            )

        # Construct initial state
        if initial_joint_pos is None:
            q_pos = th.stack([self.robot.get_joint_positions()] * self.batch_size, axis=0)
            q_vel = th.stack([self.robot.get_joint_velocities()] * self.batch_size, axis=0)
            q_eff = th.stack([self.robot.get_joint_efforts()] * self.batch_size, axis=0)
        else:
            q_pos = th.stack([initial_joint_pos] * self.batch_size, axis=0)
            q_vel = th.zeros_like(q_pos)
            q_eff = th.zeros_like(q_pos)

        cu_joint_state = lazy.curobo.types.state.JointState(
            position=self._tensor_args.to_device(q_pos),
            # TODO: Ideally these should be nonzero, but curobo fails to compute a solution if so
            # See this note from https://curobo.org/get_started/2b_isaacsim_examples.html
            # Motion generation only generates motions when the robot is static.
            # cuRobo has an experimental mode to optimize from non-static states.
            # You can try this by passing --reactive to motion_gen_reacher.py.
            # This mode will have lower success than the static mode as now the optimization
            # has to account for the robot’s current velocity and acceleration.
            # The weights have also not been tuned for reactive mode.
            velocity=self._tensor_args.to_device(q_vel) * 0.0,
            acceleration=self._tensor_args.to_device(q_eff) * 0.0,
            jerk=self._tensor_args.to_device(q_eff) * 0.0,
            joint_names=self.robot_joint_names,
        )

        # Update the locked joints with the current joint positions
        self.update_locked_joints(cu_joint_state, emb_sel)

        cu_js_batch = cu_joint_state.get_ordered_joint_state(self.mg[emb_sel].kinematics.joint_names)

        # Attach object to robot if requested
        attached_info = self._attach_objects_to_robot(
            attached_obj=attached_obj,
            attached_obj_scale=attached_obj_scale,
            cu_js_batch=cu_js_batch,
            emb_sel=emb_sel,
        )

        all_rollout_fns = [
            fn
            for fn in self.mg[emb_sel].get_all_rollout_instances()
            if isinstance(fn, lazy.curobo.rollout.arm_reacher.ArmReacher)
        ]

        # Enable/disable costs based on whether the end-effector is in the target position
        for rollout_fn in all_rollout_fns:
            (
                rollout_fn.goal_cost.enable_cost()
                if self.ee_link[emb_sel] in target_pos
                else rollout_fn.goal_cost.disable_cost()
            )
            (
                rollout_fn.pose_convergence.enable_cost()
                if self.ee_link[emb_sel] in target_pos
                else rollout_fn.pose_convergence.disable_cost()
            )
            for additional_link in self.additional_links[emb_sel]:
                (
                    rollout_fn._link_pose_costs[additional_link].enable_cost()
                    if additional_link in target_pos
                    else rollout_fn._link_pose_costs[additional_link].disable_cost()
                )
                (
                    rollout_fn._link_pose_convergence[additional_link].enable_cost()
                    if additional_link in target_pos
                    else rollout_fn._link_pose_convergence[additional_link].disable_cost()
                )

        if ik_only:
            for rollout_fn in self.mg[emb_sel].ik_solver.get_all_rollout_instances():
                (
                    rollout_fn.primitive_collision_cost.enable_cost()
                    if ik_world_collision_check
                    else rollout_fn.primitive_collision_cost.disable_cost()
                )
                (
                    rollout_fn.primitive_collision_constraint.enable_cost()
                    if ik_world_collision_check
                    else rollout_fn.primitive_collision_constraint.disable_cost()
                )

        # Determine how many internal batches we need to run based on submitted size
        num_targets = next(iter(target_pos.values())).shape[0]
        remainder = num_targets % self.batch_size
        n_batches = math.ceil(num_targets / self.batch_size)

        # If ee_link is not in target_pos, add trivial target poses to avoid errors
        if self.ee_link[emb_sel] not in target_pos:
            target_pos[self.ee_link[emb_sel]] = self._tensor_args.to_device(th.zeros((num_targets, 3)))
            target_quat[self.ee_link[emb_sel]] = self._tensor_args.to_device(th.zeros((num_targets, 4)))
            target_quat[self.ee_link[emb_sel]][..., 0] = 1.0

        # Run internal batched calls
        results, successes, paths = [], self._tensor_args.to_device(th.tensor([], dtype=th.bool)), []
        for i in range(n_batches):
            # We're using a remainder if we're on the final batch and our remainder is nonzero
            using_remainder = (i == n_batches - 1) and remainder > 0
            offset_idx = self.batch_size * i
            end_idx = remainder if using_remainder else self.batch_size

            ik_goal_batch_by_link = dict()
            for link_name in target_pos.keys():
                target_pos_link = target_pos[link_name]
                target_quat_link = target_quat[link_name]

                batch_target_pos = target_pos_link[offset_idx : offset_idx + end_idx]
                batch_target_quat = target_quat_link[offset_idx : offset_idx + end_idx]

                # Pad the goal if we're in our final batch
                if using_remainder:
                    new_batch_target_pos = self._tensor_args.to_device(th.zeros((self.batch_size, 3)))
                    new_batch_target_pos[:end_idx] = batch_target_pos
                    new_batch_target_pos[end_idx:] = batch_target_pos[-1]
                    batch_target_pos = new_batch_target_pos
                    new_batch_target_quat = self._tensor_args.to_device(th.zeros((self.batch_size, 4)))
                    new_batch_target_quat[:end_idx] = batch_target_quat
                    new_batch_target_quat[end_idx:] = batch_target_quat[-1]
                    batch_target_quat = new_batch_target_quat

                # Create IK goal
                ik_goal_batch = lazy.curobo.types.math.Pose(
                    position=batch_target_pos,
                    quaternion=batch_target_quat,
                    name=link_name,
                )

                ik_goal_batch_by_link[link_name] = ik_goal_batch

            # Run batched planning
            if self.debug:
                self.mg[emb_sel].store_debug_in_result = True

            # Pop the main ee_link goal
            main_ik_goal_batch = ik_goal_batch_by_link.pop(self.ee_link[emb_sel])

            # If no other goals (e.g. no second end-effector), set to None
            if len(ik_goal_batch_by_link) == 0:
                ik_goal_batch_by_link = None

            plan_fn = self.plan_batch if not ik_only else self.solve_ik_batch
            result, success, joint_state = plan_fn(
                cu_js_batch, main_ik_goal_batch, plan_cfg, link_poses=ik_goal_batch_by_link, emb_sel=emb_sel
            )
            if self.debug:
                breakpoint()

            # Append results
            results.append(result)
            successes = th.concatenate([successes, success[:end_idx]])
            paths += joint_state[:end_idx]

        # Detach attached object if it was attached
        self._detach_objects_from_robot(attached_info, emb_sel)

        if return_full_result:
            return results
        else:
            return successes, paths

    def path_to_joint_trajectory(self, path, get_full_js=True, emb_sel=CuRoboEmbodimentSelection.DEFAULT):
        """
        Converts raw path from motion generator into joint trajectory sequence

        Args:
            path (JointState): Joint state path to convert into joint trajectory
            get_full_js (bool): Whether to get the full joint state
            emb_sel (CuRoboEmbodimentSelection): Which embodiment to use for the robot

        Returns:
            torch.tensor: (T, D) tensor representing the interpolated joint trajectory
                to reach the desired @target_pos, @target_quat configuration, where T is the number of interpolated
                steps and D is the number of robot joints.
        """
        cmd_plan = self.mg[emb_sel].get_full_js(path) if get_full_js else path
        return cmd_plan.get_ordered_joint_state(self.robot_joint_names).position

    def add_linearly_interpolated_waypoints(self, traj: th.Tensor, max_inter_dist=0.01):
        """
        Adds waypoints to the joint trajectory so that the joint position distance
        between each pairs of neighboring waypoints is less than @max_inter_dist

        Args:
            traj: (T, D) tensor representing the joint trajectory
            max_inter_dist (float): Maximum joint position distance between two neighboring waypoints

        Returns:
            torch.tensor: (T', D) tensor representing the interpolated joint trajectory
        """
        assert len(traj) > 1, "Plan must have at least 2 waypoints to interpolate"
        interpolated_plan = []
        for i in range(len(traj) - 1):
            # Calculate maximum difference across all dimensions
            max_diff = (traj[i + 1] - traj[i]).abs().max()
            num_intervals = math.ceil(max_diff.item() / max_inter_dist)
            interpolated_plan += multi_dim_linspace(traj[i], traj[i + 1], num_intervals, endpoint=False)

        interpolated_plan.append(traj[-1])
        return th.stack(interpolated_plan)

    def path_to_eef_trajectory(
        self, path, return_axisangle=False, emb_sel=CuRoboEmbodimentSelection.DEFAULT
    ) -> Dict[str, th.Tensor]:
        """
        Converts raw path from motion generator into end-effector trajectory sequence in the robot frame.
        This trajectory sequence can be executed by an IKController, although there is no guaranteee that
        the controller will output the same joint trajectory as the one computed by cuRobo.

        Args:
            path (JointState): Joint state path to convert into joint trajectory
            return_axisangle (bool): Whether to return the interpolated orientations in quaternion or axis-angle representation
            emb_sel (CuRoboEmbodimentSelection): Which embodiment to use for the robot

        Returns:
            Dict[str, torch.Tensor]: Mapping eef link names to (T, [6, 7])-shaped array where each entry is is the (x,y,z) position
            and (x,y,z,w) quaternion (if @return_axisangle is False) or (ax, ay, az) axis-angle orientation, specified in the robot frame.
        """
        # If the base-only embodiment is selected, the eef links stay the same, return the current eef poses in the robot frame
        if emb_sel == CuRoboEmbodimentSelection.BASE:
            link_poses = dict()
            for arm_name in self.robot.arm_names:
                link_name = self.robot.eef_link_names[arm_name]
                position, orientation = self.robot.get_relative_eef_pose(arm_name)
                if return_axisangle:
                    orientation = T.quat2axisangle(orientation)
                link_poses[link_name] = th.cat([position, orientation], dim=-1)
            return link_poses

        cmd_plan = self.mg[emb_sel].get_full_js(path)
        robot_state = self.mg[emb_sel].kinematics.compute_kinematics(path)

        link_poses = dict()

        for link_name, poses in robot_state.link_poses.items():
            position = poses.position
            # wxyz -> xyzw
            orientation = poses.quaternion[:, [1, 2, 3, 0]]

            # If the robot is holonomic, we need to transform the poses to the base link frame
            if isinstance(self.robot, HolonomicBaseRobot):
                base_link_position = th.zeros_like(position)
                base_link_position[:, 0] = cmd_plan.position[:, cmd_plan.joint_names.index("base_footprint_x_joint")]
                base_link_position[:, 1] = cmd_plan.position[:, cmd_plan.joint_names.index("base_footprint_y_joint")]
                base_link_euler = th.zeros_like(position)
                base_link_euler[:, 2] = cmd_plan.position[:, cmd_plan.joint_names.index("base_footprint_rz_joint")]
                base_link_orientation = T.euler2quat(base_link_euler)
                position, orientation = T.relative_pose_transform(
                    position, orientation, base_link_position, base_link_orientation
                )

            if return_axisangle:
                orientation = T.quat2axisangle(orientation)
            link_poses[link_name] = th.cat([position, orientation], dim=-1)

        return link_poses

    @property
    def tensor_args(self):
        """
        Returns:
            TensorDeviceType: tensor arguments used by this CuRobo instance
        """
        return self._tensor_args

    def _attach_objects_to_robot(
        self,
        attached_obj,
        attached_obj_scale,
        cu_js_batch,
        emb_sel,
    ):
        """
        Helper function to attach objects to the robot.

        Args:
            attached_obj (None or Dict[str, BaseObject]): Dictionary mapping end-effector
                link names to corresponding BaseObject instances
            attached_obj_scale (None or Dict[str, float]): Dictionary mapping end-effector
                link names to corresponding scale values
            cu_js_batch (JointState): CuRobo joint state object ordered according to kinematics
            emb_sel (CuRoboEmbodimentSelection): Which embodiment selection to use

        Returns:
            list: List of attached object information for detachment
        """
        if attached_obj is None:
            return []

        attached_info = []
        for ee_link_name, obj in attached_obj.items():
            assert isinstance(obj, RigidDynamicPrim), "attached_object should be a RigidDynamicPrim object"
            obj_paths = [geom.prim_path for geom in obj.collision_meshes.values()]
            assert len(obj_paths) <= 32, f"Expected obj_paths to be at most 32, got: {len(obj_paths)}"

            position, quaternion = self.robot.links[ee_link_name].get_position_orientation()
            # xyzw to wxyz
            quaternion = quaternion[[3, 0, 1, 2]]
            ee_pose = lazy.curobo.types.math.Pose(position=position, quaternion=quaternion).to(self._tensor_args)

            scale = m.DEFAULT_ATTACHED_OBJECT_SCALE if attached_obj_scale is None else attached_obj_scale[ee_link_name]

            self.mg[emb_sel].attach_objects_to_robot(
                joint_state=cu_js_batch,
                object_names=obj_paths,
                ee_pose=ee_pose,
                link_name=self.robot.curobo_attached_object_link_names[ee_link_name],
                scale=scale,
                pitch_scale=1.0,
                merge_meshes=True,
            )

            attached_info.append(
                {"obj_paths": obj_paths, "link_name": self.robot.curobo_attached_object_link_names[ee_link_name]}
            )

        return attached_info

    def _detach_objects_from_robot(
        self,
        attached_info,
        emb_sel,
    ):
        """
        Helper function to detach previously attached objects from the robot.

        Args:
            attached_info (list): List of dictionaries containing object paths and link names
                returned by _attach_objects_to_robot
            emb_sel (CuRoboEmbodimentSelection): Which embodiment selection to use
        """
        for info in attached_info:
            self.mg[emb_sel].detach_object_from_robot(
                object_names=info["obj_paths"],
                link_name=info["link_name"],
            )
