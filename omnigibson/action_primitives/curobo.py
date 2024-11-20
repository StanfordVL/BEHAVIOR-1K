import math
import os
from collections.abc import Iterable
from enum import Enum

import torch as th  # MUST come before importing omni!!!

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states.factory import METALINK_PREFIXES
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.utils.constants import GROUND_CATEGORIES, JointType
from omnigibson.utils.control_utils import FKSolver

# Gives 1 - 5% better speedup, according to https://github.com/NVlabs/curobo/discussions/245#discussioncomment-9265692
th.backends.cudnn.benchmark = True
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT = 5.0  # meters
m.HOLONOMIC_BASE_REVOLUTE_JOINT_LIMIT = math.pi * 2  # radians


class CuroboEmbodimentSelection(str, Enum):
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


def get_obstacles(
    reference_prim_path=None,
    only_paths=None,
    ignore_paths=None,
    only_substring=None,
    ignore_substring=None,
):
    """
    Grabs world collision representation

    Args:
        reference_prim_path (None or str): If specified, the prim path defining the collision world frame
        only_paths (None or list of str): If specified, only include these sets of prim paths in the resulting
            collision representation
        ignore_paths (None or list of str): If specified, exclude these sets of prim paths from the resulting
            collision representation
        only_substring (None or list of str): If specified, only include any prim path that includes any of these
            substrings in the resulting collision representation
        ignore_substring (None or list of str): If specified, exclude any prim path that includes any of these substrings
            from the resulting collision representation
    """
    usd_help = lazy.curobo.util.usd_helper.UsdHelper()
    usd_help.stage = og.sim.stage
    return usd_help.get_obstacles_from_stage(
        reference_prim_path=reference_prim_path,
        only_paths=only_paths,
        ignore_paths=ignore_paths,
        only_substring=only_substring,
        ignore_substring=ignore_substring,
    ).get_collision_check_world()  # WorldConfig


def get_obstacles_sphere_representation(
    obstacles,
    tensor_args,
    n_spheres=20,
    sphere_radius=0.001,
):
    """
    Gest the collision sphere representation of obstacles @obstacles
    Args:
        obstacles (list of Obstacle): Obstacles whose aggregate sphere representation will be computed
        tensor_args (TensorDeviceType): Tensor device information
        n_spheres (int or list of int): Either per-obstacle or default number of collision spheres for representing
            each obstacle
        sphere_radius (float or list of float): Either per-obstacle or default radius of collision spheres for
            representing each obstacle

    Returns:
        th.Tensor: (N, 4)-shaped tensor, where each of the N computed collision spheres are defined by (x,y,z,r),
            where (x,y,z) is the global position and r defines the sphere radius
    """
    n_obstacles = len(obstacles)
    if not isinstance(n_spheres, Iterable):
        n_spheres = [n_spheres] * n_obstacles
    if not isinstance(sphere_radius, Iterable):
        sphere_radius = [sphere_radius] * n_obstacles

    sph_list = []
    for obs, n_sph, sph_radius in zip(obstacles, n_spheres, sphere_radius):
        sph = obs.get_bounding_spheres(
            n_sph,
            sph_radius,
            pre_transform_pose=lazy.curobo.types.math.Pose(
                position=th.zeros(3), quaternion=th.tensor([1.0, 0, 0, 0])
            ).to(tensor_args),
            tensor_args=tensor_args,
            fit_type=lazy.curobo.geom.sphere_fit.SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            voxelize_method="ray",
        )
        sph_list += [s.position + [s.radius] for s in sph]

    return tensor_args.to_device(th.as_tensor(sph_list))


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
        """
        # Only support one scene for now -- verify that this is the case
        assert len(og.sim.scenes) == 1

        # Store internal variables
        self._tensor_args = lazy.curobo.types.base.TensorDeviceType(device=th.device(device))
        self.debug = debug
        self.robot = robot
        self.robot_joint_names = list(robot.joints.keys())
        self._fk = FKSolver(self.robot.robot_arm_descriptor_yamls[robot.default_arm], self.robot.urdf_path)
        self.batch_size = batch_size

        # Load robot config and usd paths and make sure paths point correctly
        robot_cfg_path_dict = robot.curobo_path if robot_cfg_path is None else robot_cfg_path
        if not isinstance(robot_cfg_path_dict, dict):
            robot_cfg_path_dict = {CuroboEmbodimentSelection.DEFAULT: robot_cfg_path_dict}
        if use_default_embodiment_only:
            robot_cfg_path_dict = {
                CuroboEmbodimentSelection.DEFAULT: robot_cfg_path_dict[CuroboEmbodimentSelection.DEFAULT]
            }
        robot_usd_path = robot.usd_path if robot_usd_path is None else robot_usd_path

        # This will be shared across all MotionGen instances
        world_coll_checker = create_world_mesh_collision(
            self._tensor_args, obb_cache_size=10, mesh_cache_size=2048, max_distance=0.05
        )

        self.mg = dict()
        self.ee_link = dict()
        self.additional_links = dict()
        self.base_link = dict()
        for emb_sel, robot_cfg_path in robot_cfg_path_dict.items():
            content_path = lazy.curobo.types.file_path.ContentPath(
                robot_config_absolute_path=robot_cfg_path, robot_usd_absolute_path=robot_usd_path
            )
            robot_cfg_dict = lazy.curobo.cuda_robot_model.util.load_robot_yaml(content_path)["robot_cfg"]
            robot_cfg_dict["kinematics"]["use_usd_kinematics"] = True

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
                interpolation_dt=0.03,
                collision_activation_distance=0.005,
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
            mg.warmup(enable_graph=False, warmup_js_trajopt=False, batch=batch_size)

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

    def save_visualization(self, q, file_path, emb_sel=CuroboEmbodimentSelection.DEFAULT):
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

    def update_obstacles(self, ignore_paths=None):
        """
        Updates internal world collision cache representation based on sim state

        Args:
            ignore_paths (None or list of str): If specified, prim path substrings that should
                be ignored when updating obstacles
        """
        print("Updating CuRobo world, reading w.r.t.", self.robot.prim_path)
        ignore_paths = [] if ignore_paths is None else ignore_paths

        # Ignore any visual only objects and any objects not part of the robot's current scene
        ignore_scenes = [scene.prim_path for scene in og.sim.scenes]
        del ignore_scenes[self.robot.scene.idx]
        ignore_visual_only = [obj.prim_path for obj in self.robot.scene.objects if obj.visual_only]

        obstacles = get_obstacles(
            reference_prim_path=self.robot.root_link.prim_path,
            ignore_substring=[
                self.robot.prim_path,  # Don't include robot paths
                "/curobo",  # Don't include curobo prim
                "visual",  # Don't include any visuals
                *METALINK_PREFIXES,  # Don't include any metalinks
                *ignore_scenes,  # Don't include any scenes the robot is not in
                *ignore_visual_only,  # Don't include any visual-only objects
                *ignore_paths,  # Don't include any additional specified paths
            ],
        )
        # All embodiment selections share the same world collision checker
        self.mg[CuroboEmbodimentSelection.DEFAULT].update_world(obstacles)
        print("Synced CuRobo world from stage.")

    def update_obstacles_fast(self):
        # All embodiment selections share the same world collision checker
        world_coll_checker = self.mg[CuroboEmbodimentSelection.DEFAULT].world_coll_checker
        for i, prim_path in enumerate(world_coll_checker._env_mesh_names[0]):
            if prim_path is None:
                continue
            prim_path_tokens = prim_path.split("/")
            obj_name = prim_path_tokens[3]
            link_name = prim_path_tokens[4]
            mesh_name = prim_path_tokens[-1]
            mesh = self.robot.scene.object_registry("name", obj_name).links[link_name].collision_meshes[mesh_name]
            pos, orn = mesh.get_position_orientation()
            inv_pos, inv_orn = T.invert_pose_transform(pos, orn)
            # xyzw -> wxyz
            inv_orn = inv_orn[[3, 0, 1, 2]]
            inv_pose = self._tensor_args.to_device(th.cat([inv_pos, inv_orn]))
            world_coll_checker._mesh_tensor_list[1][0, i, :7] = inv_pose

    def check_collisions(
        self,
        q,
        check_self_collision=True,
    ):
        """
        Checks collisions between the sphere representation of the robot and the rest of the current scene

        Args:
            q (th.tensor): (N, D)-shaped tensor, representing N-total different joint configurations to check
                collisions against the world
            check_self_collision (bool): Whether to check self-collisions or not
            emb_sel (CuroboEmbodimentSelection): Which embodiment selection to use for checking collisions

        Returns:
            th.tensor: (N,)-shaped tensor, where each value is True if in collision, else False
        """
        # check_collisions only makes sense for the default embodiment where all the joints are actuated
        emb_sel = CuroboEmbodimentSelection.DEFAULT

        # Update obstacles
        self.update_obstacles()

        q_pos = self.robot.get_joint_positions().unsqueeze(0)
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

        robot_spheres = self.mg[emb_sel].compute_kinematics(cu_js).robot_spheres
        # (N_samples, n_spheres, 4) --> (N_samples, 1, n_spheres, 4)
        robot_spheres = robot_spheres.unsqueeze(dim=1)

        with th.no_grad():
            collision_dist = (
                self.mg[emb_sel].rollout_fn.primitive_collision_constraint.forward(robot_spheres).squeeze(1)
            )
            collision_results = collision_dist > 0.0
            if check_self_collision:
                self_collision_dist = (
                    self.mg[emb_sel].rollout_fn.robot_self_collision_constraint.forward(robot_spheres).squeeze(1)
                )
                self_collision_results = self_collision_dist > 0.0
                collision_results = collision_results | self_collision_results

        # Return results
        return collision_results  # shape (B,)

    def update_locked_joints(self, cu_joint_state, emb_sel):
        """
        Updates the locked joints and fixed transforms for the given embodiment selection
        This is needed to update curobo robot model about the current joint positions from Isaac.

        Args:
            cu_joint_state (JointState): JointState object representing the current joint positions
            emb_sel (CuroboEmbodimentSelection): Which embodiment selection to use for updating locked joints
        """
        kc = self.mg[emb_sel].kinematics.kinematics_config
        # Update the lock joint state position
        kc.lock_jointstate.position = cu_joint_state.get_ordered_joint_state(kc.lock_jointstate.joint_names).position[0]
        # Update all the fixed transforms between the parent links and the child links of these joints
        for joint_name in kc.lock_jointstate.joint_names:
            joint = self.robot.joints[joint_name]
            parent_link_name, child_link_name = joint.body0.split("/")[-1], joint.body1.split("/")[-1]
            parent_link = self.robot.links[parent_link_name]
            child_link = self.robot.links[child_link_name]
            relative_pose = T.pose2mat(
                T.relative_pose_transform(
                    *child_link.get_position_orientation(), *parent_link.get_position_orientation()
                )
            )
            link_idx = kc.link_name_to_idx_map[child_link_name]
            kc.fixed_transforms[link_idx] = relative_pose

    def compute_trajectories(
        self,
        target_pos,
        target_quat,
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
        emb_sel=CuroboEmbodimentSelection.DEFAULT,
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
            emb_sel (CuroboEmbodimentSelection): Which embodiment selection to use for computing trajectories
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

        # If target_pos and target_quat are torch tensors, it's assumed that they correspond to the default ee_link
        if isinstance(target_pos, th.Tensor):
            target_pos = {self.ee_link[emb_sel]: target_pos}
        if isinstance(target_quat, th.Tensor):
            target_quat = {self.ee_link[emb_sel]: target_quat}

        assert target_pos.keys() == target_quat.keys(), "Expected target_pos and target_quat to have the same keys!"

        # Make sure tensor shapes are (N, 3) and (N, 4)
        target_pos = {k: v if len(v.shape) == 2 else v.unsqueeze(0) for k, v in target_pos.items()}
        target_quat = {k: v if len(v.shape) == 2 else v.unsqueeze(0) for k, v in target_quat.items()}

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

        # Refresh the collision state
        self.update_obstacles()

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

        # Construct initial state
        q_pos = th.stack([self.robot.get_joint_positions()] * self.batch_size, axis=0)
        q_vel = th.stack([self.robot.get_joint_velocities()] * self.batch_size, axis=0)
        q_eff = th.stack([self.robot.get_joint_efforts()] * self.batch_size, axis=0)
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
        if attached_obj is not None:
            for ee_link_name, obj in attached_obj.items():
                assert isinstance(obj, RigidPrim), "attached_object should be a RigidPrim object"
                obj_paths = [geom.prim_path for geom in obj.collision_meshes.values()]
                assert len(obj_paths) <= 32, f"Expected obj_paths to be at most 32, got: {len(obj_paths)}"

                position, quaternion = self.robot.links[ee_link_name].get_position_orientation()
                # xyzw to wxyz
                quaternion = quaternion[[3, 0, 1, 2]]
                ee_pose = lazy.curobo.types.math.Pose(position=position, quaternion=quaternion).to(self._tensor_args)
                self.mg[emb_sel].attach_objects_to_robot(
                    joint_state=cu_js_batch,
                    object_names=obj_paths,
                    ee_pose=ee_pose,
                    link_name=self.robot.curobo_attached_object_link_names[ee_link_name],
                    scale=0.99 if attached_obj_scale is None else attached_obj_scale[ee_link_name],
                    pitch_scale=1.0,
                    merge_meshes=True,
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
                    rollout_fn._link_pose_convergence[additional_link].enable_cost()
                    if additional_link in target_pos
                    else rollout_fn._link_pose_convergence[additional_link].disable_cost()
                )
                (
                    rollout_fn._link_pose_costs[additional_link].enable_cost()
                    if additional_link in target_pos
                    else rollout_fn._link_pose_costs[additional_link].disable_cost()
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

            result = self.mg[emb_sel].plan_batch(
                cu_js_batch, main_ik_goal_batch, plan_cfg, link_poses=ik_goal_batch_by_link
            )
            if self.debug:
                breakpoint()

            # Append results
            results.append(result)
            successes = th.concatenate([successes, result.success[:end_idx]])

            # If result.interpolated_plan is be None (e.g. IK failure), return Nones
            if result.interpolated_plan is None:
                paths += [None] * end_idx
            else:
                paths += result.get_paths()[:end_idx]

        # Detach attached object if it was attached
        if attached_obj is not None:
            for ee_link_name, obj in attached_obj.items():
                self.mg[emb_sel].detach_object_from_robot(
                    object_names=[geom.prim_path for geom in obj.collision_meshes.values()],
                    link_name=self.robot.curobo_attached_object_link_names[ee_link_name],
                )

        if return_full_result:
            return results
        else:
            return successes, paths

    def path_to_joint_trajectory(self, path, emb_sel=CuroboEmbodimentSelection.DEFAULT):
        """
        Converts raw path from motion generator into joint trajectory sequence

        Args:
            path (JointState): Joint state path to convert into joint trajectory
            emb_sel (CuroboEmbodimentSelection): Which embodiment to use for the robot

        Returns:
            torch.tensor: (T, D) tensor representing the interpolated joint trajectory
                to reach the desired @target_pos, @target_quat configuration, where T is the number of interpolated
                steps and D is the number of robot joints.
        """
        cmd_plan = self.mg[emb_sel].get_full_js(path)
        return cmd_plan.get_ordered_joint_state(self.robot_joint_names).position

    def convert_q_to_eef_traj(self, traj, return_axisangle=False, emb_sel=CuroboEmbodimentSelection.DEFAULT):
        """
        Converts a joint trajectory @traj into an equivalent trajectory defined by end effector poses

        Args:
            traj (torch.Tensor): (T, D)-shaped joint trajectory
            return_axisangle (bool): Whether to return the interpolated orientations in quaternion or axis-angle representation
            emb_sel (CuroboEmbodimentSelection): Which embodiment to use for the robot

        Returns:
            torch.Tensor: (T, [6, 7])-shaped array where each entry is is the (x,y,z) position and (x,y,z,w) quaternion
                (if @return_axisangle is False) or (ax, ay, az) axis-angle orientation, specified in the robot frame.
        """
        # Prune the relevant joints from the trajectory
        traj = traj[:, self.robot.arm_control_idx[self.robot.default_arm]]
        n = len(traj)

        # Use forward kinematic solver to compute the EEF link positions
        positions = self._tensor_args.to_device(th.zeros((n, 3)))
        orientations = self._tensor_args.to_device(
            th.zeros((n, 4))
        )  # This will be quat initially but we may convert to aa representation

        for i, qpos in enumerate(traj):
            pose = self._fk.get_link_poses(joint_positions=qpos, link_names=[self.ee_link[emb_sel]])
            positions[i] = pose[self.ee_link[emb_sel]][0]
            orientations[i] = pose[self.ee_link[emb_sel]][1]

        # Possibly convert orientations to aa-representation
        if return_axisangle:
            orientations = T.quat2axisangle(orientations)

        return th.concatenate([positions, orientations], dim=-1)

    @property
    def tensor_args(self):
        """
        Returns:
            TensorDeviceType: tensor arguments used by this CuRobo instance
        """
        return self._tensor_args
