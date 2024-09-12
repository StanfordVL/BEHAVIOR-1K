# Third Party
from collections.abc import Iterable

import numpy as np
import torch as th  # MUST come before importing omni!!!

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.utils.control_utils import FKSolver

# Gives 1 - 5% better speedup, according to https://github.com/NVlabs/curobo/discussions/245#discussioncomment-9265692
th.backends.cudnn.benchmark = True
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True


def create_collision_world(tensor_args):
    """
    Creates a CuRobo CollisionMeshWorld to use for collision checking

    Args:
        tensor_args (TensorDeviceType): Tensor device information

    Returns:
        MeshCollisionWorld: collision world used to check against for collisions
    """
    # Checks objA inside objB
    world = lazy.curobo.geom.types.WorldConfig()
    usd_help = lazy.curobo.util.usd_helper.UsdHelper()
    usd_help.stage = og.sim.stage
    # obstacles = usd_help.get_obstacles_from_stage(only_substring=[objB.prim_path], ignore_substring=[
    #     "visual"]).get_collision_check_world()  # WorldConfig
    world_cfg = lazy.curobo.geom.sdf.world.WorldCollisionConfig.load_from_dict(
        dict(
            cache={"obb": 10, "mesh": 1024},
            n_envs=1,
            checker_type=lazy.curobo.geom.sdf.world.CollisionCheckerType.MESH,
            max_distance=0.1,
        ),
        # obstacles,
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
        ignore_paths (None or list of str): If specified, exclude any prim path that includes any of these substrings
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


def get_sphere_representation(
    obj, tensor_args, reference_prim_path=None, n_spheres=100, sphere_radius=0.001, extra_links=None, n_extra_spheres=20
):
    """
    Gest the collision sphere representation of object @obj as well as any additional links specified by @extra_links

    Args:
        obj (BaseObject or None): Non-articulated object whose sphere representation will be computed. If None, will
            not use any object
        tensor_args (TensorDeviceType): Tensor device information
        reference_prim_path (None or str): If specified, the reference prim path from which all sphere poses will
            be computed
        n_spheres (int): Number of collision spheres for representing @obj
        sphere_radius (float): Radius of the generated collision spheres
        extra_links (None or list of RigidPrim): If specified, any additional links which should be included
            in the sphere representation
        n_extra_spheres (int): Number of collision spheres for representing each additional link specified in
            @extra_links

    Returns:
        th.Tensor: (N, 4)-shaped tensor, where each of the N computed collision spheres are defined by (x,y,z,r),
            where (x,y,z) is the position and r defines the sphere radius taken wrt obj.prim_path
    """
    obstacles = []
    n_spheres_total = []

    if obj is not None:
        target_obstacle = get_obstacles(
            reference_prim_path=reference_prim_path,
            only_substring=[obj.prim_path],
            ignore_substring=["collision", "container"],
        )
        assert (
            len(target_obstacle.objects) == 1
        ), f"Found multiple target obstacles, but expected 1. Got: {[obj.name for obj in target_obstacle.objects]}"
        obstacles += [target_obstacle.get_obstacle(next(iter(obj.root_link.visual_meshes.values())).prim_path)]
        n_spheres_total += [n_spheres]

    if extra_links is not None:
        extra_coll_paths = [geom.prim_path for link in extra_links for geom in link.collision_meshes.values()]
        extra_obstacles = get_obstacles(
            reference_prim_path=reference_prim_path,
            only_substring=extra_coll_paths,
        )
        obstacles += [extra_obstacles.get_obstacle(coll_path) for coll_path in extra_coll_paths]
        n_spheres_total += [n_extra_spheres] * len(extra_coll_paths)

    return get_obstacles_sphere_representation(
        obstacles=obstacles,
        tensor_args=tensor_args,
        n_spheres=n_spheres_total,
        sphere_radius=sphere_radius,
    )


class CuRoboMotionGenerator:
    """
    Class for motion generator using CuRobo backend
    """

    def __init__(
        self,
        robot,
        robot_cfg_path=None,
        ee_link=None,
        device="cuda:0",
        motion_cfg_kwargs=None,
        ignore_collision_paths=None,
        batch_size=2,
        debug=False,
    ):
        """
        Args:
            robot (BaseRobot): Robot for which to generate motion plans
            robot_cfg_path (None or str): If specified, the path to the robot configuration to use. If None, will
                try to use a pre-configured one directly from curobo based on the robot class of @robot
            ee_link (None or str): If specified, the link name representing the end-effector to track. None defaults to
                value already set in the config from @robot_cfg
            device (str): Which device to use for curobo
            motion_cfg_kwargs (None or dict): If specified, keyward arguments to pass to
                MotionGenConfig.load_from_robot_config(...)
            ignore_collision_paths (None or list): If specified, prim path(s) to prims that should be ignored
                during collision checking
            batch_size (int): Size of batches for computing trajectories. This must be FIXED
            debug (bool): Whether to debug generation or not
        """
        # Only support one scene for now -- verify that this is the case
        assert len(og.sim.scenes)

        # Define arguments to pass to motion gen config
        self._tensor_args = lazy.curobo.types.base.TensorDeviceType(device=th.device(device))
        self.debug = debug

        # Load robot config and make sure paths point correctly
        robot_cfg_path = robot.curobo_path if robot_cfg_path is None else robot_cfg_path
        robot_cfg = lazy.curobo.util_file.load_yaml(robot_cfg_path)["robot_cfg"]
        robot_cfg["kinematics"]["external_asset_path"] = gm.ASSET_PATH
        robot_cfg["kinematics"]["external_robot_configs_path"] = robot_cfg_path

        # Possibly update ee_link
        if ee_link is not None:
            robot_cfg["kinematics"]["ee_link"] = ee_link

        motion_kwargs = dict(
            trajopt_tsteps=32,
            collision_checker_type=lazy.curobo.geom.sdf.world.CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_ik_seeds=12,
            num_batch_ik_seeds=12,
            num_batch_trajopt_seeds=1,
            ik_opt_iters=60,
            # optimize_dt=False,
            # num_trajopt_seeds=12,
            # num_graph_seeds=12,
            interpolation_dt=0.03,
            collision_cache={"obb": 10, "mesh": 1024},
            collision_max_outside_distance=0.05,
            collision_activation_distance=0.025,
            acceleration_scale=1.0,
            self_collision_check=True,
            # maximum_trajectory_dt=0.25, #2.0, #0.25,
            fixed_iters_trajopt=True,
            finetune_trajopt_iters=100,
            finetune_dt_scale=1.05,
            velocity_scale=[1.0] * robot.n_joints,  # [0.25, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        if motion_cfg_kwargs is not None:
            motion_kwargs.update(motion_cfg_kwargs)
        motion_gen_config = lazy.curobo.wrap.reacher.motion_gen.MotionGenConfig.load_from_robot_config(
            robot_cfg,
            lazy.curobo.geom.types.WorldConfig(),  # world_cfg,
            self._tensor_args,
            store_trajopt_debug=self.debug,
            **motion_kwargs,
        )
        self.mg = lazy.curobo.wrap.reacher.motion_gen.MotionGen(motion_gen_config)
        # print("CuRobo warming up...")
        # self.mg.warmup(enable_graph=False, warmup_js_trajopt=False, batch=batch_size)
        # print("CuRobo is ready.")

        # Store internal variables
        self.robot = robot
        self.ee_link = robot_cfg["kinematics"]["ee_link"]
        self._fk = FKSolver(self.robot.robot_arm_descriptor_yamls[robot.default_arm], self.robot.urdf_path)
        self._usd_help = lazy.curobo.util.usd_helper.UsdHelper()
        self._usd_help.stage = og.sim.stage
        self._ignore_collision_paths = [] if ignore_collision_paths is None else ignore_collision_paths
        assert batch_size >= 2, f"batch_size must be >= 2! Got: {batch_size}"
        self.batch_size = batch_size

        # Initialize other variables modified at runtime
        self._ignore_paths = None

    def update_obstacles(self):
        """
        Updates internal world collision cache representation based on sim state
        """
        print("Updating CuRobo world, reading w.r.t.", self.robot.prim_path)
        # Infer how many collision bodies there are in sim, and add 10 as an additional buffer
        n_cols = 0

        # Ignore all container metalinks
        self._ignore_paths = ["container"]

        # Ignore any visual only objects and any objects not part of the robot's current scene
        for idx, scene in enumerate(og.sim.scenes):
            if idx == self.robot.scene.idx:
                for obj in scene.objects:
                    if obj.visual_only:
                        self._ignore_paths.append(obj.prim_path)
                        continue
                    for link in obj.links.values():
                        n_cols += len(link.collision_meshes)
            else:
                self._ignore_paths.append(scene.prim_path)

        # # TODO: This should be modified dynamically but it breaks )):
        # assert isinstance(self.mg.world_coll_checker, WorldMeshCollision)
        # self.mg.world_coll_checker.clear_cache()
        # self.mg.world_coll_checker.create_collision_cache(mesh_cache=n_cols + 10)

        obstacles = self._usd_help.get_obstacles_from_stage(
            reference_prim_path=self.robot.root_link.prim_path,
            ignore_substring=[
                self.robot.prim_path,  # Don't include robot paths
                "/World/ground_plane",  # Don't include collisions with floor
                "visual",  # Don't include any visuals
                "/curobo",  # Don't include curobo prim
                *self._ignore_collision_paths,  # Don't include manually-specified collision paths
                *self._ignore_paths,  # Don't include any additional specified paths
            ],
        ).get_collision_check_world()
        self.mg.update_world(obstacles)
        print("Synced CuRobo world from stage.")

    def check_collisions(
        self,
        q,
        activation_distance=0.01,
    ):
        """
        Checks collisions between the sphere representation of the robot and the rest of the current scene

        Args:
            q (th.tensor): (B, N)-shaped tensor, representing B-total different joint configurations to check
                collisions against the world
            activation_distance (float): Safety buffer around robot mesh representation which will trigger a
                collision check

        Returns:
            th.tensor: (B,)-shaped tensor, where each value is True if in collision, else False
        """
        # Update obstacles
        self.update_obstacles()

        # Compute kinematics to get corresponding sphere representation
        cu_js = lazy.curobo.types.state.JointState(position=self.tensor_args.to_device(q))
        robot_spheres = self.mg.compute_kinematics(cu_js).robot_spheres
        # (N_samples, n_obs_spheres, 4) --> (N_samples, 1, n_spheres, 4)
        robot_spheres = robot_spheres.unsqueeze(dim=1)

        # Run direct collision check
        with th.no_grad():
            # Run the overlap check
            # Sphere shape should be (N_queries, 1, n_obs_spheres, 4), where 4 --> (x,y,z,radius)
            coll_query_buffer = lazy.curobo.geom.sdf.world.CollisionQueryBuffer()
            coll_query_buffer.update_buffer_shape(
                shape=robot_spheres.shape,
                tensor_args=self.tensor_args,
                collision_types=self.mg.world_coll_checker.collision_types,
            )

            dist = self.mg.world_coll_checker.get_sphere_collision(
                robot_spheres,
                coll_query_buffer,
                weight=th.tensor([50000.0], device=self.tensor_args.device),
                activation_distance=th.tensor([activation_distance], device=self.tensor_args.device),
                env_query_idx=None,
                return_loss=False,
            ).squeeze(
                dim=1
            )  # shape (N_samples, n_spheres)

            # Non-zero distances correspond to a collision detection (or close to a collision, within activation_distance
            # So valid collision-free samples are those where max(n_obs_spheres) == 0 for a given sample
            collision_results = dist.max(dim=-1).values != 0

        # Return results
        return collision_results  # shape (B,)

    def compute_trajectory(
        self,
        target_pos,
        target_quat,
        is_local=False,
        max_attempts=5,
        enable_finetune_trajopt=True,
        return_full_result=False,
        success_ratio=None,
        attached_obj=None,
    ):
        """
        Computes the robot joint trajectory to reach the desired @target_pos and @target_quat

        Args:
            target_pos ((N,3)-tensor): (N, 3) or (3,)-shaped tensor, where N is expected to be self.batch_size, and each
                entry is (x,y,z) position to reach. If only a single (3,) array is given but N > 1, it will
                be broadcast to create the appropriate array.
            target_quat ((N,4)-tensor): (N, 4) or (4,)-shaped tensor, where N is expected to be self.batch_size, and each
                entry is (x,y,z,w) quaternion orientation to reach. If only a single (3,) array is given but N > 1,
                it will be broadcast to create the appropriate array.
            is_local (bool): Whether @target_pos and @target_quat are specified in the robot's local frame or the world
                global frame
            max_attempts (int): Maximum number of attempts for trying to compute a valid trajectory
            enable_finetune_trajopt (bool): Whether to enable timing reparameterization for a smoother trajectory
            return_full_result (bool): Whether to return the full result or not. If False, will randomly sample a single
                (successful) trajectory to return
            success_ratio (None or float): If set, specifies the fraction of successes necessary given self.batch_size.
                If None, will automatically be the smallest ratio (1 / self.batch_size), i.e: any nonzero number of
                successes
            attached_obj (None or BaseObject): If specified, the object to attach to the robot end-effector when
                solving for this trajectory

        Returns:
            None or th.tensor or MotionGenResult: If @return_all is set, will return the raw MotionGenResult object
                computed from the batch trajectory computation. If not set, and there is at least a single successful
                trajectory, will return a randomly sampled (N, D) tensor representing the interpolated joint trajectory
                to reach the desired @target_pos, @target_quat configuration, where N is the number of interpolated
                steps and D is the number of robot joints. If there is no successful trajectory, will return None.
        """
        # TODO: Add sanity check for qpos being near limits -- this seems to autofail solutions
        if not th.all(th.abs(self.robot.get_joint_positions(normalized=True))[:-2] < 0.99):
            print("Robot is near joint limits! No trajectory will be computed")
            return None

        # Define the plan config
        plan_cfg = lazy.curobo.wrap.reacher.motion_gen.MotionGenPlanConfig(
            enable_graph=False,
            # enable_graph_attempt=max_attempts,
            max_attempts=max_attempts,
            timeout=2.0,
            ik_fail_return=5,
            # num_ik_seeds=10,
            # num_trajopt_seeds=10,
            enable_finetune_trajopt=enable_finetune_trajopt,
            finetune_attempts=1,
            success_ratio=1.0 / self.batch_size if success_ratio is None else success_ratio,
        )

        # Refresh the collision state
        self.update_obstacles()

        # self.mg.reset()

        # Check if batch size is greater than 1, and if so, make sure target pos / quat are the appropriate shapes
        if len(target_pos.shape) == 1:
            target_pos = th.tile(target_pos.view(1, 3), (self.batch_size, 1))
        if len(target_quat.shape) == 1:
            target_quat = th.tile(target_quat.view(1, 4), (self.batch_size, 1))
        assert target_pos.shape[0] == self.batch_size
        assert target_quat.shape[0] == self.batch_size

        # Make sure the specified target pose is in the robot frame
        robot_pos, robot_quat = self.robot.get_position_orientation()
        if not is_local:
            target_pose = th.zeros((self.batch_size, 4, 4))
            target_pose[:, 3, 3] = 1.0
            target_pose[:, :3, :3] = T.quat2mat(target_quat)
            target_pose[:, :3, 3] = target_pos
            inv_robot_pose = th.eye(4)
            inv_robot_ori = T.quat2mat(robot_quat).T
            inv_robot_pose[:3, :3] = inv_robot_ori
            inv_robot_pose[:3, 3] = -inv_robot_ori @ robot_pos
            target_pose = inv_robot_pose.view(1, 4, 4) @ target_pose
            target_pos = target_pose[:, :3, 3]
            target_quat = T.mat2quat(target_pose[:, :3, :3])

        # Map xyzw -> wxyz quat
        target_quat = target_quat[:, [3, 0, 1, 2]]

        # Create IK goal
        ik_goal_batch = lazy.curobo.types.math.Pose(
            position=self._tensor_args.to_device(target_pos).contiguous(),
            quaternion=self._tensor_args.to_device(target_quat).contiguous(),
        )

        # Construct initial state
        q_pos = th.stack([self.robot.get_joint_positions()] * self.batch_size, axis=0)
        q_vel = th.stack([self.robot.get_joint_velocities()] * self.batch_size, axis=0)
        q_eff = th.stack([self.robot.get_joint_efforts()] * self.batch_size, axis=0)
        sim_js_names = list(self.robot.joints.keys())
        cu_js_batch = lazy.curobo.types.state.JointState(
            position=self._tensor_args.to_device(q_pos),
            velocity=self._tensor_args.to_device(q_vel) * 0.0,
            # TODO: Ideally this should be nonzero, but curobo fails to compute a solution if so
            acceleration=self._tensor_args.to_device(q_eff) * 0.0,
            jerk=self._tensor_args.to_device(q_eff) * 0.0,
            joint_names=sim_js_names,
        ).get_ordered_joint_state(self.mg.kinematics.joint_names)

        # Attach object to robot if requested
        if attached_obj is not None:
            obj_paths = [geom.prim_path for geom in attached_obj.root_link.collision_meshes.values()]
            assert len(obj_paths) <= 32, f"Expected obj_paths to be at most 32, got: {len(obj_paths)}"
            # if len(obj_paths) > 5:
            #     # Use visual paths instead
            #     obj_paths = [geom.prim_path for geom in attached_obj.root_link.visual_meshes.values()]
            self.mg.attach_objects_to_robot(
                joint_state=cu_js_batch,
                object_names=obj_paths,
            )

        # Run batched planning
        if self.debug:
            self.mg.store_debug_in_result = True

        result = self.mg.plan_batch(cu_js_batch, ik_goal_batch, plan_cfg)

        if self.debug:
            from IPython import embed

            embed()

        # Detach attached object if it was attached
        if attached_obj is not None:
            self.mg.detach_object_from_robot()

        if return_full_result:
            return result

        else:
            succ = th.any(result.success).item()
            if succ:
                # Sample random successful path
                successful_paths = result.get_successful_paths()
                cmd_plan = successful_paths[np.random.choice(len(successful_paths))]
                return self.path_to_joint_trajectory(path=cmd_plan, joint_names=sim_js_names)

            else:
                print("CuRobo plan did not converge to a solution.")

    def path_to_joint_trajectory(self, path, joint_names=None):
        """
        Converts raw path from motion generator into joint trajectory sequence

        Args:
            path (JointState): Joint state path to convert into joint trajectory
            joint_names (None or list): If specified, the individual joints to use when constructing the joint
                trajectory. If None, will use all joints.

        Returns:
            torch.tensor: (N, D) joint trajectory tensor
        """
        cmd_plan = self.mg.get_full_js(path)
        # get only joint names that are in both:
        idx_list = []
        common_js_names = []
        jnts_to_idx = {name: i for i, name in enumerate(self.robot.joints.keys())}
        joint_names = list(self.robot.joints.keys()) if joint_names is None else joint_names
        for x in joint_names:
            if x in cmd_plan.joint_names:
                idx_list.append(jnts_to_idx[x])
                common_js_names.append(x)

        cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
        return cmd_plan.position

    def convert_q_to_eef_traj(self, traj, return_aa=False):
        """
        Converts a joint trajectory @traj into an equivalent trajectory defined by end effector poses

        Args:
            traj (np.ndarray): (N, D)-shaped joint trajectory
            return_aa (bool): Whether to return the interpolated orientations in quaternion or axis-angle representation

        Returns:
            np.ndarray: (N, [6, 7])-shaped array where each entry is is the (x,y,z) position and (x,y,z,w)
                quaternion (if @return_aa is False) or (ax, ay, az) axis-angle orientation, specified in the robot
                frame.
        """
        # Prune the relevant joints from the trajectory
        traj = traj[:, self.robot.arm_control_idx[self.robot.default_arm]]
        n = len(traj)

        # Use forward kinematic solver to compute the EEF link positions
        positions = np.zeros((n, 3))
        orientations = np.zeros((n, 4))  # This will be quat initially but we may convert to aa representation

        for i, qpos in enumerate(traj):
            pose = self._fk.get_link_poses(joint_positions=qpos, link_names=[self.ee_link])
            positions[i] = pose[self.ee_link][0]
            orientations[i] = pose[self.ee_link][1]

        # Possibly convert orientations to aa-representation
        if return_aa:
            orientations = T.quat2axisangle(orientations)

        return np.concatenate([positions, orientations], axis=-1)

    @property
    def tensor_args(self):
        """
        Returns:
            TensorDeviceType: tensor arguments used by this CuRobo instance
        """
        return self._tensor_args
