from functools import cached_property

import torch as th

from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.utils.python_utils import classproperty


class ArticulatedTrunkRobot(ManipulationRobot):
    """
    ManipulationRobot that is is equipped with an articulated trunk.

    NOTE: If using IK Control for both the right and left arms, note that the left arm dictates control of the trunk,
    and the right arm passively must follow. That is, sending desired delta position commands to the right end effector
    will be computed independently from any trunk motion occurring during that timestep.

    NOTE: controller_config should, at the minimum, contain:
    base: controller specifications for the controller to control this robot's base (locomotion).
        Should include:

        - name: Controller to create
        - <other kwargs> relevant to the controller being created. Note that all values will have default
            values specified, but setting these individual kwargs will override them
    """

    def get_control_dict(self):
        # In addition to super method, add in trunk endpoint state
        fcns = super().get_control_dict()

        # Add relevant trunk values
        self._add_task_frame_control_dict(
            fcns=fcns, task_name="trunk", link_name=self.joints[self.trunk_joint_names[-1]].body1.split("/")[-1]
        )

        return fcns

    @cached_property
    def trunk_links(self):
        return [self.links[name] for name in self.trunk_link_names]

    @cached_property
    def trunk_link_names(self):
        raise NotImplementedError

    @cached_property
    def trunk_joint_names(self):
        raise NotImplementedError("trunk_joint_names must be implemented in subclass")

    @cached_property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        return th.tensor([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

    @property
    def trunk_action_idx(self):
        controller_idx = self.controller_order.index("trunk")
        action_start_idx = sum([self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)])
        return th.arange(action_start_idx, action_start_idx + self.controllers["trunk"].command_dim)

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # For best generalizability use, joint controller as default
        controllers["trunk"] = "JointController"

        return controllers

    @property
    def _default_trunk_ik_controller_config(self):
        """
        Returns:
            dict: Default controller config for an Inverse kinematics controller to control this robot's trunk
        """
        return {
            "name": "InverseKinematicsController",
            "task_name": "trunk",
            "control_freq": self._control_freq,
            "reset_joint_pos": self.reset_joint_pos,
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": (
                th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
            ),
            "mode": "pose_delta_ori",
            "smoothing_filter_size": 2,
            "workspace_pose_limiter": None,
        }

    @property
    def _default_trunk_osc_controller_config(self):
        """
        Returns:
            dict: Default controller config for an Operational Space controller to control this robot's trunk
        """
        return {
            "name": "OperationalSpaceController",
            "task_name": "trunk",
            "control_freq": self._control_freq,
            "reset_joint_pos": self.reset_joint_pos,
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": (
                th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
            ),
            "mode": "pose_delta_ori",
            "workspace_pose_limiter": None,
        }

    @property
    def _default_trunk_joint_controller_config(self):
        """
        Returns:
            dict: Default base joint controller config to control this robot's base. Uses position
                control by default.
        """
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": None,
            "use_delta_commands": True,
        }

    @property
    def _default_trunk_null_joint_controller_config(self):
        """
        Returns:
            dict: Default null joint controller config to control this robot's base i.e. dummy controller
        """
        return {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "default_goal": self.reset_joint_pos[self.trunk_control_idx],
            "use_impedances": False,
        }

    @property
    def _default_controller_config(self):
        # Always run super method first
        cfg = super()._default_controller_config

        # Add supported base controllers
        cfg["trunk"] = {
            self._default_trunk_joint_controller_config["name"]: self._default_trunk_joint_controller_config,
            self._default_trunk_null_joint_controller_config["name"]: self._default_trunk_null_joint_controller_config,
            self._default_trunk_ik_controller_config["name"]: self._default_trunk_ik_controller_config,
            self._default_trunk_osc_controller_config["name"]: self._default_trunk_osc_controller_config,
        }

        return cfg

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        joint_positions = dic["joint_qpos"]
        joint_velocities = dic["joint_qvel"]
        dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos", "trunk_qvel"]

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ArticulatedTrunkRobot")
        return classes
