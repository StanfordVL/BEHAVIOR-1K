import torch as th
from typing import Dict, Optional, Tuple
import hashlib

from omnigibson.utils.backend_utils import _compute_backend as cb


class ControllerView:
    """
    A registry that maps group keys to batched controller instances.

    Controllers with the same (robot kinematic tree pattern, body_part, controller_config) key are
    grouped into a single controller instance. Each member is assigned a controller_idx that
    indexes into the group's batched state.

    **Torch boundary:** Callers interact with this view using **torch tensors** (and plain Python
    scalars / lists where controllers already accept them). Batched controllers internally use
    compute-backend arrays (``cb``). The view translates **on the way in** (e.g. ``torch.Tensor``
    → ``cb`` for :meth:`update_goal`) and **on the way out** (e.g. ``cb`` → ``torch.Tensor`` for
    :meth:`get_control`, :meth:`get_goal`, :meth:`get_dof_idx`, gains, and command limits) so that
    **no ``cb`` array is returned from or required by** the public view API. If a compute-backend
    array is passed in, it is normalized through torch and re-imported into ``cb`` before reaching
    the controller—prefer passing ``torch.Tensor`` from new code.

    Usage:
        # At controller load time:
        group_key, controller_idx = ControllerView.register(
            body_part, controller_cfg,
            articulation_root_path, link_name
        )

        # At action time:
        ControllerView.update_goal(group_key, controller_idx, command)

        # At sim step time (called once for all groups):
        ControllerView.step_all()
    """

    # Maps group_key -> BaseController instance
    _controller_groups: Dict[str, object] = {}

    @classmethod
    def register(
        cls,
        body_part: str,
        controller_cfg: dict,
        articulation_root_path: str,
        link_name: Optional[str] = None,
        control_enabled: bool = True,
    ) -> Tuple[str, int]:
        """
        Register a controller into the appropriate group.

        If no group exists for the given key, one is created. A new member is then
        added to that group's controller via add_member().

        Args:
            body_part (str): name of the body part being controlled (e.g., "arm_right", "base")
            controller_cfg (dict): controller configuration dict (must include "name" key)
            articulation_root_path (str): articulation root prim path of the new group member
            link_name (None or str): if specified, the name of the EEF or trunk link (for IK/OSC controllers)
            control_enabled (bool): if set to False, the controller is disabled. Default is true.

        Returns:
            2-tuple:
                - str: group_key identifying the controller group
                - int: controller_idx — the member's index within that group
        """
        # Build a unique string key for a controller group.
        group_key = cls._make_key(articulation_root_path, body_part, controller_cfg)

        if group_key not in cls._controller_groups:
            from omnigibson.controllers import create_controller

            controller_name = controller_cfg.get("name")
            if controller_name in {
                "InverseKinematicsController",
                "OperationalSpaceController",
            }:
                assert link_name is not None
                cls._controller_groups[group_key] = create_controller(**controller_cfg, link_name=link_name)
            else:
                cls._controller_groups[group_key] = create_controller(**controller_cfg)

        controller = cls._controller_groups[group_key]
        controller_idx = controller.add_member(
            articulation_root_path,
            control_enabled=control_enabled,
        )

        return group_key, controller_idx

    @classmethod
    def step_all(cls):
        """
        Run a batched controller step for all registered controller groups.
        """
        for controller in cls._controller_groups.values():
            controller.step()

    @classmethod
    def update_goal(cls, group_key: str, controller_idx: int, command):
        """
        Update the goal for the controller at @controller_idx in the group identified by @group_key.

        Args:
            group_key (str): key identifying the controller group
            controller_idx (int): index of the controller within the group
            command (torch.Tensor or array-like): action command for this controller (see class docstring)
        """
        cmd = cb.from_torch(command)
        cls._controller_groups[group_key].update_goal(controller_idx, cmd)

    @classmethod
    def set_control_enabled(cls, group_key: str, controller_idx: int, enabled: bool):
        cls._controller_groups[group_key].set_control_enabled(controller_idx, enabled)

    @classmethod
    def compute_no_op_action(cls, group_key: str, controller_idx: int) -> th.Tensor:
        """
        Compute the no-op action command for the controller at @controller_idx in group @group_key.

        Args:
            group_key (str): key identifying the controller group
            controller_idx (int): index of the controller within the group

        Returns:
            torch.Tensor: no-op command for this controller
        """
        out = cls._controller_groups[group_key].compute_no_op_action(controller_idx)
        return cb.to_torch(out)

    @classmethod
    def reverse_preprocess_command(cls, group_key: str, command) -> th.Tensor:
        """
        Undo command scaling (same as :meth:`BaseController._reverse_preprocess_command`).

        Args:
            command: torch tensor or array-like in *scaled* command space

        Returns:
            torch.Tensor: command in normalized input space (for stacking into a flat action vector)
        """
        controller = cls._controller_groups[group_key]
        cmd = cb.from_torch(command)
        return cb.to_torch(controller._reverse_preprocess_command(cmd))

    @classmethod
    def reset(cls, group_key: str, controller_idx: int):
        """
        Reset the goal state for the controller at @controller_idx in the group @group_key.

        Args:
            group_key (str): key identifying the controller group
            controller_idx (int): index of the controller within the group
        """
        cls._controller_groups[group_key].reset(controller_idx)

    @classmethod
    def clear(cls):
        """
        Remove all registered controller groups. Call this inside simulator._partial_clear().
        """
        cls._controller_groups.clear()

    @classmethod
    def unregister_robot(cls, controllers: dict):
        """
        Unregister one robot from controller groups without reindexing members.

        For each (group_key, controller_idx) in @controllers:
        - Locate the shared controller group.
        - Mark controller_idx as a tombstoned (unregistered) slot.
        - Keep the group if any active members remain; delete it only when all members
        are tombstoned.

        Controller logic must mask tombstoned slots during goal updates, batched
        compute, and writeback.

        Tombstoned slots will be reused for new controller members.

        Args:
            controllers (dict): The robot.controllers dict,
                mapping controller_name -> (group_key, controller_idx)

        """
        for group_key, controller_idx in controllers.values():
            if group_key not in cls._controller_groups:
                continue
            controller = cls._controller_groups[group_key]
            controller.unregister_member(controller_idx)
            if controller.has_no_active_members():
                del cls._controller_groups[group_key]

    @classmethod
    def get_command_dim(cls, group_key: str) -> int:
        return cls._controller_groups[group_key].command_dim

    @classmethod
    def get_control_dim(cls, group_key: str) -> int:
        return cls._controller_groups[group_key].control_dim

    @classmethod
    def get_dof_idx(cls, group_key: str) -> th.Tensor:
        return cb.to_torch(cls._controller_groups[group_key].dof_idx)

    @classmethod
    def get_control_type(cls, group_key: str):
        return cls._controller_groups[group_key].control_type

    @classmethod
    def get_mode(cls, group_key: str) -> str:
        controller = cls._controller_groups[group_key]
        assert hasattr(controller, "mode"), f"Controller {type(controller).__name__} does not have a 'mode' attribute"
        return controller.mode

    @classmethod
    def get_goal_dim(cls, group_key: str) -> int:
        return cls._controller_groups[group_key].goal_dim

    @classmethod
    def get_goal(cls, group_key: str, controller_idx: int) -> dict:
        raw = cls._controller_groups[group_key].get_goal(controller_idx)
        return {k: cb.to_torch(v) for k, v in raw.items()}

    @classmethod
    def get_control(cls, group_key: str, controller_idx: int) -> th.Tensor:
        return cb.to_torch(cls._controller_groups[group_key].get_control(controller_idx))

    @classmethod
    def get_use_delta_commands(cls, group_key: str) -> bool:
        return cls._controller_groups[group_key].use_delta_commands

    @classmethod
    def get_controller_type_str(cls, group_key: str) -> str:
        """Python class name of the batched controller (e.g. ``JointController``), for UI / diagnostics."""
        return type(cls._controller_groups[group_key]).__name__

    @classmethod
    def is_controller_type(cls, group_key: str, controller_cls) -> bool:
        return isinstance(cls._controller_groups[group_key], controller_cls)

    @classmethod
    def get_isaac_kp(cls, group_key: str):
        isaac_kp = cls._controller_groups[group_key].isaac_kp
        if isaac_kp is None:
            return None
        return cb.to_torch(isaac_kp)

    @classmethod
    def get_isaac_kd(cls, group_key: str):
        isaac_kd = cls._controller_groups[group_key].isaac_kd
        if isaac_kd is None:
            return None
        return cb.to_torch(isaac_kd)

    @classmethod
    def get_command_input_limits(cls, group_key: str):
        limits = cls._controller_groups[group_key].command_input_limits
        if limits is None:
            return None
        lo, hi = limits
        return (cb.to_torch(lo), cb.to_torch(hi))

    @classmethod
    def get_motor_type(cls, group_key: str) -> str:
        controller = cls._controller_groups[group_key]
        assert hasattr(
            controller, "motor_type"
        ), f"Controller {type(controller).__name__} does not have a 'motor_type' attribute"
        return controller.motor_type

    @classmethod
    def is_grasping(cls, group_key: str, controller_idx: int):
        return cls._controller_groups[group_key].is_grasping(controller_idx)

    @classmethod
    def dump_state(cls, group_key: str, controller_idx: int) -> dict:
        controller = cls._controller_groups[group_key]
        return controller.dump_state(controller_idx=controller_idx)

    @classmethod
    def load_state(cls, group_key: str, controller_idx: int, state: dict):
        controller = cls._controller_groups[group_key]
        controller.load_state(controller_idx=controller_idx, state=state)

    @classmethod
    def serialize(cls, group_key: str, controller_idx: int, state: dict) -> th.Tensor:
        controller = cls._controller_groups[group_key]
        return controller.serialize(state=state, controller_idx=controller_idx)

    @classmethod
    def deserialize(cls, group_key: str, controller_idx: int, state: th.Tensor) -> Tuple[dict, int]:
        controller = cls._controller_groups[group_key]
        return controller.deserialize(state=state, controller_idx=controller_idx)

    @staticmethod
    def _freeze_for_hash(value):
        """
        Recursively convert nested structures into hashable representations.
        """
        if isinstance(value, dict):
            return tuple(sorted((k, ControllerView._freeze_for_hash(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ControllerView._freeze_for_hash(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(ControllerView._freeze_for_hash(v) for v in value))
        # Fallback for custom / array-like objects
        try:
            hash(value)
            return value
        except TypeError:
            return str(value)

    @classmethod
    def _make_key(cls, articulation_root_path: str, body_part: str, controller_cfg: dict) -> str:
        from omnigibson.utils.usd_utils import get_robot_kinematic_tree_pattern

        pattern = get_robot_kinematic_tree_pattern(articulation_root_path)
        frozen_cfg = cls._freeze_for_hash(controller_cfg)
        cfg_bytes = repr(frozen_cfg).encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
        return f"{pattern}__{body_part}__{cfg_hash}"
