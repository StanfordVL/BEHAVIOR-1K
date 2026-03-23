from typing import Any

from omnigibson.controllers.controller_base import (
    REGISTERED_CONTROLLERS,
    REGISTERED_LOCOMOTION_CONTROLLERS,
    REGISTERED_MANIPULATION_CONTROLLERS,
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.controllers.dd_controller import DifferentialDriveController
from omnigibson.controllers.holonomic_base_joint_controller import HolonomicBaseJointController
from omnigibson.controllers.ik_controller import InverseKinematicsController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from omnigibson.controllers.null_joint_controller import NullJointController
from omnigibson.controllers.osc_controller import OperationalSpaceController
from omnigibson.utils.python_utils import assert_valid_key


def create_controller(name, **kwargs: Any):
    """
    Creates a controller of type @name with corresponding necessary keyword arguments @kwargs

    Args:
        name (str): type of controller to use (e.g. JointController, InverseKinematicsController, etc.)
        **kwargs: Any relevant keyword arguments to pass to the controller

    Returns:
        Controller: created controller
    """
    assert_valid_key(key=name, valid_keys=REGISTERED_CONTROLLERS, name="controller")
    controller_cls = REGISTERED_CONTROLLERS[name]

    return controller_cls(**kwargs)


__all__ = [
    "ControlType",
    "create_controller",
    "DifferentialDriveController",
    "GripperController",
    "HolonomicBaseJointController",
    "InverseKinematicsController",
    "IsGraspingState",
    "JointController",
    "LocomotionController",
    "ManipulationController",
    "MultiFingerGripperController",
    "NullJointController",
    "OperationalSpaceController",
    "REGISTERED_CONTROLLERS",
    "REGISTERED_LOCOMOTION_CONTROLLERS",
    "REGISTERED_MANIPULATION_CONTROLLERS",
]
