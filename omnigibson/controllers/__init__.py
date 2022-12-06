from omnigibson.controllers.controller_base import (
    REGISTERED_CONTROLLERS,
    REGISTERED_LOCOMOTION_CONTROLLERS,
    REGISTERED_MANIPULATION_CONTROLLERS,
    IsGraspingState,
    ControlType,
    LocomotionController,
    ManipulationController,
    GripperController,
)
from omnigibson.controllers.dd_controller import DifferentialDriveController
from omnigibson.controllers.ik_controller import InverseKinematicsController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from omnigibson.controllers.null_joint_controller import NullJointController
from omnigibson.utils.python_utils import assert_valid_key


def create_controller(name, **kwargs):
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
