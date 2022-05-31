from igibson.controllers.controller_base import (
    REGISTERED_CONTROLLERS,
    REGISTERED_LOCOMOTION_CONTROLLERS,
    REGISTERED_MANIPULATION_CONTROLLERS,
    IsGraspingState,
    ControlType,
    LocomotionController,
    ManipulationController,
    GripperController,
)
from igibson.controllers.dd_controller import DifferentialDriveController
from igibson.controllers.ik_controller import InverseKinematicsController
from igibson.controllers.joint_controller import JointController
from igibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from igibson.controllers.null_joint_controller import NullJointController
from igibson.utils.python_utils import assert_valid_key


def create_controller(name, **kwargs):
    """
    Creates a controller of type @name with corresponding necessary keyword arguments @kwargs

    :param name: str, type of controller to use (e.g. JointController, InverseKinematicsController, etc.)
    :param kwargs: Any relevant keyword arguments to pass to the controller

    :return Controller: created controller
    """
    assert_valid_key(key=name, valid_keys=REGISTERED_CONTROLLERS, name="controller")
    controller_cls = REGISTERED_CONTROLLERS[name]

    return controller_cls(**kwargs)
