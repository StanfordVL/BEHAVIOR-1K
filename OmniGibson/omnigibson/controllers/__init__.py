from omnigibson.controllers.controller_base import ControlType, IsGraspingState
from omnigibson.controllers.dd_controller import DifferentialDriveController
from omnigibson.controllers.holonomic_base_joint_controller import HolonomicBaseJointController
from omnigibson.controllers.ik_controller import InverseKinematicsController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.multi_finger_gripper_controller import MultiFingerGripperController
from omnigibson.controllers.null_joint_controller import NullJointController
from omnigibson.controllers.osc_controller import OperationalSpaceController

REGISTERED_CONTROLLERS = {
    "DifferentialDriveController": DifferentialDriveController,
    "HolonomicBaseJointController": HolonomicBaseJointController,
    "InverseKinematicsController": InverseKinematicsController,
    "JointController": JointController,
    "MultiFingerGripperController": MultiFingerGripperController,
    "NullJointController": NullJointController,
    "OperationalSpaceController": OperationalSpaceController,
}

__all__ = [
    "ControlType",
    "DifferentialDriveController",
    "HolonomicBaseJointController",
    "InverseKinematicsController",
    "IsGraspingState",
    "JointController",
    "MultiFingerGripperController",
    "NullJointController",
    "OperationalSpaceController",
    "REGISTERED_CONTROLLERS",
]
