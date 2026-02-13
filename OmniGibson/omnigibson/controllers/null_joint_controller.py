from omnigibson.controllers.joint_controller import JointController
from omnigibson.utils.backend_utils import _compute_backend as cb


class NullJointController(JointController):
    """
    Dummy Controller class for a null-type of joint control (i.e.: no control or constant pass-through control).
    This class has a zero-size command space, and returns either an empty array for control if dof_idx is None
    else constant values as specified by @default_command (if not specified, uses zeros)
    """

    @classmethod
    def _init_state(cls, controller_id: str):
        config = cls._configs[controller_id]
        default_goal = config.get("default_goal", None)
        if default_goal is None:
            default_goal = cb.zeros(len(config["_dof_idx"]))
        cls._state[controller_id]["default_goal"] = cb.array(default_goal)

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        return dict(target=cls._state[controller_id]["default_goal"])
    
    @classmethod
    def _preprocess_command(cls, controller_id: str, command):
        # Override super and force the processed goal to be internal stored default value
        return cb.array(cls._state[controller_id]["default_goal"])

    @classmethod
    def update_default_goal(cls, controller_id: str, target):
        assert (
            len(target) == cls.control_dim(controller_id)
        ), f"Default goal must be length: {cls.control_dim(controller_id)}, got length: {len(target)}"
        cls._state[controller_id]["default_goal"] = cb.array(target)

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        return cb.array([])

    @classmethod
    def command_dim(cls, controller_id: str):
        return 0
