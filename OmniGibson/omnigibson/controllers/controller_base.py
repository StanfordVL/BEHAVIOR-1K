import math
from collections.abc import Iterable
from enum import IntEnum

import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.python_utils import Recreatable, Serializable, assert_valid_key
from copy import deepcopy

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Set default isaac kp / kd for controllers
m.DEFAULT_ISAAC_KP = 1e7
m.DEFAULT_ISAAC_KD = 1e5


class IsGraspingState(IntEnum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


# Define macros
class ControlType:
    NONE = -1
    POSITION = 0
    VELOCITY = 1
    EFFORT = 2
    _MAPPING = {
        "none": NONE,
        "position": POSITION,
        "velocity": VELOCITY,
        "effort": EFFORT,
    }
    VALID_TYPES = set(_MAPPING.values())
    VALID_TYPES_STR = set(_MAPPING.keys())

    @classmethod
    def get_type(cls, type_str):
        """
        Args:
            type_str (str): One of "position", "velocity", or "effort" (any case), and maps it
                to the corresponding type

        Returns:
            ControlType: control type corresponding to the associated string
        """
        assert_valid_key(key=type_str.lower(), valid_keys=cls._MAPPING, name="control type")
        return cls._MAPPING[type_str.lower()]


class BaseController(Serializable, Recreatable):
    """
    Singleton-style base class for controllers. All controller classes should inherit this and operate
    on per-robot state stored in class-level dicts keyed by controller_id.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Per-subclass storage
        # Dict keyed by controller_id (f"{robot.name}:{controller_name}")
        cls._configs = {} # processed controller configs per robot/controller
        cls._goals = {} # current goal
        cls._controls = {} # last computed control signal, used by deploy_control
        cls._state = {} # extra per-controller internal state that needs persistenc (filter buffers, integrator terms, smoothing history)
        cls._robots = {} # actual robot instance, to query control_dict and other robot data

    # Shared cache across all controller classes for a single simulation step
    _ROBOT_CONTROL_STEP_CACHE = {}

    @classmethod
    def register(cls, controller_id: str, config: dict, robot=None):
        """
        Register a robot with its config. Config should contain all values required by this controller.
        """
        cls._configs[controller_id] = cls._process_config(controller_id, config)
        cls._goals.setdefault(controller_id, None)
        cls._controls.setdefault(controller_id, None)
        if controller_id not in cls._state:
            cls._state[controller_id] = {}
        if robot is not None:
            cls._robots[controller_id] = robot
        cls._init_state(controller_id=controller_id)

    @classmethod
    def unregister(cls, controller_id: str):
        cls._configs.pop(controller_id, None)
        cls._goals.pop(controller_id, None)
        cls._controls.pop(controller_id, None)
        cls._state.pop(controller_id, None)
        cls._robots.pop(controller_id, None)

    @classmethod
    def _init_state(cls, controller_id: str):
        # Subclasses may override
        pass
    
    @classmethod
    def set_goals(cls, controller_id: str, goals):
        cls._goals[controller_id] = goals

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        """
        Process and validate input controller config.
        """
        config = deepcopy(input_config)
        config["dof_idx"] = cb.as_int(config["dof_idx"])
        config["command_input_limits"] = config.get("command_input_limits", "default")
        config["command_output_limits"] = config.get("command_output_limits", "default")

        # Some classmethods (e.g., command_dim / control_type) read from cls._configs.
        # Seed a temporary entry so these methods can run during config processing.
        had_existing_config = controller_id in cls._configs
        if not had_existing_config:
            cls._configs[controller_id] = config

        control_limits = {}
        for motor_type in {"position", "velocity", "effort"}:
            if motor_type not in config["control_limits"]:
                continue
            control_limits[ControlType.get_type(motor_type)] = [
                config["control_limits"][motor_type][0],
                config["control_limits"][motor_type][1],
            ]
        assert "has_limit" in config["control_limits"], "Expected has_limit specified in control_limits, but does not exist."
        control_limits["has_limit"] = config["control_limits"]["has_limit"]
        config["control_limits"] = control_limits
        config["dof_has_limits"] = control_limits["has_limit"]

        # Generate goal information
        config["goal_shapes"] = cls._get_goal_shapes(controller_id)
        config["goal_dim"] = int(sum(cb.prod(cb.array(shape)) for shape in config["goal_shapes"].values()))

        cls._controls[controller_id] = None
        cls._goals[controller_id] = None
        config["command_scale_factor"] = None
        config["command_output_transform"] = None
        config["command_input_transform"] = None

        command_input_limits = config["command_input_limits"]
        command_output_limits = config["command_output_limits"]
        if type(command_input_limits) is str and command_input_limits == "default":
            command_input_limits = cls._generate_default_command_input_limits()
        if type(command_output_limits) is str and command_output_limits == "default":
            command_output_limits = cls._generate_default_command_output_limits(controller_id)

        command_dim = cls.command_dim(controller_id)
        config["command_input_limits"] = (
            None
            if command_input_limits is None
            else (
                cls.nums2array(command_input_limits[0], command_dim),
                cls.nums2array(command_input_limits[1], command_dim),
            )
        )
        config["command_output_limits"] = (
            None
            if command_output_limits is None
            else (
                cls.nums2array(command_output_limits[0], command_dim),
                cls.nums2array(command_output_limits[1], command_dim),
            )
        )

        # Set gains
        isaac_kp = config.get("isaac_kp", None)
        isaac_kd = config.get("isaac_kd", None)
        control_type = cls.control_type(controller_id)
        if control_type == ControlType.POSITION:
            # Set default kp / kd values if not specified
            isaac_kp = m.DEFAULT_ISAAC_KP if isaac_kp is None else isaac_kp
            isaac_kd = m.DEFAULT_ISAAC_KD if isaac_kd is None else isaac_kd
        elif control_type == ControlType.VELOCITY:
            # No kp should be specified, but kd should be
            assert (
                isaac_kp is None
            ),f"Control type for controller {controller_id} is VELOCITY, so no isaac_kp should be set!"
            
            isaac_kd = m.DEFAULT_ISAAC_KP if isaac_kd is None else isaac_kd
        elif control_type == ControlType.EFFORT:
            # Neither kp nor kd should be specified
            assert (
                isaac_kp is None
            ), f"Control type for controller {controller_id} is EFFORT, so no isaac_kp should be set!"
            
            assert (
                isaac_kd is None
            ), f"Control type for controller {controller_id} is EFFORT, so no isaac_kd should be set!"
        else:
            raise ValueError(
                f"Expected control type to be one of: [POSITION, VELOCITY, EFFORT], but got: {control_type}"
            )

        control_dim = cls.control_dim(controller_id)
        config["isaac_kp"] = None if isaac_kp is None else cls.nums2array(isaac_kp, control_dim)
        config["isaac_kd"] = None if isaac_kd is None else cls.nums2array(isaac_kd, control_dim)

        if not had_existing_config:
            cls._configs.pop(controller_id, None)
        
        return config

    @classmethod
    def _generate_default_command_input_limits(cls):
        """
        Generates default command input limits based on the control limits

        Returns:
            2-tuple:
                - int or array: min command input limits
                - int or array: max command input limits
        """
        return (-1.0, 1.0)

    @classmethod
    def _generate_default_command_output_limits(cls, controller_id: str):
        """
        Generates default command output limits based on the control limits

        Returns:
            2-tuple:
                - int or array: min command output limits
                - int or array: max command output limits
        """
        config = cls._configs[controller_id]
        return (
            config["control_limits"][cls.control_type(controller_id)][0][config["dof_idx"]],
            config["control_limits"][cls.control_type(controller_id)][1][config["dof_idx"]],
        )

    @classmethod
    def _preprocess_command(cls, controller_id, command):
        """
        Clips + scales inputted @command according to self.command_input_limits and self.command_output_limits.
        If self.command_input_limits is None, then no clipping will occur. If either self.command_input_limits
        or self.command_output_limits is None, then no scaling will occur.

        Args:
            command (Array[float] or float): Inputted command vector

        Returns:
            Array[float]: Processed command vector
        """
        config = cls._configs[controller_id]
        # Make sure command is a th.tensor
        command = cb.array([command]) if type(command) in {int, float} else command
        # We only clip and / or scale if self.command_input_limits exists
        if config["command_input_limits"] is not None:
            # Clip
            command = command.clip(*config["command_input_limits"])
            if config["command_output_limits"] is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if config["command_scale_factor"] is None:
                    config["command_scale_factor"] = abs(
                        config["command_output_limits"][1] - config["command_output_limits"][0]
                    ) / abs(config["command_input_limits"][1] - config["command_input_limits"][0])
                    config["command_output_transform"] = (
                        config["command_output_limits"][1] + config["command_output_limits"][0]
                    ) / 2.0
                    config["command_input_transform"] = (
                        config["command_input_limits"][1] + config["command_input_limits"][0]
                    ) / 2.0
                # Scale command
                command = (
                    command - config["command_input_transform"]
                ) * config["command_scale_factor"] + config["command_output_transform"]

        # Return processed command
        return command
    
    @classmethod
    def _reverse_preprocess_command(cls, controller_id, processed_command):
        """
        Reverses the scaling operation performed by _preprocess_command.
        Note: This method does not reverse the clipping operation as it's not reversible.

        Args:
            processed_command (th.Tensor[float]): Processed command vector

        Returns:
            th.Tensor[float]: Original command vector (before scaling, clipping not reversed)
        """
        config = cls._configs[controller_id]
        # We only reverse the scaling if both input and output limits exist
        if config["command_input_limits"] is not None and config["command_output_limits"] is not None:
            # If we haven't calculated how to scale the command, do that now (once)
            if config["command_scale_factor"] is None:
                config["command_scale_factor"] = abs(config["command_output_limits"][1] - config["command_output_limits"][0]) / abs(
                    config["command_input_limits"][1] - config["command_input_limits"][0]
                )
                config["command_output_transform"] = (config["command_output_limits"][1] + config["command_output_limits"][0]) / 2.0
                config["command_input_transform"] = (config["command_input_limits"][1] + config["command_input_limits"][0]) / 2.0
            original_command = (
                processed_command - config["command_output_transform"]
            ) / config["command_scale_factor"] + config["command_input_transform"]
        else:
            original_command = processed_command

        return original_command
    
    @classmethod
    def update_goal(cls, controller_id: str, command, control_dict):
        """
        Updates inputted @command internally, writing any necessary internal variables as needed.

        Args:
            command (Array[float]): inputted command to preprocess and extract relevant goal(s) to store
                internally in this controller
            control_dict (dict): Current state
        """
        config = cls._configs[controller_id]
        # Sanity check the command
        assert (
            len(command) == cls.command_dim(controller_id)
        ), f"Commands must be dimension {cls.command_dim(controller_id)}, got dim {len(command)} instead."
        # Preprocess and run internal command
        cls._goals[controller_id] = cls._update_goal(controller_id, cls._preprocess_command(controller_id, command), control_dict)

    @classmethod
    def _update_goal(cls, controller_id, command, control_dict):
        """
        Updates inputted @command internally, writing any necessary internal variables as needed.

        Args:
            command (Array[float]): inputted (preprocessed!) command and extract relevant goal(s) to store
                internally in this controller
            control_dict (dict): Current state

        Returns:
            dict: Keyword-mapped goals to store internally in this controller
        """
        raise NotImplementedError


    @classmethod
    def compute_control(cls, controller_id, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) control signal.
        Should be implemented by subclass.

        Args:
            controller_id: str, use to query cls._goals to get goal_dict (Dict[str, Any]), dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        raise NotImplementedError


    @classmethod
    def clip_control(cls, controller_id, control):
        """
        Clips the inputted @control signal based on @control_limits.

        Args:
            control (Array[float]): control signal to clip

        Returns:
            Array[float]: Clipped control signal
        """
        config = cls._configs[controller_id]
        clipped_control = control.clip(
            config["control_limits"][cls.control_type(controller_id)][0][config["dof_idx"]],
            config["control_limits"][cls.control_type(controller_id)][1][config["dof_idx"]],
        )
        idx = (
            config["dof_has_limits"][config["dof_idx"]]
            if cls.control_type(controller_id) == ControlType.POSITION
            else [True] * cls.control_dim(controller_id)
        )
        control[idx] = clipped_control[idx]
        return control
    
    @classmethod
    def step(cls, controller_id, control_dict):
        """
        Take a controller step.

        Args:
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation

        Returns:
            Array[float]: numpy array of outputted control signals
        """
        config = cls._configs[controller_id]
        # Generate no-op goal if not specified
        if cls._goals[controller_id] is None:
            cls._goals[controller_id] = cls.compute_no_op_goal(controller_id=controller_id, control_dict=control_dict)
        # Compute control, then clip and return
        control = cls.compute_control(controller_id=controller_id, control_dict=control_dict)
        assert (
            len(control) == cls.control_dim(controller_id)
        ), f"Control signal must be of length {cls.control_dim(controller_id)}, got {len(control)} instead."
        cls._controls[controller_id] = cls.clip_control(controller_id, control)
        return cls._controls[controller_id]

    @classmethod
    def reset(cls, controller_id: str):
        """
        Resets this controller. Can be extended by subclass
        """
        cls._goals[controller_id] = None

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        """
        Compute no-op goal given the current state @control_dict

        Args:
            control_dict (dict): Current state

        Returns:
            dict: Maps relevant goal keys (from self._goal_shapes.keys()) to relevant goal data to be used
                in controller computations
        """
        raise NotImplementedError
    
    @classmethod
    def compute_no_op_action(cls, controller_id: str, control_dict):
        """
        Compute a no-op action that updates the goal to match the current position
        Disclaimer: this no-op might cause drift under external load (e.g. when the controller cannot reach the goal position)
        """
        config = cls._configs[controller_id]
        if cls._goals[controller_id] is None:
            cls._goals[controller_id] = cls.compute_no_op_goal(controller_id=controller_id, control_dict=control_dict)
        command = cls._compute_no_op_command(controller_id=controller_id, control_dict=control_dict)
        return cb.to_torch(cls._reverse_preprocess_command(controller_id=controller_id, processed_command=command))

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        """
        Compute no-op command given the goal
        """
        raise NotImplementedError
    

    @classmethod
    def _dump_state(cls, controller_id: str):
        # Default is just the command
        goal = cls._goals[controller_id]
        return dict(
            goal_is_valid=goal is not None,
            goal=None if goal is None else {k: cb.to_torch(v) for k, v in goal.items()},
        )

    @classmethod
    def _load_state(cls, controller_id: str, state):
        # Make sure every entry in goal is a numpy array
        # Load goal
        if state["goal"] is None:
            cls._goals[controller_id] = None
        else:
            goal = dict()
            for name, goal_state in state["goal"].items():
                if isinstance(goal_state, th.Tensor):
                    goal[name] = cb.from_torch(goal_state)
                else:
                    goal[name] = goal_state
            cls._goals[controller_id] = goal
    
    @classmethod
    def dump_state(cls, controller_id: str, serialized=False):
        """
        Dumps the state for a specific controller_id in either dict or serialized form.
        """
        state = cls._dump_state(controller_id=controller_id)
        return cls.serialize(controller_id=controller_id, state=state) if serialized else state

    @classmethod
    def load_state(cls, controller_id: str, state, serialized=False):
        """
        Loads the state for a specific controller_id from dict or serialized form.
        """
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = cls.deserialize(controller_id=controller_id, state=state)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        cls._load_state(controller_id=controller_id, state=state)

    @classmethod
    def serialize(cls, controller_id: str, state):
        # Make sure size of the state is consistent, even if we have no goal
        goal_is_valid = state["goal_is_valid"]
        goal_state_flattened = (
            th.cat([goal_state.flatten() for goal_state in state["goal"].values()])
            if goal_is_valid
            else th.zeros(cls._configs[controller_id]["goal_dim"])
        )
        return th.cat([th.tensor([goal_is_valid]), goal_state_flattened])

    @classmethod
    def deserialize(cls, controller_id: str, state):
        goal_is_valid = bool(state[0])
        if goal_is_valid:
            # Un-flatten all the keys
            idx = 1
            goal = dict()
            for key, shape in cls._configs[controller_id]["goal_shapes"].items():
                length = math.prod(shape)
                goal[key] = state[idx : idx + length].reshape(shape)
                idx += length
        else:
            goal = None
        state_dict = dict(goal_is_valid=goal_is_valid, goal=goal)
        return state_dict, cls._configs[controller_id]["goal_dim"] + 1


    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        """
        Returns:
            dict: Maps keyword in @self.goal to its corresponding numerical shape. This should be static
                and analytically computed prior to any controller steps being taken
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to

        Returns:
            th.tensor: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to th.tensor and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return (
            nums
            if isinstance(nums, cb.arr_type)
            else cb.array(nums)
            if isinstance(nums, Iterable)
            else cb.ones(dim) * nums
        )
    
    @classmethod
    def state_size(cls, controller_id: str):
        # Default is goal dim + 1 (for whether the goal is valid or not)
        return cls.goal_dim(controller_id) + 1
    
    @classmethod
    def goal(cls, controller_id: str):
        """
        Returns:
            dict: Current goal for this controller. Maps relevant goal keys to goal values to be
                used during controller step computations
        """
        return cls._goals[controller_id]
    
    @classmethod
    def goal_dim(cls, controller_id: str):
        return cls._configs[controller_id]["goal_dim"]
    
    @classmethod
    def control(cls, controller_id: str):
        """
        Returns:
            n-array: Array of most recent controls deployed by this controller
        """
        return cls._controls[controller_id]
    
    @classmethod
    def control_freq(cls, controller_id: str):
        """
        Returns:
            float: Control frequency (Hz) of this controller
        """
        return cls._configs[controller_id]["control_freq"]
    
    @classmethod
    def control_dim(cls, controller_id: str):
        """
        Returns:
            int: Expected size of outputted controls
        """
        return len(cls._configs[controller_id]["dof_idx"])
    @classmethod
    def control_type(cls, controller_id: str):
        """
        Returns:
            ControlType: Type of control returned by this controller
        """
        raise NotImplementedError
    
    @classmethod
    def isaac_kp(cls, controller_id: str):
        """
        Returns:
            None or Array[float]: Stiffness gains that should be applied to the underlying Isaac joint motors.
                None if not specified.
        """
        return cls._configs[controller_id]["isaac_kp"]

    @classmethod
    def isaac_kd(cls, controller_id: str):
        """
        Returns:
            None or Array[float]: Stiffness gains that should be applied to the underlying Isaac joint motors.
                None if not specified.
        """
        return cls._configs[controller_id]["isaac_kd"]
    
    @classmethod
    def command_input_limits(cls, controller_id: str):
        """
        Returns:
            None or 2-tuple: If specified, returns (min, max) command input limits for this controller, where
                @min and @max are numpy float arrays of length self.command_dim. Otherwise, returns None
        """
        return cls._configs[controller_id]["command_input_limits"]
    @classmethod
    def command_output_limits(cls, controller_id: str):
        """
        Returns:
            None or 2-tuple: If specified, returns (min, max) command output limits for this controller, where
                @min and @max are numpy float arrays of length self.command_dim. Otherwise, returns None
        """
        return cls._configs[controller_id]["command_output_limits"]

    

    @classmethod
    def command_dim(cls, controller_id: str):
        raise NotImplementedError

    @classmethod
    def dof_idx(cls, controller_id: str):
        """
        Returns:
            Array[int]: DOF indices corresponding to the specific DOFs being controlled by this robot
        """
        return cls._configs[controller_id]["dof_idx"]
    
    @classmethod
    def apply_action(cls, controller_id: str, action):
        """
        Converts inputted actions into low-level control signals

        NOTE: This does NOT deploy control on the object. Use step() instead.

        Args:
            action (n-array): n-DOF length array of actions to apply to this object's internal controllers
        """
        robot = cls._robots[controller_id]
        cls.update_goal(
            controller_id=controller_id,
            command=action,
            control_dict=robot.get_control_dict(),
        )
    
    @classmethod
    def begin_controller_step(cls):
        BaseController._ROBOT_CONTROL_STEP_CACHE.clear()

    @classmethod
    def step_controller_class(cls):
        for controller_id in list(cls._configs.keys()):
            cls._goals.setdefault(controller_id, None)
            cls._controls.setdefault(controller_id, None)
            robot = cls._robots.get(controller_id, None)
            if robot is None:
                continue
            if not robot.control_enabled:
                continue
            if robot._articulation_view_direct is None or not robot._articulation_view_direct.initialized:
                continue

            robot_name = robot.name
            cache = BaseController._ROBOT_CONTROL_STEP_CACHE.get(robot_name, None)
            if cache is None:
                control_dict = robot.get_control_dict()
                u_vec = cb.zeros(robot.n_dof)
                u_type_vec = cb.array([ControlType.EFFORT] * robot.n_dof)
                cache = {
                    "robot": robot,
                    "control_dict": control_dict,
                    "u_vec": u_vec,
                    "u_type_vec": u_type_vec,
                }
                BaseController._ROBOT_CONTROL_STEP_CACHE[robot_name] = cache

            control = cls.step(controller_id=controller_id, control_dict=cache["control_dict"])
            idx = cls.dof_idx(controller_id)
            cache["u_vec"][idx] = control
            cache["u_type_vec"][idx] = cls.control_type(controller_id)

    @classmethod
    def deploy_controller_step(cls):
        for cache in BaseController._ROBOT_CONTROL_STEP_CACHE.values():
            robot = cache["robot"]
            control, control_type = robot._postprocess_control(
                control=cache["u_vec"], control_type=cache["u_type_vec"]
            )
            robot.deploy_control(control=control, control_type=control_type)
        BaseController._ROBOT_CONTROL_STEP_CACHE.clear()
            
