from abc import abstractmethod

import torch as th

from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI


class ArticulatedTrunkRobot(ManipulationRobot):
    """
    ManipulationRobot that is is equipped with an articulated trunk.
    If rigid_trunk is True, the trunk will be rigid and not articulated.
    Otherwise, it will belong to the kinematic chain of the default arm.

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

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        fixed_base=False,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,
        # Unique to ArticulatedTrunkRobot
        rigid_trunk=False,
        default_trunk_offset=0.2,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            disable_grasp_handling (bool): If True, will disable all grasp handling for this object. This means that
                sticky and assisted grasp modes will not work unless the connection/release methodsare manually called.
            rigid_trunk (bool): If True, will prevent the trunk from moving during execution.
            default_trunk_offset (float): The default height of the robot's trunk
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            disable_grasp_handling=disable_grasp_handling,
            **kwargs,
        )

    @property
    def trunk_joint_names(self):
        raise NotImplementedError("trunk_joint_names must be implemented in subclass")

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        return th.tensor([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config
        if self.rigid_trunk:
            return cfg

        # Need to override joint idx being controlled to include trunk in default arm controller configs
        for arm_cfg in cfg[f"arm_{self.default_arm}"].values():
            arm_control_idx = th.cat([self.trunk_control_idx, self.arm_control_idx[self.default_arm]])
            arm_cfg["dof_idx"] = arm_control_idx

            # Need to modify the default joint positions also if this is a null joint controller
            if arm_cfg["name"] == "NullJointController":
                arm_cfg["default_command"] = self.reset_joint_pos[arm_control_idx]

        return cfg

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        joint_positions = ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path)
        joint_velocities = ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path)
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
