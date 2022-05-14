import logging
from abc import ABCMeta
from collections import OrderedDict

from pxr import PhysxSchema, UsdPhysics

from igibson.prims.entity_prim import EntityPrim
from igibson.utils.constants import DEFAULT_COLLISION_GROUP, SPECIAL_COLLISION_GROUPS, SemanticClass
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID
from igibson.utils.usd_utils import BoundingBoxAPI, CollisionAPI, create_joint


class BaseObject(EntityPrim, metaclass=ABCMeta):
    """This is the interface that all iGibson objects must implement."""

    def __init__(
        self,
        prim_path,
        name=None,
        category="object",
        class_id=None,
        scale=1.0,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        **kwargs,
    ):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
            Note that this base object does NOT pass kwargs down into the Prim-type super() classes, and we assume
            that kwargs are only shared between all SUBclasses (children), not SUPERclasses (parents).
        """
        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        # Store values
        self.category = category
        self.fixed_base = fixed_base

        logging.info(f"Category: {self.category}")

        # TODO
        # This sets the collision group of the object. In igibson, objects are only permitted to be part of a single
        # collision group, e.g. collisions are only enabled within a single group
        self.collision_group = SPECIAL_COLLISION_GROUPS.get(self.category, DEFAULT_COLLISION_GROUP)

        # category_based_rendering_params = {}
        # if category in ["walls", "floors", "ceilings"]:
        #     category_based_rendering_params["use_pbr"] = False
        #     category_based_rendering_params["use_pbr_mapping"] = False
        # if category == "ceilings":
        #     category_based_rendering_params["shadow_caster"] = False
        #
        # if rendering_params:  # Use the input rendering params as an override.
        #     category_based_rendering_params.update(rendering_params)

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.renderer_instances = []
        self.rendering_params = rendering_params
        # self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        # self._rendering_params.update(category_based_rendering_params)

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["scale"] = scale
        load_config["visible"] = visible
        load_config["visual_only"] = visual_only
        load_config["self_collisions"] = self_collisions

        # Run super init
        super().__init__(
            prim_path=prim_path, name=name, load_config=load_config, **kwargs,
        )

    def load(self, simulator=None):
        # Run sanity check, any of these objects REQUIRE a simulator to be specified
        assert simulator is not None, "Simulator must be specified for loading any object subclassed from BaseObject!"

        # Run super method
        return super().load(simulator=simulator)

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Set visibility
        if "visible" in self._load_config and self._load_config["visible"] is not None:
            self.visible = self._load_config["visible"]

        # Add fixed joint if we're fixing the base
        print(f"obj {self.name} is fixed base: {self.fixed_base}")
        if self.fixed_base:
            # Create fixed joint, and set Body0 to be this object's root prim
            create_joint(
                prim_path=f"{self._prim_path}/rootJoint", joint_type="FixedJoint", body1=f"{self._prim_path}/base_link",
            )
        else:
            if self._prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                # If we only have a link, remove the articulation root API
                if self.n_links == 1:
                    self._prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    self._prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                else:
                    # We need to fix (change) the articulation root
                    # We have to do something very hacky because omniverse is buggy
                    # Articulation roots mess up the joint order if it's on a non-fixed base robot, e.g. a
                    # mobile manipulator. So if we have to move it to the actual root link of the robot instead.
                    # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
                    # for more info
                    self._prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    self._prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                    UsdPhysics.ArticulationRootAPI.Apply(self.root_prim)
                    PhysxSchema.PhysxArticulationAPI.Apply(self.root_prim)

        # Set self collisions if we have articulation API to set
        if self._prim.HasAPI(UsdPhysics.ArticulationRootAPI) or self.root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self.self_collisions = self._load_config["self_collisions"]

        # TODO: Do we need to explicitly add all links? or is adding articulation root itself sufficient?
        # Set the collision group
        CollisionAPI.add_to_collision_group(
            col_group=self.collision_group, prim_path=self.prim_path, create_if_not_exist=True,
        )

    @property
    def articulation_root_path(self):
        # We override this because omniverse is buggy ):
        # For non-fixed base objects (e.g.: mobile manipulators), using the default articulation root breaks the
        # kinematic chain for some reason. So, the current workaround is to set the articulation root to be the
        # actual base link of the robot instead.
        # See https://forums.developer.nvidia.com/t/inconsistent-values-from-isaacsims-dc-get-joint-parent-child-body/201452/2
        # for more info
        return (
            f"{self._prim_path}/base_link"
            if (not self.fixed_base) and (self.n_links > 1)
            else super().articulation_root_path
        )

    @property
    def bbox(self):
        """
        Get this object's actual bounding box

        Returns:
            3-array: (x,y,z) bounding box
        """
        min_corner, max_corner = BoundingBoxAPI.compute_aabb(self.prim_path)
        return max_corner - min_corner

    @property
    def mass(self):
        """
        Returns:
             float: Cumulative mass of this potentially articulated object.
        """
        mass = 0.0
        for link in self._links.values():
            mass += link.mass

        return mass

    @mass.setter
    def mass(self, mass):
        # Cannot set mass directly for this object!
        raise NotImplementedError("Cannot set mass directly for an object!")

    def get_velocities(self):
        """Get this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        return self.get_linear_velocity(), self.get_angular_velocity()

    def set_velocities(self, velocities):
        """Set this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        lin_vel, ang_vel = velocities

        self.set_linear_velocity(velocity=lin_vel)
        self.set_angular_velocity(velocity=ang_vel)

    # def set_joint_states(self, joint_states):
    #     """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
    #     # Make sure this object is articulated
    #     assert self._n_dof > 0, "Can only set joint states for objects that have > 0 DOF!"
    #     pos = np.zeros(self._n_dof)
    #     vel = np.zeros(self._n_dof)
    #     for i, joint_name in enumerate(self._dofs_infos.keys()):
    #         pos[i], vel[i] = joint_states[joint_name]
    #
    #     # Set the joint positions and velocities
    #     self.set_joint_positions(positions=pos)
    #     self.set_joint_velocities(velocities=vel)
    #
    # def get_joint_states(self):
    #     """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
    #     # Make sure this object is articulated
    #     assert self._n_dof > 0, "Can only get joint states for objects that have > 0 DOF!"
    #     pos = self.get_joint_positions()
    #     vel = self.get_joint_velocities()
    #     joint_states = dict()
    #     for i, joint_name in enumerate(self._dofs_infos.keys()):
    #         joint_states[joint_name] = (pos[i], vel[i])
    #
    #     return joint_states

    # TODO
    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)

    def dump_config(self):
        """
        Dumps relevant configuration for this object.

        Returns:
            OrderedDict: Object configuration.
        """
        return OrderedDict(
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            self_collisions=self.self_collisions,
            rendering_params=self.rendering_params,
        )

    def update(self):
        """
        Runs any relevant updates for this object. This should occur once per simulation step.
        """
        pass
