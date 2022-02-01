from abc import ABCMeta, abstractmethod
from collections import Iterable, OrderedDict

import numpy as np

from future.utils import with_metaclass

from igibson.utils.constants import (
    ALL_COLLISION_GROUPS_MASK,
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID
from igibson.utils.usd_utils import get_prim_nested_children
from igibson.prims.articulated_prim import ArticulatedPrim

from typing import Optional, Tuple
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.types import DOFInfo, JointsState, ArticulationAction
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import XFormPrimState
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage
from omni.isaac.core.materials import PreviewSurface, OmniGlass, OmniPBR, VisualMaterial
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    move_prim,
    query_parent_path,
    is_prim_path_valid,
    define_prim,
    get_prim_parent,
    get_prim_object_type,
)
import carb
from omni.isaac.core.utils.stage import get_current_stage


class BaseObject(XFormPrim, metaclass=ABCMeta):
    """This is the interface that all iGibson objects must implement."""

    def __init__(
            self,
            prim_path,
            name=None,
            category="object",
            class_id=None,
            scale=None,
            rendering_params=None,
            visible=True,
            fixed_base=False,
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
        """
        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        # Store values
        self.category = category

        # # TODO
        # # This sets the collision group of the object. In igibson, objects are only permitted to be part of a single
        # # collision group, e.g. the collision group bitvector should only have one bit set to 1.
        # self.collision_group = 1 << (
        #     SPECIAL_COLLISION_GROUPS[self.category]
        #     if self.category in SPECIAL_COLLISION_GROUPS
        #     else DEFAULT_COLLISION_GROUP
        # )

        # category_based_rendering_params = {}
        # if category in ["walls", "floors", "ceilings"]:
        #     category_based_rendering_params["use_pbr"] = False
        #     category_based_rendering_params["use_pbr_mapping"] = False
        # if category == "ceilings":
        #     category_based_rendering_params["shadow_caster"] = False
        #
        # if rendering_params:  # Use the input rendering params as an override.
        #     category_based_rendering_params.update(rendering_params)

        self.scale = np.array(scale) if isinstance(scale, Iterable) else np.array([scale] * 3)
        self.visible = visible
        self.fixed_base = fixed_base

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.renderer_instances = []
        # self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        # self._rendering_params.update(category_based_rendering_params)

        # Get reference to dynamic control interface
        self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()

        # Other values that will be filled in at runtime
        self._applied_visual_material = None
        self._binding_api = None
        self._loaded = False
        self._prim = None
        self._default_state = None
        self._handle = None                         # Handle to this (potentially articulated) prim
        self._root_handle = None                    # Handle to root body in this prim
        self._dofs_infos = None
        self._num_dof = None                        # Number of DOFs in this (potentially articulated) prim
        self._default_joints_state = None
        self._articulation_controller = None        # Low-level "controller" for sending commands to omniverse
        self._links = None
        self._joints = None
        self._mass = None

        # Run some post-loading steps if this prim has already been loaded
        if is_prim_path_valid(prim_path=self._prim_path):
            self._prim = get_prim_at_path(prim_path=self._prim_path)
            self.initialize()
            # Print warning in case the user wanted to fix this object's base
            if self.fixed_base:
                print("WARNING: Cannot create fixed joint for this object, since it was already loaded into the stage"
                      "before this object class was instantiated.")

    def initialize(self):
        """[summary]
        """
        if self._loaded:
            # Don't run this multiple times
            return
        carb.log_info("initializing handles for {}".format(self.prim_path))
        self._handle = self._dc_interface.get_articulation(self.prim_path)
        self._root_handle = self._dc_interface.get_articulation_root_body(self._handle)
        self._num_dof = self._dc_interface.get_articulation_dof_count(self._handle)

        # Add additional DOF info if this is an articulated object
        if self._num_dof > 0:
            self._articulation_controller = ArticulationController()
            self._dofs_infos = OrderedDict()
            # Grab DOF info
            for index in range(self._num_dof):
                dof_handle = self._dc_interface.get_articulation_dof(self._handle, index)
                dof_name = self._dc_interface.get_dof_name(dof_handle)
                # add dof to list
                prim_path = self._dc_interface.get_dof_path(dof_handle)
                self._dofs_infos[dof_name] = DOFInfo(prim_path=prim_path, handle=dof_handle, prim=self.prim, index=index)
            # Initialize articulation controller
            self._articulation_controller.initialize(self._handle, self._dofs_infos)
            # Default joints state is the info from the USD
            default_actions = self._articulation_controller.get_applied_action()
            self._default_joints_state = JointsState(
                positions=np.array(default_actions.joint_positions),
                velocities=np.array(default_actions.joint_velocities),
                efforts=np.zeros_like(default_actions.joint_positions),
            )

        # Set the default root state
        default_position, default_orientation = self.get_world_pose()
        self._default_state = XFormPrimState(position=default_position, orientation=default_orientation)
        self.set_visibility(visible=self.visible)

        # This object is now fully loaded
        self._loaded = True

    def _setup_references(self):
        """
        Parse this object's articulation hierarchy to get properties including joint information and mass
        """
        self._links, self._joints, self._mass = OrderedDict(), OrderedDict(), 0.0

        # Grab all nested children for this object
        prims = get_prim_nested_children(prim=self._prim)

        # Iterate over all found prims and grab any that are links and joints
        for prim in prims:
            properties = prim.GetPropertyNames()
            name = prim.GetName()
            # This is a link if it has the "mass" property
            if "physics:mass" in properties:
                self._links[name] =

    def load(self, simulator):
        """
        Load object into omniverse simulator and return loaded prim reference

        :return Usd.Prim: Prim object loaded into the simulator
        """
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._prim = self._load(simulator)

        # Initialize this prim
        self.initialize()

        # Add fixed joint if we're fixing the base
        if self.fixed_base:
            # Create fixed joint, and set Body0 to be this object's root prim
            simulator.create_joint(
                prim_path=f"{self._prim_path}/rootJoint",
                joint_type="FixedJoint",
                body0=self._dc_interface.get_rigid_body_path(self._root_handle),
            )

        # TODO
        # # Set the collision groups.
        # for body_id in self._body_ids:
        #     for link_id in [-1] + list(range(p.getNumJoints(body_id))):
        #         p.setCollisionFilterGroupMask(body_id, link_id, self.collision_group, ALL_COLLISION_GROUPS_MASK)


        return self._prim

    @property
    def loaded(self):
        return self._loaded

    # def get_body_ids(self):
    #     """
    #     Gets the body IDs belonging to this object.
    #     """
    #     return self._body_ids

    @abstractmethod
    def _load(self, simulator):
        pass

    # TODO : Move to UsdPrim <-- XFormPrim <-- BasePrim
    # # If simulator is None, we grab the current stage. Otherwise we grab the stage associated with @simulator
    # stage = get_current_stage() if simulator is None else simulator.stage

    # @property
    # def prim_path(self) -> str:
    #     """
    #     Returns:
    #         str: prim path in the stage.
    #     """
    #     return self._prim_path
    #
    # @property
    # def name(self):
    #     """
    #     Returns:
    #         str: name given to the prim when instantiating it. Otherwise None.
    #     """
    #     return self._name
    #
    # @property
    # def prim(self):
    #     """
    #     Returns:
    #         Usd.Prim: USD Prim object that this object holds.
    #     """
    #     return self._prim
    #
    # def _set_xform_properties(self) -> None:
    #     current_position, current_orientation = self.get_world_pose()
    #     properties_to_remove = [
    #         "xformOp:rotateX",
    #         "xformOp:rotateXZY",
    #         "xformOp:rotateY",
    #         "xformOp:rotateYXZ",
    #         "xformOp:rotateYZX",
    #         "xformOp:rotateZ",
    #         "xformOp:rotateZYX",
    #         "xformOp:rotateZXY",
    #         "xformOp:rotateXYZ",
    #         "xformOp:transform",
    #     ]
    #     prop_names = self.prim.GetPropertyNames()
    #     xformable = UsdGeom.Xformable(self.prim)
    #     xformable.ClearXformOpOrder()
    #     # TODO: wont be able to delete props for non root links on articulated objects
    #     for prop_name in prop_names:
    #         if prop_name in properties_to_remove:
    #             self.prim.RemoveProperty(prop_name)
    #     if "xformOp:scale" not in prop_names:
    #         xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
    #         xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
    #     else:
    #         xform_op_scale = UsdGeom.XformOp(self.prim.GetAttribute("xformOp:scale"))
    #
    #     if "xformOp:translate" not in prop_names:
    #         xform_op_tranlsate = xformable.AddXformOp(
    #             UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
    #         )
    #     else:
    #         xform_op_tranlsate = UsdGeom.XformOp(self.prim.GetAttribute("xformOp:translate"))
    #
    #     if "xformOp:orient" not in prop_names:
    #         xform_op_rot = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    #     else:
    #         xform_op_rot = UsdGeom.XformOp(self.prim.GetAttribute("xformOp:orient"))
    #     xformable.SetXformOpOrder([xform_op_tranlsate, xform_op_rot, xform_op_scale])
    #     self.set_world_pose(position=current_position, orientation=current_orientation)
    #     return
    #
    # def set_visibility(self, visible: bool) -> None:
    #     """Sets the visibility of the prim in stage.
    #
    #     Args:
    #         visible (bool): flag to set the visibility of the usd prim in stage.
    #     """
    #     imageable = UsdGeom.Imageable(self.prim)
    #     if visible:
    #         imageable.MakeVisible()
    #     else:
    #         imageable.MakeInvisible()
    #     return
    #
    # def get_visibility(self) -> bool:
    #     """
    #     Returns:
    #         bool: true if the prim is visible in stage. false otherwise.
    #     """
    #     return UsdGeom.Imageable(self.prim).ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible
    #
    # def post_reset(self) -> None:
    #     """Resets the prim to its default state (position and orientation).
    #     """
    #     self.set_world_pose(self._default_state.position, self._default_state.orientation)
    #     return
    #
    # def get_default_state(self) -> XFormPrimState:
    #     """
    #     Returns:
    #         XFormPrimState: returns the default state of the prim (position and orientation) that is used after each reset.
    #     """
    #     return self._default_state
    #
    # def set_default_state(
    #     self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None
    # ) -> None:
    #     """Sets the default state of the prim (position and orientation), that will be used after each reset.
    #
    #     Args:
    #         position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
    #                                                    Defaults to None, which means left unchanged.
    #         orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
    #                                                       quaternion is scalar-first (w, x, y, z). shape is (4, ).
    #                                                       Defaults to None, which means left unchanged.
    #     """
    #
    #     if position is not None:
    #         self._default_state.position = position
    #     if orientation is not None:
    #         self._default_state.orientation = orientation
    #     return
    #
    # def apply_visual_material(self, visual_material: VisualMaterial, weaker_than_descendants: bool = False) -> None:
    #     """Used to apply visual material to the held prim and optionally its descendants.
    #
    #     Args:
    #         visual_material (VisualMaterial): visual material to be applied to the held prim. Currently supports
    #                                           PreviewSurface, OmniPBR and OmniGlass.
    #         weaker_than_descendants (bool, optional): True if the material shouldn't override the descendants
    #                                                   materials, otherwise False. Defaults to False.
    #     """
    #     if self._binding_api is None:
    #         if self._prim.HasAPI(UsdShade.MaterialBindingAPI):
    #             self._binding_api = UsdShade.MaterialBindingAPI(self.prim)
    #         else:
    #             self._binding_api = UsdShade.MaterialBindingAPI.Apply(self.prim)
    #     if weaker_than_descendants:
    #         self._binding_api.Bind(visual_material.material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)
    #     else:
    #         self._binding_api.Bind(visual_material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    #     self._applied_visual_material = visual_material
    #     return
    #
    # def get_applied_visual_material(self) -> VisualMaterial:
    #     """Returns the current applied visual material in case it was applied using apply_visual_material OR
    #        it's one of the following materials that was already applied before: PreviewSurface, OmniPBR and OmniGlass.
    #
    #     Returns:
    #         VisualMaterial: the current applied visual material if its type is currently supported.
    #     """
    #     if self._binding_api is None:
    #         if self._prim.HasAPI(UsdShade.MaterialBindingAPI):
    #             self._binding_api = UsdShade.MaterialBindingAPI(self.prim)
    #         else:
    #             self._binding_api = UsdShade.MaterialBindingAPI.Apply(self.prim)
    #     if self._applied_visual_material is not None:
    #         return self._applied_visual_material
    #     else:
    #         visual_binding = self._binding_api.GetDirectBinding()
    #         material_path = str(visual_binding.GetMaterialPath())
    #         if material_path == "":
    #             return None
    #         else:
    #             stage = get_current_stage()
    #             material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    #             # getting the shader
    #             shader_info = material.ComputeSurfaceSource()
    #             if shader_info[0].GetPath() != "":
    #                 shader = shader_info[0]
    #             elif is_prim_path_valid(material_path + "/shader"):
    #                 shader_path = material_path + "/shader"
    #                 shader = UsdShade.Shader(get_prim_at_path(shader_path))
    #             elif is_prim_path_valid(material_path + "/Shader"):
    #                 shader_path = material_path + "/Shader"
    #                 shader = UsdShade.Shader(get_prim_at_path(shader_path))
    #             else:
    #                 carb.log_warn("the shader on xform prim {} is not supported".format(self.prim_path))
    #                 return None
    #             implementation_source = shader.GetImplementationSource()
    #             asset_sub_identifier = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
    #             shader_id = shader.GetShaderId()
    #             if implementation_source == "id" and shader_id == "UsdPreviewSurface":
    #                 self._applied_visual_material = PreviewSurface(prim_path=material_path, shader=shader)
    #                 return self._applied_visual_material
    #             elif asset_sub_identifier == "OmniGlass":
    #                 self._applied_visual_material = OmniGlass(prim_path=material_path, shader=shader)
    #                 return self._applied_visual_material
    #             elif asset_sub_identifier == "OmniPBR":
    #                 self._applied_visual_material = OmniPBR(prim_path=material_path, shader=shader)
    #                 return self._applied_visual_material
    #             else:
    #                 carb.log_warn("the shader on xform prim {} is not supported".format(self.prim_path))
    #                 return None
    #
    # def is_visual_material_applied(self) -> bool:
    #     """
    #     Returns:
    #         bool: True if there is a visual material applied. False otherwise.
    #     """
    #     if self._binding_api is None:
    #         if self._prim.HasAPI(UsdShade.MaterialBindingAPI):
    #             self._binding_api = UsdShade.MaterialBindingAPI(self.prim)
    #         else:
    #             self._binding_api = UsdShade.MaterialBindingAPI.Apply(self.prim)
    #     visual_binding = self._binding_api.GetDirectBinding()
    #     material_path = str(visual_binding.GetMaterialPath())
    #     if material_path == "":
    #         return False
    #     else:
    #         return True
    #
    # def set_world_pose(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
    #     """Sets prim's pose with respect to the world's frame.
    #
    #     Args:
    #         position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
    #                                                    Defaults to None, which means left unchanged.
    #         orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
    #                                                       quaternion is scalar-first (w, x, y, z). shape is (4, ).
    #                                                       Defaults to None, which means left unchanged.
    #     """
    #     current_position, current_orientation = XFormPrim.get_world_pose(self)
    #     if position is None:
    #         position = current_position
    #     if orientation is None:
    #         orientation = current_orientation
    #     my_world_transform = tf_matrix_from_pose(translation=position, orientation=orientation)
    #     parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
    #         Usd.TimeCode.Default()
    #     )
    #     local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
    #     transform = Gf.Transform()
    #     transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
    #     calculated_translation = transform.GetTranslation()
    #     calculated_orientation = transform.GetRotation().GetQuat()
    #     XFormPrim.set_local_pose(
    #         self, translation=np.array(calculated_translation), orientation=gf_quat_to_np_array(calculated_orientation)
    #     )
    #     return
    #
    # def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Gets prim's pose with respect to the world's frame.
    #
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: first index is position in the world frame of the prim. shape is (3, ).
    #                                        second index is quaternion orientation in the world frame of the prim.
    #                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
    #     """
    #     prim_tf = UsdGeom.Xformable(self._prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    #     transform = Gf.Transform()
    #     transform.SetMatrix(prim_tf)
    #     position = transform.GetTranslation()
    #     orientation = transform.GetRotation().GetQuat()
    #     return np.array(position), gf_quat_to_np_array(orientation)
    #
    # def get_local_pose(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Gets prim's pose with respect to the local frame (the prim's parent frame).
    #
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: first index is position in the local frame of the prim. shape is (3, ).
    #                                        second index is quaternion orientation in the local frame of the prim.
    #                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
    #     """
    #     xform_translate_op = self.prim.GetAttribute("xformOp:translate")
    #     xform_orient_op = self.prim.GetAttribute("xformOp:orient")
    #     return np.array(xform_translate_op.Get()), gf_quat_to_np_array(xform_orient_op.Get())
    #
    # def set_local_pose(
    #     self, translation: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None
    # ) -> None:
    #     """Sets prim's pose with respect to the local frame (the prim's parent frame).
    #
    #     Args:
    #         translation (Optional[np.ndarray], optional): translation in the local frame of the prim
    #                                                       (with respect to its parent prim). shape is (3, ).
    #                                                       Defaults to None, which means left unchanged.
    #         orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
    #                                                       quaternion is scalar-first (w, x, y, z). shape is (4, ).
    #                                                       Defaults to None, which means left unchanged.
    #     """
    #     properties = self.prim.GetPropertyNames()
    #     if translation is not None:
    #         translation = Gf.Vec3d(*translation.tolist())
    #         if "xformOp:translate" not in properties:
    #             carb.log_error(
    #                 "Translate property needs to be set for {} before setting its position".format(self.name)
    #             )
    #         xform_op = self.prim.GetAttribute("xformOp:translate")
    #         xform_op.Set(translation)
    #     if orientation is not None:
    #         if "xformOp:orient" not in properties:
    #             carb.log_error(
    #                 "Orient property needs to be set for {} before setting its orientation".format(self.name)
    #             )
    #         xform_op = self.prim.GetAttribute("xformOp:orient")
    #         if xform_op.GetTypeName() == "quatf":
    #             rotq = Gf.Quatf(*orientation.tolist())
    #         else:
    #             rotq = Gf.Quatd(*orientation.tolist())
    #         xform_op.Set(rotq)
    #     return
    #
    # def get_world_scale(self) -> np.ndarray:
    #     """Gets prim's scale with respect to the world's frame.
    #
    #     Returns:
    #         np.ndarray: scale applied to the prim's dimensions in the world frame. shape is (3, ).
    #     """
    #     prim_tf = UsdGeom.Xformable(self._prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    #     transform = Gf.Transform()
    #     transform.SetMatrix(prim_tf)
    #     return np.array(transform.GetScale())
    #
    # def set_local_scale(self, scale: Optional[np.ndarray]) -> None:
    #     """Sets prim's scale with respect to the local frame (the prim's parent frame).
    #
    #     Args:
    #         scale (Optional[np.ndarray]): scale to be applied to the prim's dimensions. shape is (3, ).
    #                                       Defaults to None, which means left unchanged.
    #     """
    #     scale = Gf.Vec3d(*scale.tolist())
    #     properties = self.prim.GetPropertyNames()
    #     if "xformOp:scale" not in properties:
    #         carb.log_error("Scale property needs to be set for {} before setting its scale".format(self.name))
    #     xform_op = self.prim.GetAttribute("xformOp:scale")
    #     xform_op.Set(scale)
    #     return
    #
    # def get_local_scale(self) -> np.ndarray:
    #     """Gets prim's scale with respect to the local frame (the parent's frame).
    #
    #     Returns:
    #         np.ndarray: scale applied to the prim's dimensions in the local frame. shape is (3, ).
    #     """
    #     xform_op = self.prim.GetAttribute("xformOp:scale")
    #     return np.array(xform_op.Get())
    #
    # def is_valid(self) -> bool:
    #     """
    #     Returns:
    #         bool: True is the current prim path corresponds to a valid prim in stage. False otherwise.
    #     """
    #     return is_prim_path_valid(self.prim_path)
    #
    # def change_prim_path(self, new_prim_path: str) -> None:
    #     """Moves prim from the old path to a new one.
    #
    #     Args:
    #         new_prim_path (str): new path of the prim to be moved to.
    #     """
    #     move_prim(path_from=self.prim_path, path_to=new_prim_path)
    #     self._prim_path = new_prim_path
    #     self._prim = get_prim_at_path(self._prim_path)
    #     return

    def get_position(self):
        """Get object position in the format of Array[x, y, z]"""
        return self.get_world_pose()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_world_pose()[1][[1, 2, 3, 0]]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        pos, quat = self.get_world_pose()
        return pos, quat[[1, 2, 3, 0]]

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        self.set_world_pose(position=pos)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        self.set_world_pose(orientation=orn[[3, 0, 1, 2]])

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        self.set_world_pose(position=pos, orientation=orn[[3, 0, 1, 2]])

    # def set_base_link_position_orientation(self, pos, orn):
    #     """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
    #     dynamics_info = p.getDynamicsInfo(self.get_body_ids()[0], -1)
    #     inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
    #     pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
    #     self.set_position_orientation(pos, orn)

    def get_velocities(self):
        """Get this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        return self.get_linear_velocity(), self.get_angular_velocity()

    def set_velocities(self, velocities):
        """Set this object's root body velocity in the format of Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]"""
        lin_vel, ang_vel = velocities
        self.set_linear_velocity(velocity=lin_vel)
        self.set_angular_velocity(velocity=ang_vel)

    def set_joint_states(self, joint_states):
        """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
        # Make sure this object is articulated
        assert self._num_dof > 0, "Can only set joint states for objects that have > 0 DOF!"
        pos = np.zeros(self._num_dof)
        vel = np.zeros(self._num_dof)
        for i, joint_name in enumerate(self._dofs_infos.keys()):
            pos[i], vel[i] = joint_states[joint_name]

        # Set the joint positions and velocities
        self.set_joint_positions(positions=pos)
        self.set_joint_velocities(velocities=vel)

    def get_joint_states(self):
        """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
        # Make sure this object is articulated
        assert self._num_dof > 0, "Can only get joint states for objects that have > 0 DOF!"
        pos = self.get_joint_positions()
        vel = self.get_joint_velocities()
        joint_states = dict()
        for i, joint_name in enumerate(self._dofs_infos.keys()):
            joint_states[joint_name] = (pos[i], vel[i])

        return joint_states

    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return None

    def load_state(self, dump):
        """Load the state of the object other than what's not included in pybullet state."""
        return

    def highlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(True)

    def unhighlight(self):
        for instance in self.renderer_instances:
            instance.set_highlight(False)

    def force_wakeup(self):
        """
        Force wakeup sleeping objects
        """
        self._dc_interface.wake_up_articulation(self._handle)




class Link:
    """
    Body part (link) of object
    """

    def __init__(self, obj, link_prim):
        """
        :param obj: BaseObject, the object this link belongs to.
        :param link_prim: Usd.Prim, prim object corresponding to this link
        """
        # Store args and initialize state
        self.obj = obj
        self.prim = link_prim
        self._dc = obj.dc_interface
        self.handle = self._dc.get_rigid_body(get_prim_path(self.prim))
        self.initial_pos, self.initial_quat = self.get_position_orientation()

    def get_name(self):
        """
        Get name of this link
        """
        return self.prim.GetName()

    def get_position_orientation(self):
        """
        Get pose of this link
        :return Tuple[Array[float], Array[float]]: pos (x,y,z) cartesian coordinates, quat (x,y,z,w)
            orientation in quaternion form of this link
        """
        if self.link_id == -1:
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
        else:
            _, _, _, _, pos, quat = p.getLinkState(self.body_id, self.link_id)
        return np.array(pos), np.array(quat)

    def get_position(self):
        """
        :return Array[float]: (x,y,z) cartesian coordinates of this link
        """
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """
        :return Array[float]: (x,y,z,w) orientation in quaternion form of this link
        """
        return self.get_position_orientation()[1]

    def get_local_position_orientation(self):
        """
        Get pose of this link in the robot's base frame.
        :return Tuple[Array[float], Array[float]]: pos (x,y,z) cartesian coordinates, quat (x,y,z,w)
            orientation in quaternion form of this link
        """
        base = self.robot.base_link
        return p.multiplyTransforms(
            *p.invertTransform(*base.get_position_orientation()), *self.get_position_orientation()
        )

    def get_rpy(self):
        """
        :return Array[float]: (r,p,y) orientation in euler form of this link
        """
        return np.array(p.getEulerFromQuaternion(self.get_orientation()))

    def set_position(self, pos):
        """
        Sets the link's position
        :param pos: Array[float], corresponding to (x,y,z) cartesian coordinates to set
        """
        old_quat = self.get_orientation()
        self.set_position_orientation(pos, old_quat)

    def set_orientation(self, quat):
        """
        Set the link's global orientation
        :param quat: Array[float], corresponding to (x,y,z,w) quaternion orientation to set
        """
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, quat)

    def set_position_orientation(self, pos, quat):
        """
        Set model's global position and orientation. Note: only supported if this is the base link (ID = -1!)
        :param pos: Array[float], corresponding to (x,y,z) global cartesian coordinates to set
        :param quat: Array[float], corresponding to (x,y,z,w) global quaternion orientation to set
        """
        assert self.link_id == -1, "Can only set pose for a base link (id = -1)! Got link id: {}.".format(self.link_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, quat)

    def get_velocity(self):
        """
        Get velocity of this link
        :return Tuple[Array[float], Array[float]]: linear (x,y,z) velocity, angular (ax,ay,az)
            velocity of this link
        """
        if self.link_id == -1:
            lin, ang = p.getBaseVelocity(self.body_id)
        else:
            _, _, _, _, _, _, lin, ang = p.getLinkState(self.body_id, self.link_id, computeLinkVelocity=1)
        return np.array(lin), np.array(ang)

    def get_linear_velocity(self):
        """
        Get linear velocity of this link
        :return Array[float]: linear (x,y,z) velocity of this link
        """
        return self.get_velocity()[0]

    def get_angular_velocity(self):
        """
        Get angular velocity of this link
        :return Array[float]: angular (ax,ay,az) velocity of this link
        """
        return self.get_velocity()[1]

    def contact_list(self):
        """
        Get contact points of the body part
        :return Array[ContactPoints]: list of contact points seen by this link
        """
        return p.getContactPoints(self.body_id, -1, self.link_id, -1)

    def force_wakeup(self):
        """
        Forces a wakeup for this robot. Defaults to no-op.
        """
        p.changeDynamics(self.body_id, self.link_id, activationState=p.ACTIVATION_STATE_WAKE_UP)


class RobotJoint(with_metaclass(ABCMeta, object)):
    """
    Joint of a robot
    """

    @property
    @abstractmethod
    def joint_name(self):
        pass

    @property
    @abstractmethod
    def joint_type(self):
        pass

    @property
    @abstractmethod
    def lower_limit(self):
        pass

    @property
    @abstractmethod
    def upper_limit(self):
        pass

    @property
    @abstractmethod
    def max_velocity(self):
        pass

    @property
    @abstractmethod
    def max_torque(self):
        pass

    @property
    @abstractmethod
    def damping(self):
        pass

    @abstractmethod
    def get_state(self):
        """
        Get the current state of the joint
        :return Tuple[float, float, float]: (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        pass

    @abstractmethod
    def get_relative_state(self):
        """
        Get the normalized current state of the joint
        :return Tuple[float, float, float]: Normalized (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        pass

    @abstractmethod
    def set_pos(self, pos):
        """
        Set position of joint (in metric space)
        :param pos: float, desired position for this joint, in metric space
        """
        pass

    @abstractmethod
    def set_vel(self, vel):
        """
        Set velocity of joint (in metric space)
        :param vel: float, desired velocity for this joint, in metric space
        """
        pass

    @abstractmethod
    def set_torque(self, torque):
        """
        Set torque of joint (in metric space)
        :param torque: float, desired torque for this joint, in metric space
        """
        pass

    @abstractmethod
    def reset_state(self, pos, vel):
        """
        Reset pos and vel of joint in metric space
        :param pos: float, desired position for this joint, in metric space
        :param vel: float, desired velocity for this joint, in metric space
        """
        pass

    @property
    def has_limit(self):
        """
        :return bool: True if this joint has a limit, else False
        """
        return self.lower_limit < self.upper_limit


class PhysicalJoint(RobotJoint):
    """
    A robot joint that exists in the physics simulation (e.g. in pybullet).
    """

    def __init__(self, joint_name, joint_id, body_id):
        """
        :param joint_name: str, name of the joint corresponding to @joint_id
        :param joint_id: int, ID of this joint within the joint(s) found in the body corresponding to @body_id
        :param body_id: Robot body ID containing this link
        """
        # Store args and initialize state
        self._joint_name = joint_name
        self.joint_id = joint_id
        self.body_id = body_id

        # read joint type and joint limit from the URDF file
        # lower_limit, upper_limit, max_velocity, max_torque = <limit lower=... upper=... velocity=... effort=.../>
        # "effort" is approximately torque (revolute) / force (prismatic), but not exactly (ref: http://wiki.ros.org/pr2_controller_manager/safety_limits).
        # if <limit /> does not exist, the following will be the default value
        # lower_limit, upper_limit, max_velocity, max_torque = 0.0, -1.0, 0.0, 0.0
        info = get_joint_info(self.body_id, self.joint_id)
        self._joint_type = info.jointType
        self._lower_limit = info.jointLowerLimit
        self._upper_limit = info.jointUpperLimit
        self._max_torque = info.jointMaxForce
        self._max_velocity = info.jointMaxVelocity
        self._damping = info.jointDamping

        # if joint torque and velocity limits cannot be found in the model file, set a default value for them
        if self._max_torque == 0.0:
            self._max_torque = 100.0
        if self._max_velocity == 0.0:
            # if max_velocity and joint limit are missing for a revolute joint,
            # it's likely to be a wheel joint and a high max_velocity is usually supported.
            self._max_velocity = 15.0 if self._joint_type == p.JOINT_REVOLUTE and not self.has_limit else 1.0

    @property
    def joint_name(self):
        return self._joint_name

    @property
    def joint_type(self):
        return self._joint_type

    @property
    def lower_limit(self):
        return self._lower_limit

    @property
    def upper_limit(self):
        return self._upper_limit

    @property
    def max_velocity(self):
        return self._max_velocity

    @property
    def max_torque(self):
        return self._max_torque

    @property
    def damping(self):
        return self._damping

    def __str__(self):
        return "idx: {}, name: {}".format(self.joint_id, self.joint_name)

    def get_state(self):
        """
        Get the current state of the joint
        :return Tuple[float, float, float]: (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        x, vx, _, trq = p.getJointState(self.body_id, self.joint_id)
        return x, vx, trq

    def get_relative_state(self):
        """
        Get the normalized current state of the joint
        :return Tuple[float, float, float]: Normalized (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        pos, vel, trq = self.get_state()

        # normalize position to [-1, 1]
        if self.has_limit:
            mean = (self.lower_limit + self.upper_limit) / 2.0
            magnitude = (self.upper_limit - self.lower_limit) / 2.0
            pos = (pos - mean) / magnitude

        # (trying to) normalize velocity to [-1, 1]
        vel /= self.max_velocity

        # (trying to) normalize torque / force to [-1, 1]
        trq /= self.max_torque

        return pos, vel, trq

    def set_pos(self, pos):
        """
        Set position of joint (in metric space)
        :param pos: float, desired position for this joint, in metric space
        """
        if self.has_limit:
            pos = np.clip(pos, self.lower_limit, self.upper_limit)
        p.setJointMotorControl2(self.body_id, self.joint_id, p.POSITION_CONTROL, targetPosition=pos)

    def set_vel(self, vel):
        """
        Set velocity of joint (in metric space)
        :param vel: float, desired velocity for this joint, in metric space
        """
        vel = np.clip(vel, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.body_id, self.joint_id, p.VELOCITY_CONTROL, targetVelocity=vel)

    def set_torque(self, torque):
        """
        Set torque of joint (in metric space)
        :param torque: float, desired torque for this joint, in metric space
        """
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        p.setJointMotorControl2(
            bodyIndex=self.body_id,
            jointIndex=self.joint_id,
            controlMode=p.TORQUE_CONTROL,
            force=torque,
        )

    def reset_state(self, pos, vel):
        """
        Reset pos and vel of joint in metric space
        :param pos: float, desired position for this joint, in metric space
        :param vel: float, desired velocity for this joint, in metric space
        """
        p.resetJointState(self.body_id, self.joint_id, targetValue=pos, targetVelocity=vel)
        self.disable_motor()

    def disable_motor(self):
        """
        Disable the motor of this joint
        """
        p.setJointMotorControl2(
            self.body_id,
            self.joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0,
        )


class VirtualJoint(RobotJoint):
    """A virtual joint connecting two bodies of the same robot that does not exist in the physics simulation.
    Such a joint must be handled manually by the owning robot class by providing the appropriate callback functions
    for getting and setting joint positions.
    Such a joint can also be used as a way of controlling an arbitrary non-joint mechanism on the robot.
    """

    def __init__(self, joint_name, joint_type, get_pos_callback, set_pos_callback, lower_limit=None, upper_limit=None):
        self._joint_name = joint_name

        assert joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        self._joint_type = joint_type

        self._get_pos_callback = get_pos_callback
        self._set_pos_callback = set_pos_callback

        self._lower_limit = lower_limit if lower_limit is not None else 0
        self._upper_limit = upper_limit if upper_limit is not None else -1

    @property
    def joint_name(self):
        return self._joint_name

    @property
    def joint_type(self):
        return self._joint_type

    @property
    def lower_limit(self):
        return self._lower_limit

    @property
    def upper_limit(self):
        return self._upper_limit

    @property
    def max_velocity(self):
        raise NotImplementedError("This feature is not available for virtual joints.")

    @property
    def max_torque(self):
        raise NotImplementedError("This feature is not available for virtual joints.")

    @property
    def damping(self):
        raise NotImplementedError("This feature is not available for virtual joints.")

    def get_state(self):
        return self._get_pos_callback()

    def get_relative_state(self):
        pos, _, _ = self.get_state()

        # normalize position to [-1, 1]
        if self.has_limit:
            mean = (self.lower_limit + self.upper_limit) / 2.0
            magnitude = (self.upper_limit - self.lower_limit) / 2.0
            pos = (pos - mean) / magnitude

        return pos, None, None

    def set_pos(self, pos):
        self._set_pos_callback(pos)

    def set_vel(self, vel):
        raise NotImplementedError("This feature is not implemented yet for virtual joints.")

    def set_torque(self, torque):
        raise NotImplementedError("This feature is not available for virtual joints.")

    def reset_state(self, pos, vel):
        raise NotImplementedError("This feature is not implemented yet for virtual joints.")

    def __str__(self):
        return "Virtual Joint name: {}".format(self.joint_name)


class Virtual6DOFJoint(object):
    """A wrapper for a floating (e.g. 6DOF) virtual joint between two robot body parts.
    This wrapper generates the 6 separate VirtualJoint instances needed for such a mechanism, and accumulates their
    set_pos calls to provide a single callback with a 6-DOF pose callback. Note that all 6 joints must be set for this
    wrapper to trigger its callback - partial control not allowed.
    """

    COMPONENT_SUFFIXES = ["x", "y", "z", "rx", "ry", "rz"]

    def __init__(self, joint_name, parent_link, child_link, command_callback, lower_limits=None, upper_limits=None):
        self.joint_name = joint_name
        self.parent_link = parent_link
        self.child_link = child_link
        self._command_callback = command_callback

        self._joints = [
            VirtualJoint(
                joint_name="%s_%s" % (self.joint_name, name),
                joint_type=p.JOINT_PRISMATIC if i < 3 else p.JOINT_REVOLUTE,
                get_pos_callback=lambda dof=i: (self.get_state()[dof], None, None),
                set_pos_callback=lambda pos, dof=i: self.set_pos(dof, pos),
                lower_limit=lower_limits[i] if lower_limits is not None else None,
                upper_limit=upper_limits[i] if upper_limits is not None else None,
            )
            for i, name in enumerate(Virtual6DOFJoint.COMPONENT_SUFFIXES)
        ]

        self._reset_stored_control()

    def get_state(self):
        pos, orn = self.child_link.get_position_orientation()

        if self.parent_link is not None:
            pos, orn = p.multiplyTransforms(*p.invertTransform(*self.parent_link.get_position_orientation()), pos, orn)

        # Stack the position and the Euler orientation
        return list(pos) + list(p.getEulerFromQuaternion(orn))

    def get_joints(self):
        """Gets the 1DOF VirtualJoints belonging to this 6DOF joint."""
        return tuple(self._joints)

    def set_pos(self, dof, val):
        """Calls the command callback with values for all 6 DOF once the setter has been called for each of them."""
        self._stored_control[dof] = val

        if all(ctrl is not None for ctrl in self._stored_control):
            self._command_callback(self._stored_control)
            self._reset_stored_control()

    def _reset_stored_control(self):
        self._stored_control = [None] * len(self._joints)