from abc import ABCMeta, abstractmethod
from collections import Iterable

import numpy as np

from future.utils import with_metaclass

from igibson.utils.constants import (
    ALL_COLLISION_GROUPS_MASK,
    DEFAULT_COLLISION_GROUP,
    SPECIAL_COLLISION_GROUPS,
    SemanticClass,
)
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID

from typing import Optional, Tuple
from pxr import Gf, Usd, UsdGeom, UsdShade
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import XFormPrimState
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


class BaseObject(with_metaclass(ABCMeta, XFormPrim)):
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
        """
        self._prim_path = prim_path

        # Generate a name if necessary. Note that the generation order & set of these names is not deterministic.
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(category, address)

        self._name = name
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

        if class_id is None:
            class_id = CLASS_NAME_TO_CLASS_ID.get(category, SemanticClass.USER_ADDED_OBJS)

        self.class_id = class_id
        self.renderer_instances = []
        # self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        # self._rendering_params.update(category_based_rendering_params)

        # Determine whether this prim was already loaded
        if is_prim_path_valid(prim_path=self._prim_path):
            self._prim = get_prim_at_path(prim_path=self._prim_path)
            self._loaded = True
            default_position, default_orientation = self.get_world_pose()
            self._default_state = XFormPrimState(position=default_position, orientation=default_orientation)
            self.set_visibility(visible=self.visible)
        else:
            self._loaded = False
            self._prim = None
            self._default_state = None

        # Other values that will be filled in at runtime
        self._applied_visual_material = None
        self._binding_api = None

    def load(self, simulator):
        """
        Load object into omniverse simulator and return loaded prim reference
        """
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._loaded = True
        self._prim = self._load(simulator)

        # Set visibility
        self.set_visibility(visible=self.visible)

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
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        assert len(self.get_body_ids()) == 1, "Base implementation only works with single-body objects."
        pos, orn = p.getBasePositionAndOrientation(self.get_body_ids()[0])
        return np.array(pos), np.array(orn)

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        old_orn = self.get_orientation()
        self.set_position_orientation(pos, old_orn)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        assert len(self.get_body_ids()) == 1, "Base implementation only works with single-body objects."
        p.resetBasePositionAndOrientation(self.get_body_ids()[0], pos, orn)

    def set_base_link_position_orientation(self, pos, orn):
        """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        dynamics_info = p.getDynamicsInfo(self.get_body_ids()[0], -1)
        inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
        pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
        self.set_position_orientation(pos, orn)

    def get_velocities(self):
        """Get object bodies' velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        velocities = []
        for body_id in self.get_body_ids():
            lin, ang = p.getBaseVelocity(body_id)
            velocities.append((np.array(lin), np.array(ang)))

        return velocities

    def set_velocities(self, velocities):
        """Set object base velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        assert len(velocities) == len(self.get_body_ids()), "Number of velocities should match number of bodies."

        for bid, (linear_velocity, angular_velocity) in zip(self.get_body_ids(), velocities):
            p.resetBaseVelocity(bid, linear_velocity, angular_velocity)

    def set_joint_states(self, joint_states):
        """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
        for body_id in self.get_body_ids():
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                joint_type = info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_name = info[1].decode("UTF-8")
                    joint_position, joint_velocity = joint_states[joint_name]
                    p.resetJointState(body_id, j, joint_position, targetVelocity=joint_velocity)

    def get_joint_states(self):
        """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
        joint_states = {}
        for body_id in self.get_body_ids():
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                joint_type = info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_name = info[1].decode("UTF-8")
                    joint_states[joint_name] = p.getJointState(body_id, j)[:2]
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
        for body_id in self.get_body_ids():
            for joint_id in range(p.getNumJoints(body_id)):
                p.changeDynamics(body_id, joint_id, activationState=p.ACTIVATION_STATE_WAKE_UP)
            p.changeDynamics(body_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
