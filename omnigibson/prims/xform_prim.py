from collections.abc import Iterable
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.prims.prim_base import BasePrim
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.utils.transform_utils import quat2euler
from omnigibson.utils.usd_utils import PoseAPI
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from omnigibson.macros import gm
import trimesh.transformations

class XFormPrim(BasePrim):
    """
    Provides high level functions to deal with an Xform prim and its attributes/ properties.
    If there is an Xform prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created when self.load(...) is called.

    Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
        unless it is a non-root articulation link.

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. For this xform prim, the below values can be specified:

            scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._binding_api = None
        self._material = None
        self._collision_filter_api = None
        self.original_scale = None

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        return og.sim.stage.DefinePrim(self._prim_path, "Xform")

    def _post_load(self):
        # run super first
        super()._post_load()

        # Make sure all xforms have pose and scaling info
        self._set_xform_properties()

        # Cache the original scale from the USD so that when EntityPrim sets the scale for each link (Rigid/ClothPrim),
        # the new scale is with respect to the original scale. XFormPrim's scale always matches the scale in the USD.
        self.original_scale = np.array(self.get_attribute("xformOp:scale"))

        # Create collision filter API
        self._collision_filter_api = lazy.pxr.UsdPhysics.FilteredPairsAPI(self._prim) if \
            self._prim.HasAPI(lazy.pxr.UsdPhysics.FilteredPairsAPI) else lazy.pxr.UsdPhysics.FilteredPairsAPI.Apply(self._prim)

        # Create binding API
        self._binding_api = lazy.pxr.UsdShade.MaterialBindingAPI(self.prim) if \
            self._prim.HasAPI(lazy.pxr.UsdShade.MaterialBindingAPI) else lazy.pxr.UsdShade.MaterialBindingAPI.Apply(self.prim)

        # Grab the attached material if it exists
        if self.has_material():
            material_prim_path = self._binding_api.GetDirectBinding().GetMaterialPath().pathString
            material_name = f"{self.name}:material"
            material = MaterialPrim.get_material(prim_path=material_prim_path, name=material_name)
            assert material.loaded, f"Material prim path {material_prim_path} doesn't exist on stage."
            material.add_user(self)
            self._material = material

        # Optionally set the scale and visibility
        if "scale" in self._load_config and self._load_config["scale"] is not None:
            self.scale = self._load_config["scale"]

    def remove(self):
        # Remove the material prim if one exists
        if self._material is not None:
            self._material.remove_user(self)

        # Remove the prim
        super().remove()

    def _set_xform_properties(self):
        current_position, current_orientation = self.get_position_orientation()
        properties_to_remove = [
            "xformOp:rotateX",
            "xformOp:rotateXZY",
            "xformOp:rotateY",
            "xformOp:rotateYXZ",
            "xformOp:rotateYZX",
            "xformOp:rotateZ",
            "xformOp:rotateZYX",
            "xformOp:rotateZXY",
            "xformOp:rotateXYZ",
            "xformOp:transform",
        ]
        prop_names = self.prim.GetPropertyNames()
        xformable = lazy.pxr.UsdGeom.Xformable(self.prim)
        xformable.ClearXformOpOrder()
        # TODO: wont be able to delete props for non root links on articulated objects
        for prop_name in prop_names:
            if prop_name in properties_to_remove:
                self.prim.RemoveProperty(prop_name)
        if "xformOp:scale" not in prop_names:
            xform_op_scale = xformable.AddXformOp(lazy.pxr.UsdGeom.XformOp.TypeScale, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
            xform_op_scale.Set(lazy.pxr.Gf.Vec3d([1.0, 1.0, 1.0]))
        else:
            xform_op_scale = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:scale"))

        if "xformOp:translate" not in prop_names:
            xform_op_translate = xformable.AddXformOp(
                lazy.pxr.UsdGeom.XformOp.TypeTranslate, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, ""
            )
        else:
            xform_op_translate = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:translate"))

        if "xformOp:orient" not in prop_names:
            xform_op_rot = xformable.AddXformOp(lazy.pxr.UsdGeom.XformOp.TypeOrient, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        else:
            xform_op_rot = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:orient"))
        xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

        self.set_position_orientation(position=current_position, orientation=current_orientation)
        new_position, new_orientation = self.get_position_orientation()
        r1 = R.from_quat(current_orientation).as_matrix()
        r2 = R.from_quat(new_orientation).as_matrix()
        # Make sure setting is done correctly
        assert np.allclose(new_position, current_position, atol=1e-4) and np.allclose(r1, r2, atol=1e-4), \
            f"{self.prim_path}: old_pos: {current_position}, new_pos: {new_position}, " \
            f"old_orn: {current_orientation}, new_orn: {new_orientation}"

    def has_material(self):
        """
        Returns:
            bool: True if there is a visual material bound to this prim. False otherwise
        """
        material_path = self._binding_api.GetDirectBinding().GetMaterialPath().pathString
        return False if material_path == "" else True

    def set_position_orientation(self, position=None, orientation=None):
        """
        Sets prim's pose with respect to the world frame

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
        """
        current_position, current_orientation = self.get_position_orientation()
        
        position = current_position if position is None else np.array(position, dtype=float)
        orientation = current_orientation if orientation is None else np.array(orientation, dtype=float)
        assert np.isclose(np.linalg.norm(orientation), 1, atol=1e-3), \
            f"{self.prim_path} desired orientation {orientation} is not a unit quaternion."

        my_world_transform = T.pose2mat((position, orientation))

        parent_prim = lazy.omni.isaac.core.utils.prims.get_prim_parent(self._prim)
        parent_path = str(parent_prim.GetPath())
        parent_world_transform = PoseAPI.get_world_pose_with_scale(parent_path)

        local_transform = np.linalg.inv(parent_world_transform) @ my_world_transform
        self.set_local_pose(*T.mat2pose(local_transform))

    def get_position_orientation(self):
        """
        Gets prim's pose with respect to the world's frame.

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position in the world frame
                - 4-array: (x,y,z,w) quaternion orientation in the world frame
        """
        return PoseAPI.get_world_pose(self._prim_path)

    def set_position(self, position):
        """
        Set this prim's position with respect to the world frame

        Args:
            position (3-array): (x,y,z) global cartesian position to set
        """
        self.set_position_orientation(position=position)

    def get_position(self):
        """
        Get this prim's position with respect to the world frame

        Returns:
            3-array: (x,y,z) global cartesian position of this prim
        """
        return self.get_position_orientation()[0]

    def set_orientation(self, orientation):
        """
        Set this prim's orientation with respect to the world frame

        Args:
            orientation (4-array): (x,y,z,w) global quaternion orientation to set
        """
        self.set_position_orientation(orientation=orientation)

    def get_orientation(self):
        """
        Get this prim's orientation with respect to the world frame

        Returns:
            4-array: (x,y,z,w) global quaternion orientation of this prim
        """
        return self.get_position_orientation()[1]

    def get_rpy(self):
        """
        Get this prim's orientation with respect to the world frame

        Returns:
            3-array: (roll, pitch, yaw) global euler orientation of this prim
        """
        return quat2euler(self.get_orientation())
    
    def get_2d_orientation(self):
        """
        Get this prim's orientation on the XY plane of the world frame. This is obtained by
        projecting the forward vector onto the XY plane and then computing the angle.
        """
        fwd = R.from_quat(self.get_orientation()).apply([1, 0, 0])
        fwd[2] = 0.

        # If the object is facing close to straight up, then we can't compute a 2D orientation
        # in that case, we return zero.
        if np.linalg.norm(fwd) < 1e-4:
            return 0.

        fwd /= np.linalg.norm(fwd)
        return np.arctan2(fwd[1], fwd[0])

    def get_local_pose(self):
        """
        Gets prim's pose with respect to the prim's local frame (its parent frame)

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position in the local frame
                - 4-array: (x,y,z,w) quaternion orientation in the local frame
        """
        pos, ori = lazy.omni.isaac.core.utils.xforms.get_local_pose(self.prim_path)
        return pos, ori[[1, 2, 3, 0]]

    def set_local_pose(self, position=None, orientation=None):
        """
        Sets prim's pose with respect to the local frame (the prim's parent frame).

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
        """            
        properties = self.prim.GetPropertyNames()
        if position is not None:
            position = lazy.pxr.Gf.Vec3d(*np.array(position, dtype=float))
            if "xformOp:translate" not in properties:
                lazy.carb.log_error(
                    "Translate property needs to be set for {} before setting its position".format(self.name)
                )
            self.set_attribute("xformOp:translate", position)
        if orientation is not None:
            orientation = np.array(orientation, dtype=float)[[3, 0, 1, 2]]
            if "xformOp:orient" not in properties:
                lazy.carb.log_error(
                    "Orient property needs to be set for {} before setting its orientation".format(self.name)
                )
            xform_op = self._prim.GetAttribute("xformOp:orient")
            if xform_op.GetTypeName() == "quatf":
                rotq = lazy.pxr.Gf.Quatf(*orientation)
            else:
                rotq = lazy.pxr.Gf.Quatd(*orientation)
            xform_op.Set(rotq)
        PoseAPI.invalidate()
        if gm.ENABLE_FLATCACHE:
            # If flatcache is on, make sure the USD local pose is synced to the fabric local pose.
            # Ideally we should call usdrt's set local pose directly, but there is no such API.
            # The only available API is SetLocalXformFromUsd, so we update USD first, and then sync to fabric.
            xformable_prim = lazy.usdrt.Rt.Xformable(lazy.omni.isaac.core.utils.prims.get_prim_at_path(self.prim_path, fabric=True))
            assert not xformable_prim.HasWorldXform(), "Fabric's world pose is set for a non-rigid prim which is unexpected. Please report this."
            xformable_prim.SetLocalXformFromUsd()
        return

    def get_world_scale(self):
        """
        Gets prim's scale with respect to the world's frame.

        Returns:
            np.ndarray: scale applied to the prim's dimensions in the world frame. shape is (3, ).
        """
        prim_tf = lazy.pxr.UsdGeom.Xformable(self._prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        transform = lazy.pxr.Gf.Transform()
        transform.SetMatrix(prim_tf)
        return np.array(transform.GetScale())
    
    @property
    def scaled_transform(self):
        """
        Returns the scaled transform of this prim.
        """
        return PoseAPI.get_world_pose_with_scale(self._prim_path)

    def transform_local_points_to_world(self, points):
        return trimesh.transformations.transform_points(points, self.scaled_transform)

    @property
    def scale(self):
        """
        Gets prim's scale with respect to the local frame (the parent's frame).

        Returns:
            np.ndarray: scale applied to the prim's dimensions in the local frame. shape is (3, ).
        """
        return np.array(self.get_attribute("xformOp:scale"))

    @scale.setter
    def scale(self, scale):
        """
        Sets prim's scale with respect to the local frame (the prim's parent frame).

        Args:
            scale (float or np.ndarray): scale to be applied to the prim's dimensions. shape is (3, ).
                                          Defaults to None, which means left unchanged.
        """
        scale = np.array(scale, dtype=float) if isinstance(scale, Iterable) else np.ones(3) * scale
        scale = lazy.pxr.Gf.Vec3d(*scale)
        properties = self.prim.GetPropertyNames()
        if "xformOp:scale" not in properties:
            lazy.carb.log_error("Scale property needs to be set for {} before setting its scale".format(self.name))
        self.set_attribute("xformOp:scale", scale)

    @property
    def material(self):
        """
        Returns:
            None or MaterialPrim: The bound material to this prim, if there is one
        """
        return self._material

    @material.setter
    def material(self, material):
        """
        Set the material @material for this prim. This will also bind the material to this prim

        Args:
            material (MaterialPrim): Material to bind to this prim
        """
        self._binding_api.Bind(lazy.pxr.UsdShade.Material(material.prim), bindingStrength=lazy.pxr.UsdShade.Tokens.weakerThanDescendants)
        self._material = material

    def add_filtered_collision_pair(self, prim):
        """
        Adds a collision filter pair with another prim

        Args:
            prim (XFormPrim): Another prim to filter collisions with
        """
        # Add to both this prim's and the other prim's filtered pair
        self._collision_filter_api.GetFilteredPairsRel().AddTarget(prim.prim_path)
        prim._collision_filter_api.GetFilteredPairsRel().AddTarget(self._prim_path)

    def remove_filtered_collision_pair(self, prim):
        """
        Removes a collision filter pair with another prim

        Args:
            prim (XFormPrim): Another prim to remove filter collisions with
        """
        # Add to both this prim's and the other prim's filtered pair
        self._collision_filter_api.GetFilteredPairsRel().RemoveTarget(prim.prim_path)
        prim._collision_filter_api.GetFilteredPairsRel().RemoveTarget(self._prim_path)

    def _dump_state(self):
        pos, ori = self.get_position_orientation()
        return dict(pos=pos, ori=ori)

    def _load_state(self, state):
        self.set_position_orientation(np.array(state["pos"]), np.array(state["ori"]))

    def _serialize(self, state):
        return np.concatenate([state["pos"], state["ori"]]).astype(float)

    def _deserialize(self, state):
        # We deserialize deterministically by knowing the order of values -- pos, ori
        return dict(pos=state[0:3], ori=state[3:7]), 7
