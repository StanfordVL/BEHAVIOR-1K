"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""
import carb
from typing import List, Optional, Tuple, Union, Callable
import omni.usd as ou
from omni.particle.system.core.scripts.core import Core as OmniCore
from omni.particle.system.core.scripts.utils import Utils as OmniUtils
from pxr import Sdf, UsdShade, PhysxSchema, Usd, UsdGeom, UsdPhysics
import omni
import omni.graph.core as ogc
from omnigibson.lazy_omni import _get_all_evaluators
from omni.kit.primitive.mesh.command import CreateMeshPrimWithDefaultXformCommand as CMPWDXC
import omni.timeline
from omnigibson.lazy_omni import get_prim_at_path
import numpy as np
import torch
import warp as wp
import math
from omni.isaac.core.articulations import ArticulationView as _ArticulationView
from omni.isaac.core.prims import RigidPrimView as _RigidPrimView

DEG2RAD = math.pi / 180.0


class CreateMeshPrimWithDefaultXformCommand(CMPWDXC):
    def __init__(self, prim_type: str, **kwargs):
        """
        Creates primitive.

        Args:
            prim_type (str): It supports Plane/Sphere/Cone/Cylinder/Disk/Torus/Cube.

        kwargs:
            object_origin (Gf.Vec3f): Position of mesh center in stage units.

            u_patches (int): The number of patches to tessellate U direction.

            v_patches (int): The number of patches to tessellate V direction.

            w_patches (int): The number of patches to tessellate W direction.
                             It only works for Cone/Cylinder/Cube.

            half_scale (float): Half size of mesh in centimeters. Default is None, which means it's controlled by settings.

            u_verts_scale (int): Tessellation Level of U. It's a multiplier of u_patches.

            v_verts_scale (int): Tessellation Level of V. It's a multiplier of v_patches.

            w_verts_scale (int): Tessellation Level of W. It's a multiplier of w_patches.
                                 It only works for Cone/Cylinder/Cube.
                                 For Cone/Cylinder, it's to tessellate the caps.
                                 For Cube, it's to tessellate along z-axis.

            above_ground (bool): It will offset the center of mesh above the ground plane if it's True,
                False otherwise. It's False by default. This param only works when param object_origin is not given.
                Otherwise, it will be ignored.

            stage (Usd.Stage): If specified, stage to create prim on
        """

        self._prim_type = prim_type[0:1].upper() + prim_type[1:].lower()
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage = kwargs.get("stage", self._usd_context.get_stage())
        self._settings = carb.settings.get_settings()
        self._default_path = kwargs.get("prim_path", None)
        self._select_new_prim = kwargs.get("select_new_prim", True)
        self._prepend_default_prim = kwargs.get("prepend_default_prim", True)
        self._above_round = kwargs.get("above_ground", False)

        self._attributes = {**kwargs}
        # Supported mesh types should have an associated evaluator class
        self._evaluator_class = _get_all_evaluators()[prim_type]
        assert isinstance(self._evaluator_class, type)


class Utils2022(OmniUtils):
    """
    Subclass that overrides a specific function within Omni's Utils class to fix a bug
    """
    def create_material(self, name):
        # TODO: THIS IS THE ONLY LINE WE CHANGE! "/" SHOULD BE ""
        material_path = ""
        default_prim = self.stage.GetDefaultPrim()
        if default_prim:
            material_path = default_prim.GetPath().pathString

        if not self.stage.GetPrimAtPath(material_path + "/Looks"):
            self.stage.DefinePrim(material_path + "/Looks", "Scope")
        material_path += "/Looks/" + name
        material_path = ou.get_stage_next_free_path(
            self.stage, material_path, False
        )
        material = UsdShade.Material.Define(self.stage, material_path)

        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)

        # Update Neuraylib MDL search paths
        import omni.particle.system.core as core
        core.update_mdl_search_paths()

        shader.SetSourceAsset(name + ".mdl", "mdl")
        shader.SetSourceAssetSubIdentifier(name, "mdl")
        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader, "out")

        return [material_path]


class Utils2023(OmniUtils):
    def create_material(self, name):
        material_url = carb.settings.get_settings().get("/exts/omni.particle.system.core/material")

        # TODO: THIS IS THE ONLY LINE WE CHANGE! "/" SHOULD BE ""
        material_path = ""
        default_prim = self.stage.GetDefaultPrim()
        if default_prim:
            material_path = default_prim.GetPath().pathString

        if not self.stage.GetPrimAtPath(material_path + "/Looks"):
            self.stage.DefinePrim(material_path + "/Looks", "Scope")
        material_path += "/Looks/" + name
        material_path = ou.get_stage_next_free_path(
            self.stage, material_path, False
        )
        prim = self.stage.DefinePrim(material_path, "Material")
        if material_url:
            prim.GetReferences().AddReference(material_url)
        else:
            carb.log_error("Failed to find material URL in settings")

        return [material_path]


class Core(OmniCore):
    """
    Subclass that overrides a specific function within Omni's Core class to fix a bug
    """
    def __init__(self, popup_callback: Callable[[str], None], particle_system_name: str):
        self._popup_callback = popup_callback
        from omnigibson.utils.sim_utils import meets_minimum_isaac_version
        self.utils = Utils2023() if meets_minimum_isaac_version("2023.0.0") else Utils2022()
        self.context = ou.get_context()
        self.stage = self.context.get_stage()
        self.selection = self.context.get_selection()
        self.particle_system_name = particle_system_name
        self.sub_stage_update = self.context.get_stage_event_stream().create_subscription_to_pop(self.on_stage_update)
        self.on_stage_update()

    def get_compute_graph(self, selected_paths, create_new_graph=True, created_paths=None):
        """
        Returns the first ComputeGraph found in selected_paths.
        If no graph is found and create_new_graph is true, a new graph will be created and its
        path appended to created_paths (if provided).
        """
        graph = None
        graph_paths = [path for path in selected_paths
                       if self.stage.GetPrimAtPath(path).GetTypeName() in ["ComputeGraph", "OmniGraph"] ]

        if len(graph_paths) > 0:
            graph = ogc.get_graph_by_path(graph_paths[0])
            if len(graph_paths) > 1:
                carb.log_warn(f"Multiple ComputeGraph prims selected. Only the first will be used: {graph.get_path_to_graph()}")
        elif create_new_graph:
            # If no graph was found in the selected prims, we'll make a new graph.
            # TODO: THIS IS THE ONLY LINE THAT WE CHANGE! ONCE FIXED, REMOVE THIS
            graph_path = Sdf.Path(f"/OmniGraph/{self.particle_system_name}").MakeAbsolutePath(Sdf.Path.absoluteRootPath)
            graph_path = ou.get_stage_next_free_path(self.stage, graph_path, True)

            # prim = self.stage.GetDefaultPrim()
            # path = str(prim.GetPath()) if prim else ""
            self.stage.DefinePrim("/OmniGraph", "Scope")

            container_graphs = ogc.get_global_container_graphs()
            # FIXME: container_graphs[0] should be the simulation orchestration graph, but this may change in the future.
            container_graph = container_graphs[0]
            result, wrapper_node = ogc.cmds.CreateGraphAsNode(
                graph=container_graph,
                node_name=Sdf.Path(graph_path).name,
                graph_path=graph_path,
                evaluator_name="push",
                is_global_graph=True,
                backed_by_usd=True,
                fc_backing_type=ogc.GraphBackingType.GRAPH_BACKING_TYPE_FLATCACHE_SHARED,
                pipeline_stage=ogc.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
            )
            graph = wrapper_node.get_wrapped_graph()

            if created_paths is not None:
                created_paths.append(graph.get_path_to_graph())

            carb.log_info(f"No ComputeGraph selected. A new graph has been created at {graph.get_path_to_graph()}")

        return graph


class ArticulationView(_ArticulationView):
    """ArticulationView with some additional functionality implemented."""

    def set_joint_limits(
        self,
        values: Union[np.ndarray, torch.Tensor],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Sets joint limits for articulation joints in the view.

        Args:
            values (Union[np.ndarray, torch.Tensor, wp.array]): joint limits for articulations in the view. shape (M, K, 2).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            new_values = self._physics_view.get_dof_limits()
            values = self._backend_utils.move_data(values, device="cpu")
            new_values = self._backend_utils.assign(
                values,
                new_values,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_limits(new_values, indices)
        else:
            indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(indices, self.count, self._device)
            )
            dof_types = self._backend_utils.to_list(self.get_dof_types())
            joint_indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            )
            values = self._backend_utils.to_list(values)
            articulation_read_idx = 0
            for i in indices:
                dof_read_idx = 0
                for dof_index in joint_indices:
                    dof_val = values[articulation_read_idx][dof_read_idx]
                    if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation:
                        dof_val /= DEG2RAD
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    prim.GetAttribute("physics:lowerLimit").Set(dof_val[0])
                    prim.GetAttribute("physics:upperLimit").Set(dof_val[1])
                    dof_read_idx += 1
                articulation_read_idx += 1
        return
    
    def get_joint_limits(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.array]:
        """Gets joint limits for articulation in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (Optional[bool]): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: joint limits for articulations in the view. shape (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            values = self._backend_utils.move_data(self._physics_view.get_dof_limits(), self._device)
            if clone:
                values = self._backend_utils.clone_tensor(values, device=self._device)
            result = values[
                self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
            ]
            return result
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            dof_types = self._backend_utils.to_list(self.get_dof_types())
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            values = np.zeros(shape=(indices.shape[0], joint_indices.shape[0], 2), dtype="float32")
            articulation_write_idx = 0
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    values[articulation_write_idx][dof_write_idx][0] = prim.GetAttribute("physics:lowerLimit").Get()
                    values[articulation_write_idx][dof_write_idx][1] = prim.GetAttribute("physics:upperLimit").Get()
                    if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation:
                        values[articulation_write_idx][dof_write_idx] = values[articulation_write_idx][dof_write_idx] * DEG2RAD
                    dof_write_idx += 1
                articulation_write_idx += 1
            values = self._backend_utils.convert(values, dtype="float32", device=self._device, indexed=True)
            return values
        
    def set_max_velocities(
        self,
        values: Union[np.ndarray, torch.Tensor, wp.array],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Sets maximum velocities for articulation in the view.

        Args:
            values (Union[np.ndarray, torch.Tensor, wp.array]): maximum velocities for articulations in the view. shape (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            new_values = self._physics_view.get_dof_max_velocities()
            new_values = self._backend_utils.assign(
                self._backend_utils.move_data(values, device="cpu"),
                new_values,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_max_velocities(new_values, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            articulation_read_idx = 0
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            values = self._backend_utils.to_list(values)
            for i in indices:
                dof_read_idx = 0
                for dof_index in joint_indices:
                    prim = PhysxSchema.PhysxJointAPI(get_prim_at_path(self._dof_paths[i][dof_index]))
                    if not prim.GetMaxJointVelocityAttr():
                        prim.CreateMaxJointVelocityAttr().Set(values[articulation_read_idx][dof_read_idx])
                    else:
                        prim.GetMaxJointVelocityAttr().Set(values[articulation_read_idx][dof_read_idx])
                    dof_read_idx += 1
                articulation_read_idx += 1
        return

    def get_max_velocities(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        clone: bool = True,
    ) -> Union[np.ndarray, torch.Tensor, wp.indexedarray]:
        """Gets maximum velocities for articulation in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indicies to specify which joints
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (Optional[bool]): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor, wp.indexedarray]: maximum velocities for articulations in the view. shape (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, "cpu")
            max_velocities = self._physics_view.get_dof_max_velocities()
            if clone:
                max_velocities = self._backend_utils.clone_tensor(max_velocities, device="cpu")
            result = self._backend_utils.move_data(
                max_velocities[
                    self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices
                ],
                device=self._device,
            )
            return result
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            max_velocities = np.zeros(shape=(indices.shape[0], joint_indices.shape[0]), dtype="float32")
            indices = self._backend_utils.to_list(indices)
            joint_indices = self._backend_utils.to_list(joint_indices)
            articulation_write_idx = 0
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    prim = PhysxSchema.PhysxJointAPI(get_prim_at_path(self._dof_paths[i][dof_index]))
                    max_velocities[articulation_write_idx][dof_write_idx] = prim.GetMaxJointVelocityAttr().Get()
                    dof_write_idx += 1
                articulation_write_idx += 1
            max_velocities = self._backend_utils.convert(max_velocities, dtype="float32", device=self._device, indexed=True)
            return max_velocities
        

class RigidPrimView(_RigidPrimView):
    def enable_gravities(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> None:
        """Enable gravity on rigid bodies (enabled by default).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
            data = self._physics_view.get_disable_gravities().reshape(self._count)
            data = self._backend_utils.assign(
                self._backend_utils.create_tensor_from_list([False] * len(indices), dtype="uint8"), data, indices
            )
            self._physics_view.set_disable_gravities(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._physx_rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
                    self._physx_rigid_body_apis[i] = rigid_api
                self._physx_rigid_body_apis[i].GetDisableGravityAttr().Set(False)

    def disable_gravities(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> None:
        """Disable gravity on rigid bodies (enabled by default).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        indices = self._backend_utils.resolve_indices(indices, self.count, "cpu")
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            data = self._physics_view.get_disable_gravities().reshape(self._count)
            data = self._backend_utils.assign(
                self._backend_utils.create_tensor_from_list([True] * len(indices), dtype="uint8"), data, indices
            )
            self._physics_view.set_disable_gravities(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            indices = self._backend_utils.to_list(indices)
            for i in indices:
                if self._physx_rigid_body_apis[i] is None:
                    if self._prims[i].HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
                    self._physx_rigid_body_apis[i] = rigid_api
                self._physx_rigid_body_apis[i].GetDisableGravityAttr().Set(True)
            return
