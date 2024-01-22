"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""
import omnigibson.lazy as lazy
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import warp as wp
import math

DEG2RAD = math.pi / 180.0


class CreateMeshPrimWithDefaultXformCommand(lazy.omni.kit.primitive.mesh.command.CreateMeshPrimWithDefaultXformCommand):
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
        self._usd_context = lazy.omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage = kwargs.get("stage", self._usd_context.get_stage())
        self._settings = lazy.carb.settings.get_settings()
        self._default_path = kwargs.get("prim_path", None)
        self._select_new_prim = kwargs.get("select_new_prim", True)
        self._prepend_default_prim = kwargs.get("prepend_default_prim", True)
        self._above_round = kwargs.get("above_ground", False)

        self._attributes = {**kwargs}
        # Supported mesh types should have an associated evaluator class
        self._evaluator_class = lazy.omni.kit.primitive.mesh.command._get_all_evaluators()[prim_type]
        assert isinstance(self._evaluator_class, type)


class Utils2022(lazy.omni.particle.system.core.scripts.utils.Utils):
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
        material_path = lazy.omni.usd.get_stage_next_free_path(
            self.stage, material_path, False
        )
        material = lazy.pxr.UsdShade.Material.Define(self.stage, material_path)

        shader_path = material_path + "/Shader"
        shader = lazy.pxr.UsdShade.Shader.Define(self.stage, shader_path)

        # Update Neuraylib MDL search paths
        import lazy.omni.particle.system.core as core
        core.update_mdl_search_paths()

        shader.SetSourceAsset(name + ".mdl", "mdl")
        shader.SetSourceAssetSubIdentifier(name, "mdl")
        shader.GetImplementationSourceAttr().Set(lazy.pxr.UsdShade.Tokens.sourceAsset)
        shader.CreateOutput("out", lazy.pxr.Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader, "out")

        return [material_path]


class Utils2023(lazy.omni.particle.system.core.scripts.utils.Utils):
    def create_material(self, name):
        material_url = lazy.carb.settings.get_settings().get("/exts/omni.particle.system.core/material")

        # TODO: THIS IS THE ONLY LINE WE CHANGE! "/" SHOULD BE ""
        material_path = ""
        default_prim = self.stage.GetDefaultPrim()
        if default_prim:
            material_path = default_prim.GetPath().pathString

        if not self.stage.GetPrimAtPath(material_path + "/Looks"):
            self.stage.DefinePrim(material_path + "/Looks", "Scope")
        material_path += "/Looks/" + name
        material_path = lazy.omni.usd.get_stage_next_free_path(
            self.stage, material_path, False
        )
        prim = self.stage.DefinePrim(material_path, "Material")
        if material_url:
            prim.GetReferences().AddReference(material_url)
        else:
            lazy.carb.log_error("Failed to find material URL in settings")

        return [material_path]


class Core(lazy.omni.particle.system.core.scripts.core.Core):
    """
    Subclass that overrides a specific function within Omni's Core class to fix a bug
    """
    def __init__(self, popup_callback: Callable[[str], None], particle_system_name: str):
        self._popup_callback = popup_callback
        from omnigibson.utils.sim_utils import meets_minimum_isaac_version
        self.utils = Utils2023() if meets_minimum_isaac_version("2023.0.0") else Utils2022()
        self.context = lazy.omni.usd.get_context()
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
            graph = lazy.omni.graph.core.get_graph_by_path(graph_paths[0])
            if len(graph_paths) > 1:
                lazy.carb.log_warn(f"Multiple ComputeGraph prims selected. Only the first will be used: {graph.get_path_to_graph()}")
        elif create_new_graph:
            # If no graph was found in the selected prims, we'll make a new graph.
            # TODO: THIS IS THE ONLY LINE THAT WE CHANGE! ONCE FIXED, REMOVE THIS
            graph_path = lazy.pxr.Sdf.Path(f"/OmniGraph/{self.particle_system_name}").MakeAbsolutePath(lazy.pxr.Sdf.Path.absoluteRootPath)
            graph_path = lazy.omni.usd.get_stage_next_free_path(self.stage, graph_path, True)

            # prim = self.stage.GetDefaultPrim()
            # path = str(prim.GetPath()) if prim else ""
            self.stage.DefinePrim("/OmniGraph", "Scope")

            container_graphs = lazy.omni.graph.core.get_global_container_graphs()
            # FIXME: container_graphs[0] should be the simulation orchestration graph, but this may change in the future.
            container_graph = container_graphs[0]
            result, wrapper_node = lazy.omni.graph.core.cmds.CreateGraphAsNode(
                graph=container_graph,
                node_name=lazy.pxr.Sdf.Path(graph_path).name,
                graph_path=graph_path,
                evaluator_name="push",
                is_global_graph=True,
                backed_by_usd=True,
                fc_backing_type=lazy.omni.graph.core.GraphBackingType.GRAPH_BACKING_TYPE_FLATCACHE_SHARED,
                pipeline_stage=lazy.omni.graph.core.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
            )
            graph = wrapper_node.get_wrapped_graph()

            if created_paths is not None:
                created_paths.append(graph.get_path_to_graph())

            lazy.carb.log_info(f"No ComputeGraph selected. A new graph has been created at {graph.get_path_to_graph()}")

        return graph


class ArticulationView(lazy.omni.isaac.core.articulations.ArticulationView):
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
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    if dof_types[dof_index] == lazy.omni.physics.tensors.DofType.Rotation:
                        dof_val /= DEG2RAD
                    prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(self._dof_paths[i][dof_index])
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
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(self._dof_paths[i][dof_index])
                    values[articulation_write_idx][dof_write_idx][0] = prim.GetAttribute("physics:lowerLimit").Get()
                    values[articulation_write_idx][dof_write_idx][1] = prim.GetAttribute("physics:upperLimit").Get()
                    if dof_types[dof_index] == lazy.omni.physics.tensors.DofType.Rotation:
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
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    prim = lazy.pxr.PhysxSchema.PhysxJointAPI(lazy.omni.isaac.core.utils.prims.get_prim_at_path(self._dof_paths[i][dof_index]))
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
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    prim = lazy.pxr.PhysxSchema.PhysxJointAPI(lazy.omni.isaac.core.utils.prims.get_prim_at_path(self._dof_paths[i][dof_index]))
                    max_velocities[articulation_write_idx][dof_write_idx] = prim.GetMaxJointVelocityAttr().Get()
                    dof_write_idx += 1
                articulation_write_idx += 1
            max_velocities = self._backend_utils.convert(max_velocities, dtype="float32", device=self._device, indexed=True)
            return max_velocities
        
    def set_joint_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint positions of articulations in the view

        .. warning::

            This method will immediately set (teleport) the affected joints to the indicated value.
            Use the ``set_joint_position_targets`` or the ``apply_action`` methods to control the articulation joints.

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): joint positions of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set all the articulation joints.
            >>> # Since there are 5 envs, the joint positions are repeated 5 times
            >>> positions = np.tile(np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8, 0.04, 0.04]), (num_envs, 1))
            >>> prims.set_joint_positions(positions)
            >>>
            >>> # set only the fingers in closed position: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 0.0
            >>> # for the first, middle and last of the 5 envs
            >>> positions = np.tile(np.array([0.0, 0.0]), (3, 1))
            >>> prims.set_joint_positions(positions, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            new_dof_pos = self._physics_view.get_dof_positions()
            new_dof_pos = self._backend_utils.assign(
                self._backend_utils.move_data(positions, device=self._device),
                new_dof_pos,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_positions(new_dof_pos, indices)
            
            # THIS IS THE FIX: COMMENT OUT THE BELOW LINE AND SET TARGETS INSTEAD
            # self._physics_view.set_dof_position_targets(new_dof_pos, indices)
            self.set_joint_position_targets(positions, indices, joint_indices)
        else:
            lazy.carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_positions")

    def set_joint_velocities(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint velocities of articulations in the view

        .. warning::

            This method will immediately set the affected joints to the indicated value.
            Use the ``set_joint_velocity_targets`` or the ``apply_action`` methods to control the articulation joints.

        Args:
            velocities (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): joint velocities of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set the velocities for all the articulation joints to the indicated values.
            >>> # Since there are 5 envs, the joint velocities are repeated 5 times
            >>> velocities = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), (num_envs, 1))
            >>> prims.set_joint_velocities(velocities)
            >>>
            >>> # set the fingers velocities: panda_finger_joint1 (7) and panda_finger_joint2 (8) to -0.1
            >>> # for the first, middle and last of the 5 envs
            >>> velocities = np.tile(np.array([-0.1, -0.1]), (3, 1))
            >>> prims.set_joint_velocities(velocities, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            new_dof_vel = self._physics_view.get_dof_velocities()
            new_dof_vel = self._backend_utils.assign(
                self._backend_utils.move_data(velocities, device=self._device),
                new_dof_vel,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_velocities(new_dof_vel, indices)

            # THIS IS THE FIX: COMMENT OUT THE BELOW LINE AND SET TARGETS INSTEAD
            # self._physics_view.set_dof_velocity_targets(new_dof_vel, indices)
            self.set_joint_velocity_targets(velocities, indices, joint_indices)
        else:
            lazy.carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_velocities")
        return

    def set_joint_efforts(
        self,
        efforts: Optional[Union[np.ndarray, torch.Tensor, wp.array]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the joint efforts of articulations in the view

        .. note::

            This method can be used for effort control. For this purpose, there must be no joint drive
            or the stiffness and damping must be set to zero.

        Args:
            efforts (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): efforts of articulations in the view to be set to in the next frame.
                                                                    shape is (M, K).
            indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).

        .. hint::

            This method belongs to the methods used to set the articulation kinematic states:

            ``set_velocities`` (``set_linear_velocities``, ``set_angular_velocities``),
            ``set_joint_positions``, ``set_joint_velocities``, ``set_joint_efforts``

        Example:

        .. code-block:: python

            >>> # set the efforts for all the articulation joints to the indicated values.
            >>> # Since there are 5 envs, the joint efforts are repeated 5 times
            >>> efforts = np.tile(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]), (num_envs, 1))
            >>> prims.set_joint_efforts(efforts)
            >>>
            >>> # set the fingers efforts: panda_finger_joint1 (7) and panda_finger_joint2 (8) to 10
            >>> # for the first, middle and last of the 5 envs
            >>> efforts = np.tile(np.array([10, 10]), (3, 1))
            >>> prims.set_joint_efforts(efforts, indices=np.array([0, 2, 4]), joint_indices=np.array([7, 8]))
        """
        if not self._is_initialized:
            lazy.carb.log_warn("ArticulationView needs to be initialized.")
            return

        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)

            # THIS IS THE FIX: COMMENT OUT THE BELOW LINE AND USE ACTUATION FORCES INSTEAD
            # new_dof_efforts = self._backend_utils.create_zeros_tensor(
            #     shape=[self.count, self.num_dof], dtype="float32", device=self._device
            # )
            new_dof_efforts = self._physics_view.get_dof_actuation_forces()
            new_dof_efforts = self._backend_utils.assign(
                self._backend_utils.move_data(efforts, device=self._device),
                new_dof_efforts,
                [self._backend_utils.expand_dims(indices, 1) if self._backend != "warp" else indices, joint_indices],
            )
            self._physics_view.set_dof_actuation_forces(new_dof_efforts, indices)
        else:
            lazy.carb.log_warn("Physics Simulation View is not created yet in order to use set_joint_efforts")
        return

class RigidPrimView(lazy.omni.isaac.core.prims.RigidPrimView):
    def enable_gravities(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> None:
        """Enable gravity on rigid bodies (enabled by default).

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indicies to specify which prims
                                                                                 to manipulate. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
        """
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    if self._prims[i].HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
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
        if not lazy.omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
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
                    if self._prims[i].HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(self._prims[i])
                    else:
                        rigid_api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(self._prims[i])
                    self._physx_rigid_body_apis[i] = rigid_api
                self._physx_rigid_body_apis[i].GetDisableGravityAttr().Set(True)
            return
