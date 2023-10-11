"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""
import carb
from typing import Callable
import omni.usd as ou
from omni.particle.system.core.scripts.core import Core as OmniCore
from omni.particle.system.core.scripts.utils import Utils as OmniUtils
from pxr import Sdf, UsdShade
import omni
import omni.graph.core as ogc
from omni.kit.primitive.mesh.command import _get_all_evaluators
from omni.kit.primitive.mesh.command import CreateMeshPrimWithDefaultXformCommand as CMPWDXC


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


class Core(OmniCore):
    """
    Subclass that overrides a specific function within Omni's Core class to fix a bug
    """
    def __init__(self, popup_callback: Callable[[str], None], particle_system_name: str):
        super().__init__(popup_callback=popup_callback)
        self.particle_system_name = particle_system_name

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

