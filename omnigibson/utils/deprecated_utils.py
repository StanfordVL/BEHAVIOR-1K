"""
A set of utility functions slated to be deprecated once Omniverse bugs are fixed
"""
import carb
from typing import Callable
import omni.usd as ou
from omni.particle.system.core.scripts.core import Core as OmniCore
from omni.particle.system.core.scripts.utils import Utils as OmniUtils
from pxr import Sdf, UsdShade
import omni.graph.core as ogc


class Utils(OmniUtils):
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


class Core(OmniCore):
    """
    Subclass that overrides a specific function within Omni's Core class to fix a bug
    """
    def __init__(self, popup_callback: Callable[[str], None], particle_system_name: str):
        self._popup_callback = popup_callback
        self.utils = Utils()    # TODO: THIS IS THE ONLY LINE THAT WE CHANGE! ONCE FIXED, REMOVE THIS
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
                       if self.stage.GetPrimAtPath(path).GetTypeName() == "ComputeGraph"]

        if len(graph_paths) > 0:
            graph = ogc.get_graph_by_path(graph_paths[0])
            if len(graph_paths) > 1:
                carb.log_warn(f"Multiple ComputeGraph prims selected. Only the first will be used: {graph.get_path_to_graph()}")
        elif create_new_graph:
            # If no graph was found in the selected prims, we'll make a new graph.
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

