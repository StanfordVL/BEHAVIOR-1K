import contextlib


class UsdStageServicesMixin:
    """Pure-USD RenderBackend stage services; must precede RenderBackend in the bases."""

    def create_default_pbr_material(self, scope_path, material_path, target_prim_path):
        pass

    def get_stage(self):
        return self.sim.stage

    def copy_prim(self, source_prim_path, dest_prim_path):
        from omnigibson.utils.usd_utils import copy_prim_tree_to_path

        copy_prim_tree_to_path(source_prim_path, dest_prim_path)

    def copy_mesh_prim(self, source_prim_path, dest_prim_path):
        from omnigibson.utils.usd_utils import copy_mesh_prim_to_path

        copy_mesh_prim_to_path(source_prim_path, dest_prim_path)

    def deactivate_prim(self, prim):
        prim.SetActive(False)

    def delete_prim(self, prim_path, destructive=True):
        if destructive:
            self.sim.stage.RemovePrim(prim_path)
        else:
            self.sim.stage.GetPrimAtPath(prim_path).SetActive(False)

    def is_prim_ancestral(self, prim):
        prim_stack = prim.GetPrimStack()
        prim_stage = prim.GetStage()
        own_layers = {prim_stage.GetRootLayer(), prim_stage.GetSessionLayer()}
        return bool(prim_stack) and prim_stack[0].layer not in own_layers

    def add_reference_to_stage(self, asset_path, prim_path):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = self.sim.stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(asset_path)
        return prim

    def create_primitive_mesh(self, primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
        from omnigibson.utils.usd_utils import _create_mesh_prim_standalone

        _create_mesh_prim_standalone(primitive_type, prim_path, u_patches=u_patches, v_patches=v_patches, stage=stage)

    def compute_world_aabb(self, prim_path):
        from omnigibson.utils.usd_utils import compute_path_world_aabb

        return compute_path_world_aabb(prim_path)

    def recompute_extents(self, prim):
        pass

    def add_semantic_labels(self, prim, label, instance_name="class"):
        pass

    def suppress_log(self, channels=None):
        return contextlib.nullcontext()
