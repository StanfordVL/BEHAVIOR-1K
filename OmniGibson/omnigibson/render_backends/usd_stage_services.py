"""
Pure-pxr implementations of RenderBackend's stage / application services, for render backends that
run without a live Kit application (see ``RenderBackend``'s "Stage / application services" section).
Kit brings its own versions of these utilities (see ``KitRenderBackend``); without it, they are either
done directly on the USD stage or, where they only matter to a Kit renderer, skipped.
"""

import contextlib


class UsdStageServicesMixin:
    """
    Mixin implementing every RenderBackend stage service on plain USD. Must precede RenderBackend in
    the subclass's bases so these satisfy its abstract methods.
    """

    def create_default_pbr_material(self, scope_path, material_path, target_prim_path):
        # Purely visual, and the material/binding helpers go through omni.kit.commands -- nothing to do.
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
        # Ancestral iff the strongest spec defining this prim lives outside the stage's own root/session
        # layers, i.e. it was brought in through a reference arc.
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
        # Only keeps the (purely visual) authored extent attribute in sync for a renderer.
        pass

    def add_semantic_labels(self, prim, label, instance_name="class"):
        # No segmentation pipeline to label for.
        pass

    def suppress_log(self, channels=None):
        # No Kit logging system to suppress.
        return contextlib.nullcontext()
