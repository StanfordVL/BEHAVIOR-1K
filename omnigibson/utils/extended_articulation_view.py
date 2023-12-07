from typing import List, Optional, Tuple, Union
import carb
import omni.timeline
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
import numpy as np
import torch
import warp as wp

from omni.isaac.core.articulations import ArticulationView

class ExtendedArticulationView(ArticulationView):
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
            joint_indices = self._backend_utils.to_list(
                self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            )
            values = self._backend_utils.to_list(values)
            articulation_read_idx = 0
            for i in indices:
                dof_read_idx = 0
                for dof_index in joint_indices:
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    prim.GetAttribute("physics:lowerLimit").Set(values[articulation_read_idx][dof_read_idx][0])
                    prim.GetAttribute("physics:upperLimit").Set(values[articulation_read_idx][dof_read_idx][1])
                    dof_read_idx += 1
                articulation_read_idx += 1
        return