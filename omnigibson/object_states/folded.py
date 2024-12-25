import math
from collections import namedtuple

import torch as th
from scipy.spatial import ConvexHull, QhullError, distance_matrix

from omnigibson.macros import create_module_macros
from omnigibson.object_states.cloth_mixin import ClothStateMixin
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Criterion #1: the threshold on the area ratio of the convex hull of the projection on the XY plane
m.FOLDED_AREA_THRESHOLD = 0.75
m.UNFOLDED_AREA_THRESHOLD = 0.9

# Criterion #2: the threshold on the diagonal ratio of the convex hull of the projection on the XY plane
m.FOLDED_DIAGONAL_THRESHOLD = 0.85
m.UNFOLDED_DIAGONAL_THRESHOLD = 0.9

# Criterion #3: the percentage of face normals that need to be close to the z-axis
m.NORMAL_Z_PERCENTAGE = 0.5

# Whether to visualize the convex hull of the projection on the XY plane
m.DEBUG_CLOTH_PROJ_VIS = False

# Angle threshold for checking smoothness of the cloth; surface normals need to be close enough to the z-axis
m.NORMAL_Z_ANGLE_DIFF = th.deg2rad(th.tensor([45.0])).item()

"""
FoldedLevelData contains the following fields:
    smoothness (float): percentage of surface normals that are sufficiently close to the z-axis
    area (float): the area of the convex hull of the projected points compared to the initial unfolded state
    diagonal (float): the diagonal of the convex hull of the projected points compared to the initial unfolded state
"""
FoldedLevelData = namedtuple("FoldedLevelData", ("smoothness", "area", "diagonal"))


class FoldedLevel(AbsoluteObjectState, ClothStateMixin):
    """
    State representing the object's folded level.
    Value is a FoldedLevelData object.
    """

    def _initialize(self):
        super()._initialize()
        # Assume the initial state is unfolded
        self.area_unfolded, self.diagonal_unfolded = self.calculate_projection_area_and_diagonal_maximum()

    def _get_value(self):
        smoothness = self.calculate_smoothness()
        area, diagonal = self.calculate_projection_area_and_diagonal([0, 1])
        return FoldedLevelData(smoothness, area / self.area_unfolded, diagonal / self.diagonal_unfolded)

    def calculate_smoothness(self):
        """
        Calculate the percantage of surface normals that are sufficiently close to the z-axis.
        """
        cloth = self.obj.root_link
        normals = cloth.compute_face_normals(face_ids=cloth.keyface_idx)

        # projection onto the z-axis
        proj = th.abs(normals @ th.tensor([0.0, 0.0, 1.0], dtype=th.float32))
        percentage = th.mean((proj > math.cos(m.NORMAL_Z_ANGLE_DIFF)).float()).item()
        return percentage

    def calculate_projection_area_and_diagonal_maximum(self):
        """
        Calculate the maximum projection area and the diagonal length along different axes

        Returns:
            area_max (float): area of the convex hull of the projected points
            diagonal_max (float): diagonal of the convex hull of the projected points
        """
        # use the largest projection area as the unfolded area
        area_max = 0.0
        diagonal_max = 0.0
        dims_list = [[0, 1], [0, 2], [1, 2]]  # x-y plane, x-z plane, y-z plane

        for dims in dims_list:
            area, diagonal = self.calculate_projection_area_and_diagonal(dims)
            if area > area_max:
                area_max = area
                diagonal_max = diagonal

        return area_max, diagonal_max

    def calculate_projection_area_and_diagonal(self, dims):
        """
        Calculate the projection area and the diagonal length when projecting to the plane defined by the input dims
        E.g. if dims is [0, 1], the points will be projected onto the x-y plane.

        Args:
            dims (2-array): Global axes to project area onto. Options are {0, 1, 2}.
                E.g. if dims is [0, 1], project onto the x-y plane.

        Returns:
            area (float): area of the convex hull of the projected points
            diagonal (float): diagonal of the convex hull of the projected points
        """
        cloth = self.obj.root_link
        points = cloth.keypoint_particle_positions[:, dims]
        try:
            hull = ConvexHull(points)

        # The points may be 2D-degenerate, so catch the error and return 0 if so
        except QhullError:
            # This is a degenerate hull, so return 0 area and diagonal
            return 0.0, 0.0

        # When input points are 2-dimensional, this is the area of the convex hull.
        # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        area = hull.volume
        diagonal = distance_matrix(points[hull.vertices], points[hull.vertices]).max()

        if m.DEBUG_CLOTH_PROJ_VIS:
            import matplotlib.pyplot as plt

            ax = plt.gca()
            ax.set_aspect("equal")

            plt.plot(points[:, dims[0]], points[:, dims[1]], "o")
            for simplex in hull.simplices:
                plt.plot(points[simplex, dims[0]], points[simplex, dims[1]], "k-")
            plt.plot(points[hull.vertices, dims[0]], points[hull.vertices, dims[1]], "r--", lw=2)
            plt.plot(points[hull.vertices[0], dims[0]], points[hull.vertices[0], dims[1]], "ro")
            plt.show()

        return area, diagonal


class Folded(AbsoluteObjectState, BooleanStateMixin, ClothStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(FoldedLevel)
        return deps

    def _get_value(self):
        # Check the smoothness of the cloth
        folded_level = self.obj.states[FoldedLevel].get_value()
        return (
            folded_level.smoothness >= m.NORMAL_Z_PERCENTAGE
            and folded_level.area < m.FOLDED_AREA_THRESHOLD
            and folded_level.diagonal < m.FOLDED_DIAGONAL_THRESHOLD
        )

    def _set_value(self, new_value):
        if not new_value:
            raise NotImplementedError("Folded does not support set_value(False)")

        # TODO (eric): add this support
        raise NotImplementedError("Folded does not support set_value(True)")

    # We don't need to dump / load anything since the cloth objects should handle it themselves


class Unfolded(AbsoluteObjectState, BooleanStateMixin, ClothStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(FoldedLevel)
        return deps

    def _get_value(self):
        # Check the smoothness of the cloth
        folded_level = self.obj.states[FoldedLevel].get_value()
        return (
            folded_level.smoothness >= m.NORMAL_Z_PERCENTAGE
            and folded_level.area >= m.UNFOLDED_AREA_THRESHOLD
            and folded_level.diagonal >= m.UNFOLDED_DIAGONAL_THRESHOLD
        )

    def _set_value(self, new_value):
        if not new_value:
            raise NotImplementedError("Unfolded does not support set_value(False)")

        self.obj.root_link.reset()

        return True

    # We don't need to dump / load anything since the cloth objects should handle it themselves
