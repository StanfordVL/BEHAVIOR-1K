import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanState, AbsoluteObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEBUG_VISUALIZATION = False

# Criterion #1: the threshold on the area reduction ratio of the convex hull of the projection on the XY plane
m.AREA_REDUCTION_THRESHOLD = 0.75

# Criterion #2: the threshold on the reduction of the diagonal of the projection's convex hull
m.DIAGONAL_REDUCTION_THRESHOLD = 0.75

# Criterion #3: the percentage of face normals that need to be close to the z-axis.
m.NORMAL_Z_PERCENTAGE = 0.5
m.NORMAL_Z_ANGLE_DIFF = np.deg2rad(30.0)


class Folded(AbsoluteObjectState, BooleanState):

    def calculate_projection_area_and_diagonal(self, dims):
        """
        Calculate the projection area and the diagonal length when projecting to the plane defined by the input dims
        E.g. if dims is [0, 1], the points will be projected onto the x-y plane.

        Args:
            dims (2-array): Global axes to project area onto. Options are {0, 1, 2}.
                E.g. if dims is [0, 1], project onto the x-y plane.

        Returns:
            area (float): 
        """
        cloth = self.obj.links["base_link"]
        points = cloth.particle_positions[:, dims]

        hull = ConvexHull(points)

        diagonal = distance_matrix(points[hull.vertices], points[hull.vertices]).max()

        if m.DEBUG_VISUALIZATION:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.set_aspect('equal')

            plt.plot(points[:, dims[0]], points[:, dims[1]], 'o')
            for simplex in hull.simplices:
                plt.plot(points[simplex, dims[0]], points[simplex, dims[1]], 'k-')
            plt.plot(points[hull.vertices, dims[0]], points[hull.vertices, dims[1]], 'r--', lw=2)
            plt.plot(points[hull.vertices[0], dims[0]], points[hull.vertices[0], dims[1]], 'ro')
            plt.show()

        return hull.area, diagonal

    def calculate_projection_area_and_diagonal_unfolded(self):
        """
        Calculate the maximum projection area and the diagonal length along different axes in the unfolded state.
        Should be called in the initialize function. Assume the object's default pose is unfolded.
        """
        # use the largest projection area as the unfolded area
        area_unfolded = 0.0
        diagonal_unfolded = 0.0
        dims_list = [[0, 1], [0, 2], [1, 2]]  # x-y plane, x-z plane, y-z plane

        for dims in dims_list:
            area, diagonal = self.calculate_projection_area_and_diagonal(dims=dims)
            if area > area_unfolded:
                area_unfolded = area
                diagonal_unfolded = diagonal

        return area_unfolded, diagonal_unfolded

    def check_projection_area_and_diagonal(self):
        """
        Check whether the current projection area and diagonal length satisfy the thresholds
        """
        area, diagonal = self.calculate_projection_area_and_diagonal([0, 1])

        # Check area reduction ratio
        flag_area_reduction = (area / self.area_unfolded) < m.AREA_REDUCTION_THRESHOLD

        # Check the diagonal reduction ratio
        flag_diagonal_reduction = (diagonal / self.diagonal_unfolded) < m.DIAGONAL_REDUCTION_THRESHOLD

        return flag_area_reduction, flag_diagonal_reduction

    def check_smoothness(self):
        """
        Check the smoothness of the cloth; the face normals of the cloth need to be close to the z-axis.
        """
        cloth = self.obj.links["base_link"]
        face_vertex_counts = np.array(cloth.get_attribute("faceVertexCounts"))
        assert (face_vertex_counts == 3).all(), "cloth prim is expected to only contain triangle faces"
        face_vertex_indices = np.array(cloth.get_attribute("faceVertexIndices"))
        points = cloth.particle_positions[face_vertex_indices]
        # Shape [F, 3, 3] where F is the number of faces
        points = points.reshape((face_vertex_indices.shape[0] // 3, 3, 3))

        # Shape [F, 3]
        v1 = points[:, 2, :] - points[:, 0, :]
        v2 = points[:, 1, :] - points[:, 0, :]
        normals = np.cross(v1, v2)
        normals_norm = np.linalg.norm(normals, axis=1)

        valid_normals = normals[normals_norm.nonzero()] / np.expand_dims(normals_norm[normals_norm.nonzero()], axis=1)
        assert valid_normals.shape[0] > 0

        # projection onto the z-axis
        proj = np.abs(np.dot(valid_normals, np.array([0.0, 0.0, 1.0])))
        percentage = np.mean(proj > np.cos(m.NORMAL_Z_ANGLE_DIFF))
        return percentage > m.NORMAL_Z_PERCENTAGE

    def _initialize(self):
        self.area_unfolded, self.diagonal_unfolded = self.calculate_projection_area_and_diagonal_unfolded()

    def _get_value(self):
        # Check the smoothness of the cloth
        flag_smoothness = self.check_smoothness()

        # Early stopping
        if not flag_smoothness:
            return False

        # Calculate the area and the diagonal of the current state
        flag_area_reduction, flag_diagonal_reduction = self.check_projection_area_and_diagonal()

        return flag_diagonal_reduction and flag_smoothness

    def _set_value(self, new_value):
        """
        Set the folded state. Currently, it's not supported yet.
        """
        raise NotImplementedError("_set_value of the Folded state has not been implemented")

    # We don't need to dump / load anything since the cloth objects should handle it themselves
