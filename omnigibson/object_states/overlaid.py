from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.constants import PrimType
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros

from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.qhull import QhullError
import numpy as np
import trimesh
import itertools

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Percentage of xy-plane of the object's base aligned bbox that needs to covered by the cloth
m.OVERLAP_AREA_PERCENTAGE = 0.5

# Subsample cloth particle points to fit a convex hull for efficiency purpose
m.N_POINTS_CONVEX_HULL = 1000

class Overlaid(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + RelativeObjectState.get_dependencies() + [Touching]

    def _set_value(self, other, new_value, use_ray_casting_method=False):
        raise NotImplementedError("Overlaid state currently does not support setting.")

    def _get_value(self, other):
        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Overlaid state requires obj1 is cloth and obj2 is rigid.")

        touching = self.obj.states[Touching].get_value(other)
        if not touching:
            return False

        points = self.obj.root_link.particle_positions[:, :2]
        if points.shape[0] > m.N_POINTS_CONVEX_HULL:
            # If there are too many points, subsample m.N_POINTS_CONVEX_HULL deterministically for efficiency purpose
            np.random.seed(0)
            random_idx = np.random.randint(0, points.shape[0], m.N_POINTS_CONVEX_HULL)
            points = points[random_idx]

        cloth_hull = ConvexHull(points)

        bbox_center, bbox_orn, bbox_extent, _ = other.get_base_aligned_bbox(xy_aligned=True)
        vertices_local = np.array(list(itertools.product((1, -1), repeat=3))) * (bbox_extent / 2)
        vertices = trimesh.transformations.transform_points(vertices_local, T.pose2mat((bbox_center, bbox_orn)))
        rigid_hull = ConvexHull(vertices[:, :2])

        center_xy = vertices.mean(axis=0)[:2]
        half_spaces = np.vstack((cloth_hull.equations, rigid_hull.equations))
        try:
            half_space_intersection = HalfspaceIntersection(half_spaces, center_xy)
        except QhullError:
            # The bbox center of the rigid body does not lie in the intersection, return False
            return False

        # When input points are 2-dimensional, this is the area of the convex hull.
        # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        intersection_area = ConvexHull(half_space_intersection.intersections).volume
        rigid_xy_area = bbox_extent[0] * bbox_extent[1]

        return (intersection_area / rigid_xy_area) > m.OVERLAP_AREA_PERCENTAGE
