import itertools

import torch as th
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import sample_cloth_on_rigid

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Percentage of xy-plane of the object's base aligned bbox that needs to covered by the cloth
m.OVERLAP_AREA_PERCENTAGE = 0.5

# z-offset for sampling
m.SAMPLING_Z_OFFSET = 0.01


class Overlaid(KinematicsMixin, RelativeObjectState, BooleanStateMixin):

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Touching)
        return deps

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("Overlaid does not support set_value(False)")
        state = og.sim.dump_state(serialized=False)

        if sample_cloth_on_rigid(self.obj, other, randomize_xy=False) and self.get_value(other):
            return True
        else:
            og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        """
        Check whether the (cloth) object is overlaid on the other (rigid) object.
        First, the cloth object needs to be touching the rigid object.
        Then, the convex hull of the particles of the cloth object needs to cover a decent percentage of the
        base aligned bounding box of the other rigid object.
        """
        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Overlaid state requires obj1 is cloth and obj2 is rigid.")

        if not self.obj.states[Touching].get_value(other):
            return False

        # Compute the convex hull of the particles of the cloth object.
        points = self.obj.root_link.keypoint_particle_positions[:, :2]
        cloth_hull = ConvexHull(points)

        # Compute the base aligned bounding box of the rigid object.
        bbox_center, bbox_orn, bbox_extent, _ = other.get_base_aligned_bbox(xy_aligned=True)
        vertices_local = th.tensor(list(itertools.product((1, -1), repeat=3))) * (bbox_extent / 2)
        vertices = T.transform_points(vertices_local, T.pose2mat((bbox_center, bbox_orn)))
        rigid_hull = ConvexHull(vertices[:, :2])

        # The goal is to find the intersection of the convex hull and the bounding box.
        # We can do so with HalfspaceIntersection, which takes as input a list of equations that define the half spaces,
        # and an interior point. We assume the center of the bounding box is an interior point.
        interior_pt = th.mean(vertices, dim=0)[:2]
        half_spaces = th.vstack((th.tensor(cloth_hull.equations), th.tensor(rigid_hull.equations)))
        try:
            half_space_intersection = HalfspaceIntersection(half_spaces, interior_pt)
        except QhullError:
            # The bbox center of the rigid body does not lie in the intersection, return False.
            return False

        # Compute the ratio between the intersection area and the bounding box area in the x-y plane.
        # When input points are 2-dimensional, this is the area of the convex hull.
        # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        intersection_area = ConvexHull(half_space_intersection.intersections).volume
        rigid_xy_area = bbox_extent[0] * bbox_extent[1]

        return (intersection_area / rigid_xy_area) > m.OVERLAP_AREA_PERCENTAGE
