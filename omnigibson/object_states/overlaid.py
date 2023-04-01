from omnigibson.object_states.kinematics import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanState, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.utils.constants import PrimType
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros

import omnigibson as og

from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.qhull import QhullError
import numpy as np
import trimesh
import itertools

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Percentage of xy-plane of the object's base aligned bbox that needs to covered by the cloth
m.OVERLAP_AREA_PERCENTAGE = 0.5

# z-offset for sampling
m.SAMPLING_Z_OFFSET = 0.01


class Overlaid(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + RelativeObjectState.get_dependencies() + [Touching]

    def _set_value(self, other, new_value):
        if not new_value:
            raise NotImplementedError("Overlaid does not support set_value(False)")

        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Overlaid state requires obj1 is cloth and obj2 is rigid.")

        state = og.sim.dump_state(serialized=False)

        # Get the top center of the object below
        aabb_low, aabb_hi = other.states[AABB].get_value()
        top_center = np.array([(aabb_low[0] + aabb_hi[0]) / 2.0, (aabb_low[1] + aabb_hi[1]) / 2.0, aabb_hi[2]])

        # Reset the cloth
        self.obj.root_link.reset()

        # Add the half-extent of the object on top in the z-axis with additional offset
        aabb_low, aabb_hi = self.obj.states[AABB].get_value()
        pos = top_center + np.array([0., 0., (aabb_hi - aabb_low)[2] / 2.0 + m.SAMPLING_Z_OFFSET])

        for _ in range(10):
            # Use a random orientation in the z-axis
            orn = T.euler2quat(np.array([0., 0., np.random.uniform(0, np.pi * 2)]))
            self.obj.set_position_orientation(pos, orn)
            self.obj.root_link.reset()
            self.obj.keep_still()
            # Let it fall for 0.2 second
            for _ in range(int(0.2 / og.sim.get_physics_dt())):
                og.sim.step_physics()
                if len(self.obj.states[ContactBodies].get_value()) > 0:
                    break
            self.obj.keep_still()

            if self.get_value(other) == new_value:
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        """
        Check whether the (cloth) object is overlaid on the other (rigid) object.
        First, the two objects need to be Touching.
        Then, the convex hull of the particles of the cloth object needs to cover a decent percentage of the
        base aligned bounding box of the other rigid object.
        """
        if not (self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
            raise ValueError("Overlaid state requires obj1 is cloth and obj2 is rigid.")

        # Make sure the two objects are touching.
        touching = self.obj.states[Touching].get_value(other)
        if not touching:
            return False

        # Compute the convex hull of the particles of the cloth object.
        points = self.obj.root_link.get_particle_positions(keypoints_only=True)[:, :2]
        cloth_hull = ConvexHull(points)

        # Compute the base aligned bounding box of the rigid object.
        bbox_center, bbox_orn, bbox_extent, _ = other.get_base_aligned_bbox(xy_aligned=True)
        vertices_local = np.array(list(itertools.product((1, -1), repeat=3))) * (bbox_extent / 2)
        vertices = trimesh.transformations.transform_points(vertices_local, T.pose2mat((bbox_center, bbox_orn)))
        rigid_hull = ConvexHull(vertices[:, :2])

        # The goal is to find the intersection of the convex hull and the bounding box.
        # We can do so with HalfspaceIntersection, which takes as input a list of equations that define the half spaces,
        # and an interior point. We assume the center of the bounding box is an interior point.
        interior_pt = vertices.mean(axis=0)[:2]
        half_spaces = np.vstack((cloth_hull.equations, rigid_hull.equations))
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
