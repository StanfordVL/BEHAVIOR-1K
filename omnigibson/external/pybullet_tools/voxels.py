"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
import os

import numpy as np
import time
from itertools import product

from .utils import unit_pose, safe_zip, multiply, Pose, AABB, create_box, set_pose, get_all_links, LockRenderer, \
    get_aabb, pairwise_link_collision, remove_body, draw_aabb, get_box_geometry, create_shape, create_body, STATIC_MASS, \
    unit_quat, unit_point, CLIENT, create_shape_array, set_color, get_point, clip, load_model, TEMP_DIR, NULL_ID, elapsed_time

MAX_TEXTURE_WIDTH = 418 # max square dimension
MAX_PIXEL_VALUE = 255
MAX_LINKS = 125  # Max links seems to be 126


class VoxelGrid(object):
    # https://github.mit.edu/caelan/ROS/blob/master/sparse_voxel_grid.py
    # https://github.mit.edu/caelan/ROS/blob/master/base_navigation.py
    # https://github.mit.edu/caelan/ROS/blob/master/utils.py
    # https://github.mit.edu/caelan/ROS/blob/master/voxel_detection.py
    # TODO: can always display the grid in RVIZ after filtering
    # TODO: compute the maximum sized cuboid (rectangle) in a grid (matrix)

    def __init__(self, resolutions, color=(1, 0, 0, 0.5)):
    #def __init__(self, sizes, centers, pose=unit_pose()):
        #assert len(sizes) == len(centers)
        self.resolutions = resolutions
        self.occupied = set()
        self.world_from_grid = unit_pose() # TODO: support for real
        self.color = color
        #self.bodies = None
        # TODO: store voxels more intelligently spatially
    def __len__(self):
        return len(self.occupied)

    def voxel_from_point(self, point):
        return tuple(np.floor(np.divide(point, self.resolutions)).astype(int))
    def voxels_from_aabb(self, aabb):
        lower_voxel, upper_voxel = map(self.voxel_from_point, aabb)
        return map(tuple, product(*[range(l, u + 1) for l, u in safe_zip(lower_voxel, upper_voxel)]))

    def lower_from_voxel(self, voxel):
        return np.multiply(voxel, self.resolutions)
    def center_from_voxel(self, voxel):
        return self.lower_from_voxel(voxel) + self.resolutions/2.
    def upper_from_voxel(self, voxel):
        return self.lower_from_voxel(voxel) + self.resolutions
    def pose_from_voxel(self, voxel):
        return multiply(self.world_from_grid, Pose(self.center_from_voxel(voxel)))
    def aabb_from_voxel(self, voxel):
        return AABB(self.lower_from_voxel(voxel), self.upper_from_voxel(voxel))

    def is_occupied(self, voxel):
        return voxel in self.occupied
    def set_occupied(self, voxel):
        if self.is_occupied(voxel):
            return False
        self.occupied.add(voxel)
        return True
    def set_free(self, voxel):
        if not self.is_occupied(voxel):
            return False
        self.occupied.remove(voxel)
        return True

    def get_neighbors(self, index):
        for i in range(len(index)):
            direction = np.zeros(len(index), dtype=int)
            for n in (-1, +1):
                direction[i] = n
                yield tuple(np.array(index) + direction)

    def get_clusters(self, voxels=None):
        if voxels is None:
            voxels = self.occupied

        clusters = []
        assigned = set()
        def dfs(current):
            if (current in assigned) or (not self.is_occupied(current)):
                return []
            cluster = [current]
            assigned.add(current)
            for neighbor in self.get_neighbors(current):
                cluster.extend(dfs(neighbor))
            return cluster

        for voxel in voxels:
            cluster = dfs(voxel)
            if cluster:
                clusters.append(cluster)
        return clusters

    # TODO: implicitly check collisions
    def create_box(self):
        color = (0, 0, 0, 0)
        #color = None
        box = create_box(*self.resolutions, color=color)
        #set_color(box, color=color)
        set_pose(box, self.world_from_grid)
        return box
    def get_affected(self, bodies, occupied):
        assert self.world_from_grid == unit_pose()
        check_voxels = {}
        for body in bodies:
            for link in get_all_links(body):
                aabb = get_aabb(body, link) # TODO: pad using threshold
                for voxel in self.voxels_from_aabb(aabb):
                    if self.is_occupied(voxel) == occupied:
                        check_voxels.setdefault(voxel, []).append((body, link))
        return check_voxels
    def check_collision(self, box, voxel, pairs, threshold=0.):
        box_pairs = [(box, link) for link in get_all_links(box)]
        set_pose(box, self.pose_from_voxel(voxel))
        return any(pairwise_link_collision(body1, link1, body2, link2, max_distance=threshold)
                   for (body1, link1), (body2, link2) in product(pairs, box_pairs))

    def add_point(self, point):
        self.set_occupied(self.voxel_from_point(point))
    def add_aabb(self, aabb):
        for voxel in self.voxels_from_aabb(aabb):
            self.set_occupied(voxel)
    def add_bodies(self, bodies, threshold=0.):
        # Otherwise, need to transform bodies
        check_voxels = self.get_affected(bodies, occupied=False)
        box = self.create_box()
        for voxel, pairs in check_voxels.items(): # pairs typically only has one element
            if self.check_collision(box, voxel, pairs, threshold=threshold):
                self.set_occupied(voxel)
        remove_body(box)
    def remove_bodies(self, bodies, threshold=1e-2):
        # TODO: could also just iterate over the voxels directly
        check_voxels = self.get_affected(bodies, occupied=True)
        box = self.create_box()
        for voxel, pairs in check_voxels.items():
            if self.check_collision(box, voxel, pairs, threshold=threshold):
                self.set_free(voxel)
        remove_body(box)

    def draw_voxel_bodies(self):
        # TODO: transform into the world frame
        with LockRenderer():
            handles = []
            for voxel in sorted(self.occupied):
                handles.extend(draw_aabb(self.aabb_from_voxel(voxel), color=self.color[:3]))
            return handles
    def create_voxel_bodies1(self):
        start_time = time.time()
        geometry = get_box_geometry(*self.resolutions)
        collision_id, visual_id = create_shape(geometry, color=self.color)
        bodies = []
        for voxel in sorted(self.occupied):
            body = create_body(collision_id, visual_id)
            #scale = self.resolutions[0]
            #body = load_model('models/voxel.urdf', fixed_base=True, scale=scale)
            set_pose(body, self.pose_from_voxel(voxel))
            bodies.append(body) # 0.0462474774444 / voxel
        print(elapsed_time(start_time))
        return bodies
    def create_voxel_bodies2(self):
        geometry = get_box_geometry(*self.resolutions)
        collision_id, visual_id = create_shape(geometry, color=self.color)
        ordered_voxels = sorted(self.occupied)
        bodies = []
        for start in range(0, len(ordered_voxels), MAX_LINKS):
            voxels = ordered_voxels[start:start + MAX_LINKS]
            body = p.createMultiBody(#baseMass=STATIC_MASS,
                                     #baseCollisionShapeIndex=-1,
                                     #baseVisualShapeIndex=-1,
                                     #basePosition=unit_point(),
                                     #baseOrientation=unit_quat(),
                                     #baseInertialFramePosition=unit_point(),
                                     #baseInertialFrameOrientation=unit_quat(),
                                      linkMasses=len(voxels)*[STATIC_MASS],
                                      linkCollisionShapeIndices=len(voxels)*[collision_id],
                                      linkVisualShapeIndices=len(voxels)*[visual_id],
                                      linkPositions=list(map(self.center_from_voxel, voxels)),
                                      linkOrientations=len(voxels)*[unit_quat()],
                                      linkInertialFramePositions=len(voxels)*[unit_point()],
                                      linkInertialFrameOrientations=len(voxels)*[unit_quat()],
                                      linkParentIndices=len(voxels)*[0],
                                      linkJointTypes=len(voxels)*[p.JOINT_FIXED],
                                      linkJointAxis=len(voxels)*[unit_point()],
                                      physicsClientId=CLIENT)
            set_pose(body, self.world_from_grid)
            bodies.append(body) # 0.0163199263677 / voxel
        return bodies
    def create_voxel_bodies3(self):
        ordered_voxels = sorted(self.occupied)
        geoms = [get_box_geometry(*self.resolutions) for _ in ordered_voxels]
        poses = list(map(self.pose_from_voxel, ordered_voxels))
        #colors = [list(self.color) for _ in self.voxels] # TODO: colors don't work
        colors = None
        collision_id, visual_id = create_shape_array(geoms, poses, colors)
        body = create_body(collision_id, visual_id) # Max seems to be 16
        #dump_body(body)
        set_color(body, self.color)
        return [body]
    def create_voxel_bodies(self):
        with LockRenderer():
            return self.create_voxel_bodies1()
            #return self.create_voxel_bodies2()
            #return self.create_voxel_bodies3()

    def project2d(self):
        # TODO: combine adjacent voxels into larger lines
        # TODO: greedy algorithm that combines lines/boxes
        tallest_voxel = {}
        for i, j, k in self.occupied:
            tallest_voxel[i, j] = max(k, tallest_voxel.get((i, j), k))
        return {(i, j, k) for (i, j), k in tallest_voxel.items()}
    def create_height_map(self, plane, plane_size, width=MAX_TEXTURE_WIDTH, height=MAX_TEXTURE_WIDTH):
        min_z, max_z = 0., 2.
        plane_extent = plane_size*np.array([1, 1, 0])
        plane_lower = get_point(plane) - plane_extent/2.
        #plane_aabb = (plane_lower, plane_lower + plane_extent)
        #plane_aabb = get_aabb(plane) # TODO: bounding box is effectively empty
        #plane_lower, plane_upper = plane_aabb
        #plane_extent = (plane_upper - plane_lower)
        image_size = np.array([width, height])
        # TODO: fix width/height order
        pixel_from_point = lambda point: np.floor(
            image_size * (point - plane_lower)[:2] / plane_extent[:2]).astype(int)

        # TODO: last row/col doesn't seem to be filled
        height_map = np.zeros(image_size)
        for voxel in self.project2d():
            voxel_aabb = self.aabb_from_voxel(voxel)
            #if not aabb_contains_aabb(aabb2d_from_aabb(voxel_aabb), aabb2d_from_aabb(plane_aabb)):
            #    continue
            (x1, y1), (x2, y2) = map(pixel_from_point, voxel_aabb)
            if (x1 < 0) or (width <= x2) or (y1 < 0) or (height <= y2):
                continue
            scaled_z = (clip(voxel_aabb[1][2], min_z, max_z) - min_z) / max_z
            for c in range(x1, x2+1):
                for y in range(y1, y2+1):
                    r = height - y - 1 # TODO: can also just set in bulk if using height_map
                    height_map[r, c] = max(height_map[r, c], scaled_z)
        return height_map


def create_textured_square(size, color=None,
                           width=MAX_TEXTURE_WIDTH, height=MAX_TEXTURE_WIDTH):
    body = load_model('models/square.urdf', scale=size)
    if color is not None:
        set_color(body, color)
    path = os.path.join(TEMP_DIR, 'texture.png')
    image = MAX_PIXEL_VALUE*np.ones((width, height, 3), dtype=np.uint8)
    import scipy.misc
    scipy.misc.imsave(path, image)
    texture = p.loadTexture(path)
    p.changeVisualShape(body, NULL_ID, textureUniqueId=texture, physicsClientId=CLIENT)
    return body, texture


def set_texture(texture, image):
    # Alias/WaveFront Material (.mtl) File Format
    # https://people.cs.clemson.edu/~dhouse/courses/405/docs/brief-mtl-file-format.html
    #print(get_visual_data(body))
    width, height, channels = image.shape
    pixels = image.flatten().tolist()
    assert len(pixels) <= 524288
    # b3Printf: uploadBulletFileToSharedMemory 747003 exceeds max size 524288
    p.changeTexture(texture, pixels, width, height, physicsClientId=CLIENT)
    # TODO: it's important that width and height are the same as the original


def rgb_interpolate(grey_image, min_color, max_color):
    width, height = grey_image.shape
    channels = 3
    rgb_image = np.zeros((width, height, channels), dtype=np.uint8)
    for k in range(channels):
        rgb_image[..., k] = MAX_PIXEL_VALUE*(min_color[k]*(1-grey_image) + max_color[k]*grey_image)
    return rgb_image
