# TODO! Refactor
#
# #!/usr/bin/env python
#
# import argparse
# import os
# import random
#
# import cv2
# import numpy as np
# from PIL import Image
# from scipy.spatial import ConvexHull
# from tqdm import tqdm
#
# from igibson.scenes.interactive_traversable_scene import SCENE_SOURCE, InteractiveIndoorScene
# from igibson.simulator import Simulator
# from igibson.utils.asset_utils import get_3dfront_scene_path, get_cubicasa_scene_path, get_ig_scene_path
# from igibson.utils.map_utils import gen_trav_map
#
# """
# script to generate all traversability maps:
#
# for file in ../../igibson/ig_dataset/scenes/*
#   python generate_trav_map.py $(basename $file)
#
# """
#
#
# def get_xy_floors(vertices, faces, dist_threshold=-0.98):
#     z_faces = []
#     z = np.array([0, 0, 1])
#     faces_selected = []
#     for face in tqdm(faces):
#         normal = np.cross(vertices[face[2]] - vertices[face[1]], vertices[face[1]] - vertices[face[0]])
#         dist = np.dot(normal, z) / np.linalg.norm(normal)
#         if (dist_threshold is None) or ((dist_threshold is not None) and (dist < dist_threshold)):
#             z_faces.append(vertices[face[0]][2])
#             faces_selected.append(face)
#
#     return np.array(z_faces), vertices, faces_selected
#
#
# def gen_trav_map(
#     vertices,
#     faces,
#     output_folder,
#     add_clutter=False,
#     trav_map_filename_format="floor_trav_{}.png",
#     obstacle_map_filename_format="floor_{}.png",
# ):
#     """
#     Generate traversability maps.
#     """
#     floors = [0.0]
#
#     z_faces, vertices, faces_selected = get_xy_floors(vertices, faces)
#     z_faces_all, vertices_all, faces_selected_all = get_xy_floors(vertices, faces, dist_threshold=None)
#
#     xmin, ymin, _ = vertices.min(axis=0)
#     xmax, ymax, _ = vertices.max(axis=0)
#
#     max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
#     max_length = np.ceil(max_length).astype(np.int)
#
#     wall_maps = gen_map(vertices, faces, output_folder, img_filename_format=obstacle_map_filename_format)
#     wall_pts = np.array(np.where(wall_maps[0] == 0)).T
#     wall_convex_hull = ConvexHull(wall_pts)
#     wall_map_hull = np.zeros(wall_maps[0].shape).astype(np.uint8)
#     cv2.fillPoly(
#         wall_map_hull,
#         [wall_convex_hull.points[wall_convex_hull.vertices][:, ::-1].reshape((-1, 1, 2)).astype(np.int32)],
#         255,
#     )
#
#     for i_floor in range(len(floors)):
#         floor = floors[i_floor]
#         mask = np.abs(z_faces - floor) < 0.2
#         faces_new = np.array(faces_selected)[mask, :]
#
#         t = (vertices[faces_new][:, :, :2] + max_length) * 100
#         t = t.astype(np.int32)
#
#         floor_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))
#
#         cv2.fillPoly(floor_map, t, 1)
#
#         if add_clutter is True:  # Build clutter map
#             mask1 = ((z_faces_all - floor) < 2.0) * ((z_faces_all - floor) > 0.05)
#             faces_new1 = np.array(faces_selected_all)[mask1, :]
#
#             t1 = (vertices_all[faces_new1][:, :, :2] + max_length) * 100
#             t1 = t1.astype(np.int32)
#
#             clutter_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))
#             cv2.fillPoly(clutter_map, t1, 1)
#             floor_map = np.float32((clutter_map == 0) * (floor_map == 1))
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#         erosion = cv2.dilate(floor_map, kernel, iterations=2)
#         erosion = cv2.erode(erosion, kernel, iterations=2)
#         wall_map = wall_maps[i_floor]
#         wall_map = cv2.erode(wall_map, kernel, iterations=1)
#         erosion[wall_map == 0] = 0
#         erosion[wall_map_hull == 0] = 0  # crop using convex hull
#
#         cur_img = Image.fromarray((erosion * 255).astype(np.uint8))
#         # cur_img = Image.fromarray(np.flipud(cur_img))
#         cur_img.save(os.path.join(output_folder, trav_map_filename_format.format(i_floor)))
#
#
# INTERSECT_EDGE = 0
# INTERSECT_VERTEX = 1
#
#
# class Plane(object):
#     def __init__(self, orig, normal):
#         self.orig = orig
#         self.n = normal / np.linalg.norm(normal)
#
#     def __str__(self):
#         return "plane(o=%s, n=%s)" % (self.orig, self.n)
#
#
# def point_to_plane_dist(p, plane):
#     return np.dot((p - plane.orig), plane.n)
#
#
# def compute_triangle_plane_intersections(vertices, faces, tid, plane, dists, dist_tol=1e-8):
#     """
#     Compute the intersection between a triangle and a plane
#     Returns a list of intersections in the form
#     (INTERSECT_EDGE, <intersection point>, <edge>) for edges intersection
#     (INTERSECT_VERTEX, <intersection point>, <vertex index>) for vertices
#     This return between 0 and 2 intersections :
#     - 0 : the plane does not intersect the plane
#     - 1 : one of the triangle's vertices lies on the plane (so it just
#     "touches" the plane without really intersecting)
#     - 2 : the plane slice the triangle in two parts (either vertex-edge,
#     vertex-vertex or edge-edge)
#     """
#
#     # TODO: Use an edge intersection cache (we currently compute each edge
#     # intersection twice : once for each tri)
#
#     # This is to avoid registering the same vertex intersection twice
#     # from two different edges
#     vert_intersect = {vid: False for vid in faces[tid]}
#
#     # Iterate through the edges, cutting the ones that intersect
#     intersections = []
#     for e in ((faces[tid][0], faces[tid][1]), (faces[tid][0], faces[tid][2]), (faces[tid][1], faces[tid][2])):
#         v1 = vertices[e[0]]
#         d1 = dists[e[0]]
#         v2 = vertices[e[1]]
#         d2 = dists[e[1]]
#
#         if np.fabs(d1) < dist_tol:
#             # Avoid creating the vertex intersection twice
#             if not vert_intersect[e[0]]:
#                 # point on plane
#                 intersections.append((INTERSECT_VERTEX, v1, e[0]))
#                 vert_intersect[e[0]] = True
#         if np.fabs(d2) < dist_tol:
#             if not vert_intersect[e[1]]:
#                 # point on plane
#                 intersections.append((INTERSECT_VERTEX, v2, e[1]))
#                 vert_intersect[e[1]] = True
#
#         # If vertices are on opposite sides of the plane, we have an edge
#         # intersection
#         if d1 * d2 < 0:
#             # Due to numerical accuracy, we could have both a vertex intersect
#             # and an edge intersect on the same vertex, which is impossible
#             if not vert_intersect[e[0]] and not vert_intersect[e[1]]:
#                 # intersection factor (between 0 and 1)
#                 # here is a nice drawing :
#                 # https://ravehgonen.files.wordpress.com/2013/02/slide8.png
#                 # keep in mind d1, d2 are *signed* distances (=> d1 - d2)
#                 s = d1 / (d1 - d2)
#                 vdir = v2 - v1
#                 ipos = v1 + vdir * s
#                 intersections.append((INTERSECT_EDGE, ipos, e))
#
#     return intersections
#
#
# def gen_map(vertices, faces, output_folder, img_filename_format="floor_{}.png"):
#     xmin, ymin, _ = vertices.min(axis=0)
#     xmax, ymax, _ = vertices.max(axis=0)
#
#     max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
#     max_length = np.ceil(max_length).astype(np.int)
#
#     floors = [0.0]
#     print(floors)
#
#     floor_maps = []
#
#     for i_floor, floor in enumerate(floors):
#         dists = []
#         z = float(floor) + 0.5
#         cross_section = []
#         plane = Plane(np.array([0, 0, z]), np.array([0, 0, 1]))
#
#         for v in vertices:
#             dists.append(point_to_plane_dist(v, plane))
#
#         for i in tqdm(range(len(faces))):
#             res = compute_triangle_plane_intersections(vertices, faces, i, plane, dists)
#             if len(res) == 2:
#                 cross_section.append((res[0][1], res[1][1]))
#
#         floor_map = np.ones((2 * max_length * 100, 2 * max_length * 100))
#
#         for item in cross_section:
#             x1, x2 = (item[0][0] + max_length) * 100, (item[1][0] + max_length) * 100
#             y1, y2 = (item[0][1] + max_length) * 100, (item[1][1] + max_length) * 100
#
#             cv2.line(floor_map, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 0), thickness=2)
#
#         floor_maps.append(floor_map)
#         cur_img = Image.fromarray((floor_map * 255).astype(np.uint8))
#         # cur_img = Image.fromarray(np.flipud(cur_img))
#         img_filename = img_filename_format.format(i_floor)
#         cur_img.save(os.path.join(output_folder, img_filename))
#
#     return floor_maps
#
#
# def generate_trav_map(scene_name, scene_source, load_full_scene=True):
#     if scene_source not in SCENE_SOURCE:
#         raise ValueError("Unsupported scene source: {}".format(scene_source))
#     if scene_source == "IG":
#         scene_dir = get_ig_scene_path(scene_name)
#     elif scene_source == "CUBICASA":
#         scene_dir = get_cubicasa_scene_path(scene_name)
#     else:
#         scene_dir = get_3dfront_scene_path(scene_name)
#     random.seed(0)
#     scene = InteractiveIndoorScene(
#         scene_name, build_graph=False, texture_randomization=False, scene_source=scene_source
#     )
#     if not load_full_scene:
#         scene._set_first_n_objects(3)
#     s = Simulator(mode="headless", image_width=512, image_height=512, device_idx=0)
#     s.import_scene(scene)
#
#     if load_full_scene:
#         scene.open_all_doors()
#
#     for i in range(20):
#         s.step()
#
#     vertices_info, faces_info = s.renderer.dump()
#     s.disconnect()
#
#     if load_full_scene:
#         trav_map_filename_format = "floor_trav_{}.png"
#         obstacle_map_filename_format = "floor_{}.png"
#     else:
#         trav_map_filename_format = "floor_trav_no_obj_{}.png"
#         obstacle_map_filename_format = "floor_no_obj_{}.png"
#
#     gen_trav_map(
#         vertices_info,
#         faces_info,
#         output_folder=os.path.join(scene_dir, "layout"),
#         trav_map_filename_format=trav_map_filename_format,
#         obstacle_map_filename_format=obstacle_map_filename_format,
#     )
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Generate Traversability Map")
#     parser.add_argument("scene_names", metavar="s", type=str, nargs="+", help="The name of the scene to process")
#     parser.add_argument(
#         "--source", dest="source", help="Source of the scene, should be among [CUBICASA, IG, THREEDFRONT]"
#     )
#
#     args = parser.parse_args()
#     for scene_name in args.scene_names:
#         generate_trav_map(scene_name, args.source, False)
#         generate_trav_map(scene_name, args.source, True)
#
#
# if __name__ == "__main__":
#     main()
