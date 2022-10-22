"""Generate the (original) traversability maps based on where the floors are, but without taking
clutter into account.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_obj_np(
    filename_obj, normalization=False, texture_size=4, load_texture=False, texture_wrapping="REPEAT", use_bilinear=True
):
    """Load Wavefront .obj file into numpy array
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "v":
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    # load textures
    textures = None

    assert load_texture is False  # Since I commented out the block below
    # if load_texture:
    #     for line in lines:
    #         if line.startswith('mtllib'):
    #             filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
    #             textures = load_textures(filename_obj, filename_mtl, texture_size,
    #                                      texture_wrapping=texture_wrapping,
    #                                      use_bilinear=use_bilinear)
    #     if textures is None:
    #         raise Exception('Failed to load textures.')
    #     textures = textures.cpu().numpy()

    assert normalization is False  # Since I commented out the block below
    # # normalize into a unit cube centered zero
    # if normalization:
    #     vertices -= vertices.min(0)[0][None, :]
    #     vertices /= torch.abs(vertices).max()
    #     vertices *= 2
    #     vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces


def get_xy_floors(root, dist_threshold=-0.98):
    vertices, faces = load_obj_np(os.path.join(root, "mesh_z_up.obj"))
    z_faces = []
    z = np.array([0, 0, 1])
    faces_selected = []
    for face in tqdm(faces):
        normal = np.cross(vertices[face[2]] - vertices[face[1]], vertices[face[1]] - vertices[face[0]])
        dist = np.dot(normal, z) / np.linalg.norm(normal)
        if (dist_threshold is None) or ((dist_threshold is not None) and (dist < dist_threshold)):
            z_faces.append(vertices[face[0]][2])
            faces_selected.append(face)

    return np.array(z_faces), vertices, faces_selected


def gen_trav_map(mp3d_dir, add_clutter=False):
    """Generate traversability maps.

    Args:
        mp3d_dir: Root directory of Matterport3D or Gibson. Under this root directory should be
                subdirectories, each of which represents a model/environment. Within each
                subdirectory should be a file named 'mesh_z_up.obj'.
        add_clutter: Boolean for whether to generate traversability maps with or without clutter.
    """
    subdirectory_names = ["."]
    for scene in subdirectory_names:
        try:
            root = "{}/{}/".format(mp3d_dir, scene)
            print(root)
            with open(os.path.join(root, "floors.txt"), "r") as ff:
                floors = sorted(list(map(float, ff.readlines())))

            z_faces, vertices, faces_selected = get_xy_floors(root)
            z_faces_all, vertices_all, faces_selected_all = get_xy_floors(root, dist_threshold=None)

            xmin, ymin, _ = vertices.min(axis=0)
            xmax, ymax, _ = vertices.max(axis=0)

            max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
            max_length = np.ceil(max_length).astype(np.int)

            for i_floor in range(len(floors)):
                floor = floors[i_floor]
                mask = np.abs(z_faces - floor) < 0.2
                faces_new = np.array(faces_selected)[mask, :]

                t = (vertices[faces_new][:, :, :2] + max_length) * 100
                t = t.astype(np.int32)

                floor_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))

                cv2.fillPoly(floor_map, t, 1)

                if add_clutter is True:  # Build clutter map
                    mask1 = ((z_faces_all - floor) < 2.0) * ((z_faces_all - floor) > 0.05)
                    faces_new1 = np.array(faces_selected_all)[mask1, :]

                    t1 = (vertices_all[faces_new1][:, :, :2] + max_length) * 100
                    t1 = t1.astype(np.int32)

                    clutter_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))
                    cv2.fillPoly(clutter_map, t1, 1)
                    floor_map = np.float32((clutter_map == 0) * (floor_map == 1))

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                erosion = cv2.dilate(floor_map, kernel, iterations=2)
                erosion = cv2.erode(erosion, kernel, iterations=4)

                if add_clutter is True:
                    filename_format = "floor_trav_{}.png"
                else:
                    filename_format = "floor_trav_{}_v1.png"

                cur_img = Image.fromarray((erosion * 255).astype(np.uint8))
                # cur_img = Image.fromarray(np.flipud(cur_img))
                cur_img.save(os.path.join(mp3d_dir, scene, filename_format.format(i_floor)))
        except Exception as e:  # Which exception are we trying to ignore here?
            print(e)


if __name__ == "__main__":
    gen_trav_map(sys.argv[1], True)
