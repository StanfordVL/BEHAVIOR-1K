# Code adapted from Fei Xia

import os

import cv2
import meshcut
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


def get_hist_num_faces(obj_filepath):
    vertices, faces = load_obj_np(obj_filepath)
    z_faces = []
    weights = []
    z = np.array([0, 0, 1])
    for face in tqdm(faces):
        normal = np.cross(vertices[face[2]] - vertices[face[1]], vertices[face[1]] - vertices[face[0]])
        dist = np.dot(normal, z) / np.linalg.norm(normal)
        if dist < -0.99:
            z_faces.append(vertices[face[0]][-1])
            a = np.linalg.norm(vertices[face[2]] - vertices[face[1]])
            b = np.linalg.norm(vertices[face[2]] - vertices[face[0]])
            c = np.linalg.norm(vertices[face[0]] - vertices[face[1]])
            s = (a + b + c) / 2
            area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
            weights.append(area)

    hist = np.histogram(np.array(z_faces), bins=100, weights=np.array(weights))
    return hist


def get_floor_height(hist, n_floors=1):
    heights = []

    for i in range(n_floors):
        pos = np.where(hist[0] == np.max(hist[0]))[0][0]
        height = (hist[1][pos] + hist[1][pos + 1]) / 2.0
        hist[0][np.abs(hist[1][1:] - height) < 0.5] = 0
        heights.append(height)
    return heights


def gen_map(obj_filepath, mesh_dir, img_filename_format="floor_{}.png"):
    vertices, faces = load_obj_np(obj_filepath)
    xmin, ymin, _ = vertices.min(axis=0)
    xmax, ymax, _ = vertices.max(axis=0)

    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)

    with open(os.path.join(mesh_dir, "floors.txt")) as f:
        floors = map(float, f.readlines())
        floors = sorted(floors)
        print(floors)
        for i_floor, floor in enumerate(floors):
            z = float(floor) + 0.5
            cross_section = meshcut.cross_section(vertices, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

            floor_map = np.ones((2 * max_length * 100, 2 * max_length * 100))

            for item in cross_section:
                for i in range(len(item) - 1):
                    x1, x2 = (item[i : i + 2, 0] + max_length) * 100
                    y1, y2 = (item[i : i + 2, 1] + max_length) * 100
                    cv2.line(floor_map, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)

            cur_img = Image.fromarray((floor_map * 255).astype(np.uint8))
            # cur_img = Image.fromarray(np.flipud(cur_img))
            img_filename = img_filename_format.format(i_floor)
            cur_img.save(os.path.join(mesh_dir, img_filename))

            write_yaml(mesh_dir, np.array(cur_img), img_filename, "floor_{}.yaml".format(i_floor), resolution=0.01)


def get_obj_filepath(mesh_dir):
    return mesh_dir + "/mesh_z_up.obj"


def get_n_floors(mesh_dir):
    return 1


# def get_n_floors(mesh_dir):
#    house_seg_filepaths = glob.glob(os.path.join(mesh_dir, 'house_segmentations', '*.house'))
#    assert len(house_seg_filepaths) == 1
#    with open(house_seg_filepaths[0]) as f:
#        content = f.readlines()
#    content = [x.strip() for x in content]#

#    n_levels = 0
#    for line in content:
#        if line.startswith('L '):
#            n_levels += 1
#    return n_levels


def fill_template(map_filepath, resolution, origin):  # NOTE: Copied from generate_map_yaml.py
    """Return a string that contains the contents for the yaml file, filling out the blanks where
    appropriate.

    Args:
        map_filepath: Absolute path to map file (e.g. PNG).
        resolution: Resolution of each pixel in the map in meters.
        origin: Uhhh.
    """
    template = """image: MAP_FILEPATH
resolution: RESOLUTION
origin: [ORIGIN_X, ORIGIN_Y, YAW]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    template = template.replace("MAP_FILEPATH", map_filepath)
    template = template.replace("RESOLUTION", str(resolution))
    template = template.replace("ORIGIN_X", str(origin[0]))
    template = template.replace("ORIGIN_Y", str(origin[1]))
    template = template.replace("YAW", str(origin[2]))
    return template


def write_yaml(
    mesh_dir, map_img, map_img_filepath, yaml_filename, resolution=0.01
):  # NOTE: Copied from generate_map_yaml.py
    origin_px_coord = (map_img.shape[0] / 2, map_img.shape[1] / 2)  # (row, col)
    cur_origin_map_coord = (
        -float(origin_px_coord[1]) * resolution,
        float(origin_px_coord[0] - map_img.shape[0]) * resolution,
        0.0,
    )  # (x, y, yaw)
    yaml_content = fill_template(map_img_filepath, resolution=resolution, origin=cur_origin_map_coord)

    cur_yaml_filepath = os.path.join(mesh_dir, yaml_filename)
    print("Writing to:", cur_yaml_filepath)
    with open(cur_yaml_filepath, "w") as f:
        f.write(yaml_content)


def generate_floorplan(mesh_dir):
    obj_filepath = get_obj_filepath(mesh_dir)

    # Generate floors.txt files
    print(mesh_dir)
    n_floors = get_n_floors(mesh_dir)  # Get number of floors
    hist = get_hist_num_faces(obj_filepath)

    hist = list(hist)
    hist[0] = np.nan_to_num(hist[0])
    hist = tuple(hist)

    heights = get_floor_height(hist, n_floors=n_floors)
    with open(os.path.join(mesh_dir, "floors.txt"), "w") as f:
        for height in heights:
            f.write("{}\n".format(height))

    gen_map(obj_filepath, mesh_dir)  # Generate floor maps


import sys

if __name__ == "__main__":
    generate_floorplan(sys.argv[1])
