import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

"""
Usage python combine_texture.py <data path> <uuid> <texture unit size>
"""

data_path = sys.argv[1]  # enter matterport data path
uuid = sys.argv[2]  # enter UUID of the matterport model
files = os.listdir(data_path)
files = [file for file in files if file.startswith(uuid) and file.endswith("jpg")]  # select all texture files
grid_size = int(np.ceil(np.sqrt(len(files))))

target_texture_grid_size = int(
    sys.argv[3]
)  # select a target texture size, the raw size is 2048, choose between 512,1024
# or 2048 depending on how many images you have

combined_img = np.zeros((grid_size * target_texture_grid_size, grid_size * target_texture_grid_size, 3))
files = sorted(files)
# print(files)
for num_file, file in tqdm(enumerate(files)):
    img = np.array(
        Image.open(os.path.join(data_path, file)).resize((target_texture_grid_size, target_texture_grid_size))
    )
    i = grid_size - 1 - num_file // grid_size
    j = num_file % grid_size
    combined_img[
        i * target_texture_grid_size : (i + 1) * target_texture_grid_size,
        j * target_texture_grid_size : (j + 1) * target_texture_grid_size,
        :,
    ] = img

Image.fromarray(combined_img.astype(np.uint8)).save(os.path.join(data_path, "combined.jpg"))

# change the obj file to use the combined texture, and rename to mesh_z_up

vt_mtl_dict = {}

with open("{}/{}.obj".format(data_path, uuid)) as f:
    for line in f:
        if line.startswith("usemtl"):
            current_mtl = int(line[-8:-5])

        if line.startswith("f"):
            face = line.strip().split()
            for vvt in face[1:]:
                vt_mtl_dict[int(vvt.split("/")[1])] = current_mtl

n_vt = 0
fo = open("{}/{}.obj".format(data_path, "mesh_z_up"), "w")
mtl_set = False
with open("{}/{}.obj".format(data_path, uuid)) as f:
    for line in tqdm(f):
        if line.startswith("usemtl"):
            current_mtl = int(line[-8:-5])
            # print(int(current_mtl))
            if not mtl_set:
                fo.write("usemtl combined\n")
            mtl_set = True
        elif line.startswith("vt"):
            n_vt += 1
            n_mtl = vt_mtl_dict[n_vt]
            i = n_mtl // grid_size
            j = n_mtl % grid_size
            _, u, v = line.strip().split()
            u = float(u)
            v = float(v)
            new_u = u / grid_size + j / grid_size
            new_v = v / grid_size + i / grid_size
            # if n_mtl == 0:
            fo.write("vt {} {}\n".format(new_u, new_v))
            # else:
            #    fo.write('vt {} {}\n'.format(0, 0))
        elif line.startswith("mtllib"):
            fo.write("mtllib default.mtl\n")
        elif line.startswith("g"):
            pass
        else:
            fo.write(line)

fo.close()


# create a mtl file
with open("{}/default.mtl".format(data_path), "w") as fo:
    fo.write(
        """
newmtl combined
Ns 0.000000
Ka 1.000000 1.000000 1.000000
Kd 0.000000 0.000000 0.000000
Ks 1.000000 1.000000 1.000000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd combined.jpg"""
    )
