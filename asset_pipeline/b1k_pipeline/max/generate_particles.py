import json
import pymxs

rt = pymxs.runtime

import csv
import string
import random
# from b1k_pipeline.max.convex_decomposition import generate_convex_decompositions


def get_shape(shape, size):
    if shape == "cube":
        obj = rt.Box()
        obj.width = float(size)
        obj.height = float(size)
        obj.length = float(size)
    elif shape == "sphere":
        obj = rt.Sphere()
        obj.radius = float(size) / 2
    else:
        raise ValueError("Unknown shape " + shape)

    return obj


def generate_particle(name, shape, colors, size):
    objs = []
    for color in colors.split(";"):
        # Generate the object
        obj = get_shape(shape, size)

        # Assign a name
        model_id = "".join(random.choices(string.ascii_lowercase, k=6))
        obj.name = f"{name}-{model_id}-0"

        # Create the collision version
        col_obj = get_shape(shape, size)
        col_obj.name = obj.name + "-Mcollision"
        col_obj.parent = obj
        col_obj.isHidden = True

        # Create the material
        r, g, b = [int(x) for x in color.split(",")]
        mat = rt.VRayMtl()
        mat.diffuse = rt.Color(r, g, b)
        obj.material = mat

        objs.append(obj)
        print("Created", obj.name)
    return objs


def substance_generation_round_one():
    with open(r"D:\BEHAVIOR-1K\asset_pipeline\metadata\substance_generation.csv") as f:
        particles = []
        for row in csv.DictReader(f):
            if row["Request"] != "GENERATE" or row["Shape"] not in ("cube", "sphere"):
                continue
            particles.extend(
                generate_particle(
                    row["Name"], row["Shape"], row["Color"], row["Diameter (mm)"]
                )
            )

        for i, particle in enumerate(particles):
            particle.position = rt.Point3(i * 100, 0, 0)


def substance_generation_round_two():
    with open(
        r"D:/BEHAVIOR-1K/asset_pipeline/metadata/diced_particle_systems_colors.json"
    ) as f:
        data = json.load(f)

    particles = []
    for particle_name, particle_info in data.items():
        hex_color = particle_info["color"]
        rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        rgb_color_str = f"{rgb_color[0]},{rgb_color[1]},{rgb_color[2]}"
        size_cm = particle_info["llm_size_cm"]
        size_mm = size_cm * 10
        (particle,) = generate_particle(particle_name, "cube", rgb_color_str, size_mm)
        # generate_convex_decompositions(particle, preferred_method="chull")
        particles.append(particle)

    for i, particle in enumerate(particles):
        particle.position = rt.Point3(i * 100, 3000, 0)


if __name__ == "__main__":
    substance_generation_round_two()
