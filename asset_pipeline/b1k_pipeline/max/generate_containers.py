import collections
import os
import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import b1k_pipeline.utils

import csv
import string
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap
import tqdm


def generate_texture(label, path):
    # create an image
    out = Image.new("RGB", (1000, 500), (255, 255, 255))
    HEIGHT = out.height
    WIDTH = out.width

    lines = textwrap.wrap(label, 12)
    assert len(lines) <= 3
    text = "\n".join(lines)
    FONT_SIZE = 172
    fnt = ImageFont.truetype("impact.ttf", FONT_SIZE, encoding="unic")
    d = ImageDraw.Draw(out)
    d.multiline_text(
        (WIDTH // 2, HEIGHT // 2),
        text,
        font=fnt,
        anchor="mm",
        align="center",
        fill=(30, 30, 30),
    )

    out.save(path)


def generate_container(cat, label, base_model, last_seen_id):
    # Find the stuff belonging to the base
    base_objs = [x for x in rt.objects if base_model in x.name]
    children = [x for x in rt.objects if x not in base_objs and x.parent in base_objs]
    assert base_objs, f"Could not find any objects with ID {base_model}"

    # Compile the basename for replacement
    base_cat = b1k_pipeline.utils.parse_name(base_objs[0].name).group("category")
    base_name = f"{base_cat}-{base_model}"

    # What will this object be called?
    new_id = "".join(random.choices(string.ascii_lowercase, k=6))
    new_name = f"{cat}-{new_id}"

    # Generate the new objects. Everything that contains the base name will be copied.
    # Everything that doesn't can get instanced.
    base_copies = []  # (old, new, expectedparentname)
    for base_obj in base_objs:
        success, base_copy = rt.maxOps.cloneNodes(
            base_obj,
            cloneType=rt.name("copy"),
            newNodes=pymxs.byref(None),
        )
        assert success, f"Could not clone {base_obj.name}"
        (base_copy,) = base_copy
        base_copy.name = base_obj.name.replace(base_name, new_name)
        expected_parent_name = (
            base_obj.parent.name.replace(base_name, new_name)
            if base_obj.parent is not None
            else None
        )
        base_copies.append((base_obj, base_copy, expected_parent_name))

    children_copies = []  # (old, new, expectedparentname)
    for child in children:
        success, child_copy = rt.maxOps.cloneNodes(
            child,
            cloneType=rt.name("instance"),
            newNodes=pymxs.byref(None),
        )
        assert success, f"Could not clone {child.name}"
        (child_copy,) = child_copy

        # Compute the new model ID for the child
        child_model = b1k_pipeline.utils.parse_name(child.name).group("model_id")
        child_old_iid = b1k_pipeline.utils.parse_name(child.name).group("instance_id")
        child_new_iid = last_seen_id[child_model] + 1
        last_seen_id[child_model] = child_new_iid

        # Update the name
        child_copy.name = child.name.replace(f"-{child_old_iid}-", f"-{child_new_iid}-")
        expected_parent_name = (
            child.parent.name.replace(base_name, new_name)
            if child.parent is not None
            else None
        )
        children_copies.append((child, child_copy, expected_parent_name))

    # Check that everything that shows up in the parent is appropriately parented
    for _, item, parent_name in base_copies + children_copies:
        if not parent_name:
            continue
        parent_candidates = [x for _, x, _ in base_copies if x.name == parent_name]
        assert parent_candidates, f"Could not find parent {parent_name} for {item.name}"
        (parent,) = parent_candidates
        item.parent = parent

    # Replace any uppers with an instance of the lower
    for i in range(len(base_copies)):
        entry = base_copies[i]
        item = entry[1]
        if b1k_pipeline.utils.parse_name(item.name).group("joint_side") != "upper":
            continue
        lower_name = item.name.replace("upper", "lower")
        lower_candidates = [
            x for _, x, _ in base_copies + children_copies if x.name == lower_name
        ]
        assert lower_candidates, f"Could not find lower {lower_name} for {item.name}"
        (lower,) = lower_candidates

        success, lower_copy = rt.maxOps.cloneNodes(
            lower,
            cloneType=rt.name("instance"),
            newNodes=pymxs.byref(None),
        )
        assert success, f"Could not clone {lower.name}"
        (lower_copy,) = lower_copy
        lower_copy.transform = item.transform
        lower_copy.name = item.name
        rt.delete(item)

        base_copies[i] = (entry[0], lower_copy, entry[2])

    # Find the label object
    for _, item, _ in base_copies + children_copies:
        if "label" in item.name:
            label_obj = item
            break
    else:
        raise AssertionError("Could not find label object")

    # Generate the texture
    texture_path = os.path.join(rt.maxFilePath, "textures", f"{cat}.png")
    generate_texture(label, texture_path)

    # Create a material
    mat = rt.VRayMtl()
    label_obj.material = mat

    # Assign the bitmap
    bmp = rt.VrayBitmap()
    bmp.HDRIMapName = texture_path
    mat.texmap_diffuse = bmp

    print("Created", cat)
    return [obj for _, obj, _ in base_copies + children_copies]


def main():
    with open(r"D:\BEHAVIOR-1K\asset_pipeline\metadata\container_generation.csv") as f:
        containers = []
        last_seen_id = collections.defaultdict(int)
        for row in tqdm.tqdm(list(csv.DictReader(f))):
            if row["Action"] != "GENERATE":
                continue
            containers.append(
                (
                    row["Actual Object"],
                    generate_container(
                        row["Category"],
                        row["Label"],
                        row["Actual Object"],
                        last_seen_id,
                    ),
                )
            )

        position_ctr = collections.Counter()
        for kind, container_set in containers:
            position_ctr[kind] += 1
            for container in container_set:
                if container.parent is None:
                    container.position += rt.Point3(0, position_ctr[kind] * 400, 0)


if __name__ == "__main__":
    main()
