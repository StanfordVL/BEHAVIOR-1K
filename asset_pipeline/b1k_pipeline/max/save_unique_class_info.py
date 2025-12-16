import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name

import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R
import json

rt = pymxs.runtime
local_coordsys = pymxs.runtime.Name("local")


def get_object_key(obj):
    """
    Get a unique key for an object.
    """
    return parse_name(obj.name).group("model_id")


def get_object_category(obj):
    """
    Get a unique key for an object.
    """
    return parse_name(obj.name).group("category")


def save_unique_class_infos():
    class_infos = {}
    for obj in rt.objects:
        # Skip everything that's not an editable poly
        if rt.classOf(obj) != rt.editable_poly:
            continue

        # Skip if object is not the base link
        link_name = parse_name(obj.name).group("link_name")
        if link_name is not None and link_name != "basename":
            continue

        model_id, category, material_name, eigvals = get_unique_class_info(obj)
        class_infos[model_id] = {
            "category": category,
            "material_name": material_name,
            "eigvals": eigvals.tolist(),
        }

    json.dump(class_infos, open("unique_class_info.json", "w"))


def get_unique_class_info(src_obj):
    """
    Uniquely identify a class by its material name and the eigenvalues of its covariance matrix.
    """
    X = np.array(
        rt.polyop.getVerts(
            src_obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(src_obj))
        )
    )
    n, m = X.shape
    u = np.mean(X, axis=0)
    # rt.messageBox("u: {}".format(u))
    B = X - u
    # rt.messageBox("B: {}".format(B))
    C = 1 / (n - 1) * B.T @ B
    eigvals, V = np.linalg.eig(C)
    sorted_idxs = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idxs]
    V = V[:, sorted_idxs]
    # rt.messageBox("eigvals: {}".format(eigvals))
    # rt.messageBox("V: {}".format(V))

    model_id = get_object_key(src_obj)
    category = get_object_category(src_obj)
    material_name = src_obj.material.name if src_obj.material else None

    return model_id, category, material_name, eigvals
    # json.dump(
    #     {
    #         "model_id": model_id,
    #         "material_name": material_name,
    #         "eigvals": eigvals.tolist(),
    #     },
    #     open("unique_class_info.json", "a"),
    # )


def save_unique_class_info_button():
    try:
        save_unique_class_infos()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Unique class info saved")


if __name__ == "__main__":
    save_unique_class_info_button()
