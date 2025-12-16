from b1k_pipeline.utils import parse_name

import re
from collections import defaultdict

import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R
import json

rt = pymxs.runtime
local_coordsys = pymxs.runtime.Name("local")

EIGVAL_TOL = 1e-3


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


def change_object_category(obj, new_category):
    """
    Change the category of an object.
    """
    # Get the old category
    old_category = get_object_category(obj)

    # Change the category
    obj.name = obj.name.replace(old_category, new_category)


def change_object_key(obj, new_key):
    """
    Change the key of an object.
    """
    # Get the old key
    old_key = get_object_key(obj)

    # Change the key
    obj.name = obj.name.replace(old_key, new_key)


def load_unique_class_infos():
    saved_infos = json.load(open("unique_class_info.json", "r"))
    for obj in rt.objects:
        # Skip everything that's not an editable poly
        if rt.classOf(obj) != rt.editable_poly:
            continue

        # Skip if object is not the base link
        link_name = parse_name(obj.name).group("link_name")
        if link_name is not None and link_name != "basename":
            continue

        model_id, category, material_name, eigvals = get_unique_class_info(obj)
        # Get the class with the same material name and eigvals within EIGVAL_TOL
        # i = 0
        for saved_model_id, saved_info in saved_infos.items():
            # i+=1
            if saved_info["material_name"] == material_name and np.allclose(
                saved_info["eigvals"], eigvals, atol=EIGVAL_TOL
            ):
                # Change the category
                rt.messageBox(
                    "Changing key of {} to {}".format(obj.name, saved_model_id)
                )
                change_object_key(obj, saved_model_id)
                rt.messageBox(
                    "Changed key of {} to {}".format(obj.name, saved_model_id)
                )
                # if i>1:
                # assert False
                break


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


def load_unique_class_info_button():
    try:
        load_unique_class_infos()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Unique class info loaded")


if __name__ == "__main__":
    load_unique_class_info_button()
