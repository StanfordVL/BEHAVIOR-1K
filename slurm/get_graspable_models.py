import pathlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import itertools

import sys
import os
import numpy as np
from omnigibson.utils.asset_utils import decrypted
from pxr import Usd

sys.path.append("/home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy")

from vid2scene_policy.data_collection.object_classifier import ObjectClassifier
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.objects.dataset_object import DatasetObject

OBJECT_CLASSIFIER = ObjectClassifier("/home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/data_collection/classifiers/object_embeddings.npz", "/home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/data_collection/classifiers")
MAX_GRIPPER_OPENING = 0.12  # 12cm - Stretch gripper max opening
MIN_OBJECT_HEIGHT = 0.03    # 3cm minimum object height

def is_model_graspable(dataset_name: str, cat: str, mdl: str) -> bool:
    """
    Check if an object is interesting.
    """
    usd_path = DatasetObject.get_usd_path(category=cat, model=mdl, dataset_name=dataset_name)
    encrypted_usd_path = usd_path.replace(".usd", ".encrypted.usd")
    if os.path.exists(usd_path):
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetDefaultPrim()
        bounding_box = np.array(prim.GetAttribute("ig:nativeBB").Get())
    elif os.path.exists(encrypted_usd_path):
        with decrypted(encrypted_usd_path) as fpath:
            stage = Usd.Stage.Open(fpath)
            prim = stage.GetDefaultPrim()
            bounding_box = np.array(prim.GetAttribute("ig:nativeBB").Get())
    else:
        print("Cant find USD file for", cat, mdl, dataset_name, usd_path)
        return False

    # Check the bounding box fits the gripper constraints
    # Smaller of X,Y must fit in gripper (grasp along longer axis), Z must be tall enough
    size_x, size_y, size_z = bounding_box
    min_xy = min(size_x, size_y)
    fits_gripper = min_xy <= MAX_GRIPPER_OPENING
    tall_enough = size_z >= MIN_OBJECT_HEIGHT

    return fits_gripper and tall_enough


def main():
    with ProcessPoolExecutor() as executor:
        futures = {}
        graspable_models = []
        with tqdm(desc="Queueing models") as pbar:
            for dataset_name in ("behavior-1k-assets", "spoc"):
                categories = get_all_object_categories(dataset_names=(dataset_name,))
                good_categories = [
                    category
                    for category in categories
                    if OBJECT_CLASSIFIER.is_graspable(category)
                ]
                for category in good_categories:
                    models = get_all_object_category_models(
                        category, dataset_names=(dataset_name,)
                    )
                    for model in models:
                        future = executor.submit(
                            is_model_graspable, dataset_name, category, model
                        )
                        futures[future] = (dataset_name, category, model)
                        pbar.update(1)

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing models"
        ):
            result = future.result()
            if result:
                graspable_models.append(futures[future])

    with open("graspable_models.json", "w") as f:
        json.dump(graspable_models, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
