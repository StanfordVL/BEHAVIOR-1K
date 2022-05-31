import os

import numpy as np
from bddl.object_taxonomy import ObjectTaxonomy

import igibson
from igibson.objects.usd_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets, get_ig_avg_category_specs

download_assets()

ABILITY_NAME = "stainable"
CATEGORIES = [
    "hardback",
    "notebook",
]

USE_ABILITY_TO_FETCH_CATEGORIES = False

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_categories():
    dir = os.path.join(igibson.ig_dataset_path, "objects")
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def get_obj(folder, **kwargs):
    return URDFObject(os.path.join(folder, os.path.basename(folder) + ".urdf"), name="obj", model_path=folder, **kwargs)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def main():
    # Collect the relevant categories.
    categories = CATEGORIES
    if USE_ABILITY_TO_FETCH_CATEGORIES:
        categories = []
        for cat in get_categories():
            # Check that the category has this label.
            klass = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
            if not klass:
                continue

            if not OBJECT_TAXONOMY.has_ability(klass, ABILITY_NAME):
                continue

            categories.append(cat)

    print("%d categories: %s" % (len(categories), ", ".join(categories)))

    # Now collect the actual objects.
    objects = []
    objects_by_category = {}
    for cat in categories:
        cd = get_category_directory(cat)
        objects_by_category[cat] = []
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)
            objects.append((cat, objdirfull))
            objects_by_category[cat].append(objdirfull)

    print("%d objects.\n" % len(objects))
    current_x = 0

    failed_objects = []

    batch_start = 0
    batch_size = 100
    max_attempts = 100

    s = Simulator(mode="gui_interactive", use_pb_gui=True)
    scene = EmptyScene()
    s.import_scene(scene)
    acs = get_ig_avg_category_specs()

    for cat, objdir in objects[batch_start : batch_start + batch_size]:
        avg_category_spec = acs.get(cat)
        cd = get_category_directory(cat)
        objdirfull = os.path.join(cd, objdir)

        obj = get_obj(objdirfull, fit_avg_dim_volume=True, avg_obj_dims=avg_category_spec, category=cat)
        s.import_object(obj)
        obj_pos = np.array([current_x, 0.0, 0.5])
        obj.set_position(obj_pos)
        obj.set_orientation(obj.sample_orientation())

        # Uncomment if you want to test for Dusty/Stained sampling
        # if object_states.Dusty in obj.states:
        #     attempts = 0
        #     while not obj.states[object_states.Dusty].set_value(True) and attempts < max_attempts:
        #         attempts += 1

        #     if not obj.states[object_states.Dusty].get_value():
        #         failed_objects.append((cat, objdir))
        # else:
        #     print(obj.category + " is not dustyable")

        x_sign = -1 if current_x < 0 else 1
        current_x = (abs(current_x) + avg_category_spec["size"][0] * 2) * x_sign * -1

    print(failed_objects)

    s.disconnect()


if __name__ == "__main__":
    main()
