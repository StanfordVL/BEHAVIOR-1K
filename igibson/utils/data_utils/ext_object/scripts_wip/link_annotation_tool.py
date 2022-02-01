import itertools
import json
import os

import numpy as np

from bddl.object_taxonomy import ObjectTaxonomy
from pynput import keyboard

import igibson
from igibson.objects.usd_object import URDFObject
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets

download_assets()

ABILITY_NAME = "cleaningTool"
SYNSETS = [
    "alarm.n.02",
    "printer.n.03",
    "facsimile.n.02",
    "scanner.n.02",
    "modem.n.01",
]
CATEGORIES = [
    "broom",
    "carpet_sweeper",
    "scraper",
    "scrub_brush",
    "toothbrush",
    "vacuum",
]
MODE = "synset"  # "ability, "category"

LINK_NAME = "toggle_button"
IS_CUBOID = False
SKIP_EXISTING = False

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_categories():
    dir = os.path.join(igibson.ig_dataset_path, "objects")
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def get_obj(folder):
    return URDFObject(os.path.join(folder, os.path.basename(folder) + ".urdf"), name="obj", model_path=folder)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def get_corner_positions(base, rotation, size):
    quat = p.getQuaternionFromEuler(rotation)
    options = [-1, 1]
    outputs = []
    for pos in itertools.product(options, options, options):
        res = p.multiplyTransforms(base, quat, np.array(pos) * size / 2.0, [0, 0, 0, 1])
        outputs.append(res)
    return outputs


def main():
    # Collect the relevant categories.
    categories = CATEGORIES
    if MODE == "ability":
        categories = []
        for cat in get_categories():
            # Check that the category has this label.
            klass = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
            if not klass:
                continue

            if not OBJECT_TAXONOMY.has_ability(klass, ABILITY_NAME):
                continue

            categories.append(cat)
    elif MODE == "synset":
        categories = []
        for synset in SYNSETS:
            categories.extend(OBJECT_TAXONOMY.get_igibson_categories(synset))
        categories = set(categories) & set(get_categories())

    print("%d categories: %s" % (len(categories), ", ".join(categories)))

    # Now collect the actual objects.
    objects = []
    objects_by_category = {}
    for cat in categories:
        cd = get_category_directory(cat)
        objects_by_category[cat] = []
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)
            objects.append(objdirfull)
            objects_by_category[cat].append(objdirfull)

    print("%d objects.\n" % len(objects))

    for cat in categories:
        cd = get_category_directory(cat)
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)

            mfn = get_metadata_filename(objdirfull)
            with open(mfn, "r") as mf:
                meta = json.load(mf)

            offset = np.array([0.0, 0.0, 0.0])
            size = np.array([0.0, 0.0, 0.0])
            rotation = np.array([0.0, 0.0, 0.0])

            existing = False
            if "links" in meta and LINK_NAME in meta["links"]:
                print("%s/%s already has the requested link." % (cat, objdir))

                if SKIP_EXISTING:
                    continue

                existing = True
                offset = np.array(meta["links"][LINK_NAME]["xyz"])
                if IS_CUBOID:
                    size = np.array(meta["links"][LINK_NAME]["size"])
                    rotation = np.array(meta["links"][LINK_NAME]["rpy"])

            s = Simulator(mode="gui_interactive", use_pb_gui=True)
            scene = EmptyScene()
            s.import_scene(scene)
            obj = get_obj(objdirfull)
            s.import_object(obj)
            obj_pos = np.array([0.0, 0.0, 1.0])
            obj.set_position(obj_pos)

            dim = max(obj.bounding_box)
            marker_size = dim / 100.0
            steps = [dim * 0.1, dim * 0.01, dim * 0.001]
            rot_steps = [np.deg2rad(1), np.deg2rad(5), np.deg2rad(10)]

            m = VisualMarker(radius=marker_size, rgba_color=[0, 0, 1, 0.5])
            s.import_object(m)
            if IS_CUBOID:
                initial_poses = get_corner_positions(obj_pos + offset, rotation, size)
                markers = [VisualMarker(radius=marker_size, rgba_color=[0, 1, 0, 0.5]) for _ in initial_poses]
                [s.import_object(m) for m in markers]
                for marker, (pos, orn) in zip(markers, initial_poses):
                    marker.set_position_orientation(pos, orn)

            # if existing:
            #     e = VisualMarker(radius=0.02, rgba_color=[1, 0, 0, 0.5])
            #     s.import_object(e)
            #     e.set_position(obj_pos + offset)

            step_size = steps[1]
            rot_step_size = rot_steps[1]
            done = False
            while not done:
                with keyboard.Events() as events:
                    for event in events:
                        if (
                            event is None
                            or not isinstance(event, keyboard.Events.Press)
                            or not hasattr(event.key, "char")
                        ):
                            continue

                        if event.key.char == "w":
                            print("Moving forward one")
                            offset += np.array([0, 1, 0]) * step_size
                        elif event.key.char == "a":
                            print("Moving left one")
                            offset += np.array([-1, 0, 0]) * step_size
                        elif event.key.char == "s":
                            print("Moving back one")
                            offset += np.array([0, -1, 0]) * step_size
                        elif event.key.char == "d":
                            print("Moving right one")
                            offset += np.array([1, 0, 0]) * step_size
                        elif event.key.char == "q":
                            print("Moving up one")
                            offset += np.array([0, 0, 1]) * step_size
                        elif event.key.char == "z":
                            print("Moving down one")
                            offset += np.array([0, 0, -1]) * step_size
                        elif event.key.char == "1":
                            print("Sizing forward one")
                            size += np.array([0, 1, 0]) * step_size
                        elif event.key.char == "2":
                            print("Sizing back one")
                            size += np.array([0, -1, 0]) * step_size
                        elif event.key.char == "4":
                            print("Sizing left one")
                            size += np.array([-1, 0, 0]) * step_size
                        elif event.key.char == "5":
                            print("Sizing right one")
                            size += np.array([1, 0, 0]) * step_size
                        elif event.key.char == "7":
                            print("Sizing up one")
                            size += np.array([0, 0, 1]) * step_size
                        elif event.key.char == "8":
                            print("Sizing down one")
                            size += np.array([0, 0, -1]) * step_size
                        elif event.key.char == "t":
                            print("Rotation +X one")
                            rotation += np.array([1, 0, 0]) * rot_step_size
                        elif event.key.char == "y":
                            print("Rotation -X one")
                            rotation += np.array([-1, 0, 0]) * rot_step_size
                        elif event.key.char == "u":
                            print("Rotation +Y one")
                            rotation += np.array([0, 1, 0]) * rot_step_size
                        elif event.key.char == "i":
                            print("Rotation -Y one")
                            rotation += np.array([0, -1, 0]) * rot_step_size
                        elif event.key.char == "o":
                            print("Rotation +Z one")
                            rotation += np.array([0, 0, 1]) * rot_step_size
                        elif event.key.char == "p":
                            print("Rotation -Z one")
                            rotation += np.array([0, 0, -1]) * rot_step_size
                        elif event.key.char == "h":
                            print("Step to 0.1")
                            step_size = steps[0]
                            rot_step_size = rot_steps[0]
                        elif event.key.char == "j":
                            print("Step to 0.01")
                            step_size = steps[1]
                            rot_step_size = rot_steps[1]
                        elif event.key.char == "k":
                            print("Step to 0.001")
                            step_size = steps[2]
                            rot_step_size = rot_steps[2]
                        elif event.key.char == "b":
                            print("Updating box to match bounding box.")
                            offset = np.array([0.0, 0.0, 0.0])
                            rotation = np.array([0.0, 0.0, 0.0])
                            size = np.array(obj.bounding_box, dtype=float)
                        elif event.key.char == "c":
                            done = True
                            break

                        print("New position:", offset)
                        m.set_position(obj_pos + offset)
                        if IS_CUBOID:
                            print("New rotation:", rotation)
                            print("New size:", size)
                            print("")
                            poses = get_corner_positions(obj_pos + offset, rotation, size)
                            for marker, (pos, orn) in zip(markers, poses):
                                marker.set_position_orientation(pos, orn)

            # Record it into the meta file.
            if "links" not in meta:
                meta["links"] = dict()

            body_ids = obj.get_body_ids()
            assert len(body_ids) == 1, "Only single-body objects are supported."
            body_id = body_ids[0]
            dynamics_info = p.getDynamicsInfo(body_id, -1)
            inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]

            rel_position, rel_orn = p.multiplyTransforms(
                offset, p.getQuaternionFromEuler(rotation), inertial_pos, inertial_orn
            )

            if IS_CUBOID:
                meta["links"][LINK_NAME] = {
                    "geometry": "box",
                    "size": list(size),
                    "xyz": list(rel_position),
                    "rpy": list(p.getEulerFromQuaternion(rel_orn)),
                }
            else:
                meta["links"][LINK_NAME] = {"geometry": None, "size": None, "xyz": list(rel_position), "rpy": None}

            with open(mfn, "w") as mf:
                json.dump(meta, mf)
                print("Updated %s" % mfn)

            input("Hit enter to continue.")

            s.disconnect()


if __name__ == "__main__":
    main()
