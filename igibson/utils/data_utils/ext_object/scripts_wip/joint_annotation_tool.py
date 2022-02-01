import json
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from bddl.object_taxonomy import ObjectTaxonomy
from pynput import keyboard

import igibson
import igibson.object_states.open as open_state
from igibson.external.pybullet_tools import utils
from igibson.objects.usd_object import URDFObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils import urdf_utils
from igibson.utils.assets_utils import download_assets

download_assets()

ABILITY_NAME = "openable"
META_FIELD = "openable_joint_ids"
BOTH_SIDES_META_FIELD = "openable_both_sides"
SKIP_EXISTING = True
ALLOWED_JOINT_TYPES = ["revolute", "prismatic", "continuous"]

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_categories():
    dir = os.path.join(igibson.ig_dataset_path, "objects")
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def get_urdf(objdir):
    return os.path.join(objdir, os.path.basename(objdir) + ".urdf")


def get_obj(objdir):
    return URDFObject(get_urdf(objdir), name="obj", model_path=objdir)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def get_joint_selection(s, objdirfull, offline_joints):
    print("Starting %s" % objdirfull)
    obj = get_obj(objdirfull)
    s.import_object(obj)
    obj_pos = np.array([0.0, 0.0, 1.0])
    obj.set_position(obj_pos)

    body_ids = obj.get_body_ids()
    assert len(body_ids) == 1, "Only single-body objects are supported."
    bid = body_ids[0]
    joints = utils.get_joints(bid)
    joint_infos = [utils.get_joint_info(bid, j) for j in joints]
    relevant_joints = [ji for ji in joint_infos if ji.jointType in open_state._JOINT_THRESHOLD_BY_TYPE.keys()]
    relevant_joint_names = [ji.jointName for ji in relevant_joints]
    assert set(offline_joints) == set(relevant_joint_names), "Offline IDs mismatched with Online IDs: %r vs %r" % (
        offline_joints,
        relevant_joint_names,
    )

    accepted_joint_infos = []
    both_sides = False
    for joint in relevant_joints:
        if joint.jointUpperLimit <= joint.jointLowerLimit:
            print("Bounds seem funky.")

        toggle_time = time.time()
        reverse = False
        percentages = [0, 0.25, 0.5, 0.75, 1.0]
        i = 0
        while True:
            # Repeatedly move the joint for some visualization.
            if time.time() - toggle_time > 0.5:
                toggle_time = time.time()
                i += 1
                idx = i % len(percentages)
                if reverse:
                    idx = len(percentages) - idx - 1
                percentage = percentages[idx]
                pos = (1 - percentage) * joint.jointLowerLimit + percentage * joint.jointUpperLimit
                p.resetJointState(bid, joint.jointIndex, pos)

            with keyboard.Events() as events:
                event = events.get(0.5)
                if event is None or not isinstance(event, keyboard.Events.Press) or not hasattr(event.key, "char"):
                    continue

                elif event.key.char == "y":
                    accepted_joint_infos.append((joint, -1 if reverse else 1))
                    break

                elif event.key.char == "n":
                    break

                elif event.key.char == "r":
                    reverse = not reverse
                    print("Reverse:", reverse)

                elif event.key.char == "b":
                    both_sides = not both_sides
                    print("Both ways:", both_sides)

    obj.set_position([300, 300, 5])
    return accepted_joint_infos, both_sides


def main():
    # Collect the relevant categories.
    categories = []
    for cat in get_categories():
        # Check that the category has this label.
        klass = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
        if not klass:
            continue

        if not OBJECT_TAXONOMY.has_ability(klass, ABILITY_NAME):
            continue

        categories.append(cat)

    print("%d categories with ability %s: %s" % (len(categories), ABILITY_NAME, ", ".join(categories)))

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

    print("%d objects with ability %s.\n" % (len(objects), ABILITY_NAME))

    # Now collect the objects that have relevant joints.
    objects_and_joints = {}
    objects_with_unusable_joints = []
    for objdirfull in objects:
        urdf_path = get_urdf(objdirfull)

        # Load the object tree
        ot = ET.parse(urdf_path)
        _, _, joint_map, _ = urdf_utils.parse_urdf(ot)

        joint_types_by_name = {key: info[2] for key, info in joint_map.items()}

        # Filter out the joints that are not relevant to our use case.
        joint_names = [name for name, joint_type in joint_types_by_name.items() if joint_type in ALLOWED_JOINT_TYPES]

        objects_and_joints[objdirfull] = joint_names

        if len(joint_names) != len(joint_types_by_name):
            objects_with_unusable_joints.append(objdirfull)

    # Check which categories have fully unopenable vs. partially unopenable objects.
    fully_jointless = []
    partially_jointless = []
    fully_jointed = []
    for cat in categories:
        num_objects = len(objects_by_category[cat])
        num_jointless_objects = sum(1 for obj in objects_by_category[cat] if len(objects_and_joints[obj]) == 0)

        if num_jointless_objects == 0:
            fully_jointed.append(cat)
        elif num_jointless_objects == num_objects:
            fully_jointless.append(cat)
        else:
            partially_jointless.append(cat)

    print("Category statistics:")
    print("%d fully jointless categories: %s" % (len(fully_jointless), ", ".join(fully_jointless)))
    print("%d partially jointless categories: %s" % (len(partially_jointless), ", ".join(partially_jointless)))
    print("%d fully jointed categories: %s\n" % (len(fully_jointed), ", ".join(fully_jointed)))

    print("Object statistics:")
    print(
        "%d objects have unusable joints: %s"
        % (len(objects_with_unusable_joints), ", ".join(objects_with_unusable_joints))
    )
    print(
        "%d objects have 0 relevant joints. Consider disabling ability."
        % sum(1 for obj, joints in objects_and_joints.items() if len(joints) == 0)
    )
    print(
        "%d objects have 1 relevant joint. Presuming these are correct."
        % sum(1 for obj, joints in objects_and_joints.items() if len(joints) == 1)
    )
    print(
        "%d objects have multiple relevant joints. Will annotate.\n"
        % sum(1 for obj, joints in objects_and_joints.items() if len(joints) > 1)
    )

    relevant_joint_counts = Counter()
    for obj, joints in objects_and_joints.items():
        relevant_joint_counts[len(joints)] += 1
    sorted_joint_counts = sorted(relevant_joint_counts.items(), key=lambda x: x[0])
    print("Objects per joint count:\n%s" % "\n".join(["%d joints: %d" % pair for pair in sorted_joint_counts]))

    s = Simulator(mode="gui_interactive", use_pb_gui=True)
    scene = EmptyScene()
    s.import_scene(scene)
    i = 0
    for objdirfull, joints in objects_and_joints.items():
        if len(joints) == 0:
            continue

        mfn = get_metadata_filename(objdirfull)
        with open(mfn, "r") as mf:
            meta = json.load(mf)

        existing = False
        if META_FIELD in meta:
            print("%s already has the requested link." % objdirfull)
            existing = True
            processed_allowed_joints = meta[META_FIELD]
            both_sides = meta[BOTH_SIDES_META_FIELD] if BOTH_SIDES_META_FIELD in meta else False

        if not existing or not SKIP_EXISTING:
            joint_names_matching = [bytes("obj_" + j, encoding="utf-8") for j in joints]
            allowed_joints, both_sides = get_joint_selection(s, objdirfull, joint_names_matching)
            processed_allowed_joints = [
                (ji.jointIndex, str(ji.jointName[4:], encoding="utf-8"), reverse) for ji, reverse in allowed_joints
            ]

        # Do some final validation
        if len(processed_allowed_joints) == 0:
            print("Object %s has joints but none were useful for openable." % objdirfull)

        meta[META_FIELD] = processed_allowed_joints
        meta[BOTH_SIDES_META_FIELD] = both_sides

        with open(mfn, "w") as mf:
            json.dump(meta, mf)
            i += 1
            print("%s saved." % mfn)

    print(
        "%d out of %d non-jointless objects now have annotations."
        % (i, sum(1 for obj, joints in objects_and_joints.items() if len(joints) >= 1))
    )


if __name__ == "__main__":
    main()
