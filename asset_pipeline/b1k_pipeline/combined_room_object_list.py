import argparse

from collections import Counter, defaultdict
import glob
import json
import os

import numpy as np
import b1k_pipeline.utils
from nltk.corpus import wordnet as wn
import yaml
import csv


def get_approved_room_types(pipeline_fs):
    approved = []
    with pipeline_fs.open("metadata/allowed_room_types.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            approved.append(row[0])

    return approved


def main():
    scenes = {}
    skipped_files = []

    not_approved_rooms = defaultdict(set)

    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        approved_rooms = set(get_approved_room_types(pipeline_fs))

        # Get the list of targets
        with pipeline_fs.open("params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = params["scenes"]
            final_scenes = [x.replace("scenes/", "") for x in params["final_scenes"]]

        outgoing_portals = {}
        incoming_portals = {}

        # Add the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("object_list.json", "r") as f:
                    object_list = json.load(f)

            if not object_list["success"]:
                skipped_files.append(target)
                continue

            scene_name = target.split("/")[-1]
            scene_info = object_list["objects_by_room"]

            room_types = {rm_name.rsplit("_", 1)[0] for rm_name in scene_info.keys()}
            this_not_approved_rooms = room_types - approved_rooms
            for rm in this_not_approved_rooms:
                not_approved_rooms[scene_name].add(rm)

            scene_content_info = {}
            for rm, models in scene_info.items():
                contents = Counter()

                for model, cnt in models.items():
                    contents[model] += cnt

                scene_content_info[rm] = dict(contents)

            scenes[scene_name] = scene_content_info

            outgoing_portals[scene_name] = object_list["outgoing_portals"]
            incoming_portals[scene_name] = object_list["incoming_portal"]

        # Merge the stuff
        room_collision_errors = []
        for base, room_objects in scenes.items():
            for addition_scene, portal_info in outgoing_portals[base].items():
                # Check that the corresponding scene also has the incoming portal
                assert incoming_portals[
                    addition_scene
                ], f"Scene {addition_scene} has no incoming portal but is linked to {base}"

                # Assert the portal sizes are the same
                outgoing_size = portal_info[2]
                incoming_size = incoming_portals[addition_scene][2]
                assert np.allclose(
                    outgoing_size, incoming_size
                ), f"Scene {addition_scene} has incoming portal size {incoming_size} but {base} has outgoing portal size {outgoing_size}"

                # Add all the rooms
                base_keys = set(room_objects.keys())
                add_keys = set(scenes[addition_scene].keys())
                if not base_keys.isdisjoint(add_keys):
                    room_collision_errors.append(
                        f"Rooms colliding between {base} and {addition_scene}: {base_keys} vs {add_keys}"
                    )
                scenes[base].update(scenes[addition_scene])

        assert not room_collision_errors, "\n".join(room_collision_errors)

        # Delete any exclusion scenes
        scenes_to_exclude = [x for x in scenes if x not in final_scenes]
        for scene in scenes_to_exclude:
            del scenes[scene]

        # Check that the IDs of all the rooms are contiguous
        non_contiguous_rooms = defaultdict(dict)
        for scene_name, scene_info in scenes.items():
            try:
                room_type_keys = defaultdict(set)
                for rm_name in scene_info.keys():
                    if "_" not in rm_name:
                        print(
                            "Found invalid room name", rm_name, "in scene", scene_name
                        )
                        continue
                    room_type, room_id = rm_name.rsplit("_", 1)
                    room_type_keys[room_type].add(int(room_id))

                for room_type, room_ids in room_type_keys.items():
                    if room_ids != set(range(len(room_ids))):
                        non_contiguous_rooms[scene_name][room_type] = sorted(room_ids)
            except:
                raise ValueError(
                    "Error parsing room name", rm_name, "in scene", scene_name
                )

        success = (
            len(skipped_files) == 0
            and len(not_approved_rooms) == 0
            and len(non_contiguous_rooms) == 0
        )
        with pipeline_fs.pipeline_output() as pipeline_output_fs:
            with pipeline_output_fs.open("combined_room_object_list.json", "w") as f:
                json.dump(
                    {
                        "success": success,
                        "scenes": scenes,
                        "error_skipped_files": sorted(skipped_files),
                        "error_not_approved_rooms": {
                            rm: list(scenes)
                            for rm, scenes in sorted(not_approved_rooms.items())
                        },
                        "error_non_contiguous_rooms": non_contiguous_rooms,
                    },
                    f,
                    indent=4,
                )


if __name__ == "__main__":
    main()
