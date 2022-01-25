import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from igibson.object_states.utils import sample_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs


def main(args):
    scene_names = [
        "Beechwood_0_int",
        "Beechwood_1_int",
        "Benevolence_1_int",
        "Ihlen_0_int",
        "Merom_0_int",
        "Pomaria_0_int",
        "Pomaria_2_int",
        "Wainscott_0_int",
        "Benevolence_0_int",
        "Benevolence_2_int",
        "Ihlen_1_int",
        "Merom_1_int",
        "Pomaria_1_int",
        "Rs_int",
        "Wainscott_1_int",
    ]
    if args.scene_name not in scene_names and args.scene_name != "all":
        print("%s is not a valid scene name" % args.scene_name)
        return

    objects_to_sample = []
    object_id_dict = {}
    object_cat_dirs = {}
    with open(args.csv_name, "r") as f:
        for line in f:
            parts = line.split(",")
            cat = parts[0]
            count = int(parts[1])

            object_cat_dir = "data/ig_dataset/objects/%s" % (cat)
            if not os.path.isdir(object_cat_dir):
                print("%s is not a valid object" % (cat))
                return

            object_cat_dirs[cat] = object_cat_dir
            objects_to_sample.append((cat, count))

            object_ids = os.listdir(object_cat_dir)
            object_id_dict[cat] = object_ids

    settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=True)
    if args.scene_name == "all":
        all_scene_names = scene_names
    else:
        all_scene_names = [args.scene_name]
    for scene_name in all_scene_names:
        s = Simulator(mode="headless", image_width=800, image_height=800, rendering_settings=settings)
        support_categories = [
            "dining_table",
            "desk",
            "pedestal_table",
            "gaming_table",
            "stand",
            "console_table",
            "coffee_table",
            "fridge",
            "countertop",
            "top_cabinet",
            "bookshelf",
            "bottom_cabinet",
            "bottom_cabinet_no_top",
            "coffee_table",
            "carpet",
        ]
        simulator = s
        scene = InteractiveIndoorScene(
            scene_name,
            texture_randomization=False,
            object_randomization=False,
            load_object_categories=support_categories,
        )
        s.import_scene(scene)
        renderer = s.renderer

        category_supporting_objects = {}
        obj_counts = {}
        placements_counts = {}
        room_placements_counts = {}
        for obj_name in scene.objects_by_name:
            obj = scene.objects_by_name[obj_name]
            if not obj.supporting_surfaces:
                continue
            obj_counts[obj] = 0
            cat = obj.category
            if "table" in cat or "stand" in cat:
                cat = "table"
            if "shelf" in cat:
                cat = "shelf"
            if "counter" in cat:
                cat = "counter"
            room = obj.in_rooms[0][:-2]
            if (cat, room) not in category_supporting_objects:
                category_supporting_objects[(cat, room)] = []
            category_supporting_objects[(cat, room)].append(obj)
            if room not in room_placements_counts:
                room_placements_counts[room] = 0
            if (cat, room) not in placements_counts:
                placements_counts[(cat, room)] = 0

        placement_count = 0
        avg_category_spec = get_ig_avg_category_specs()
        random.shuffle(objects_to_sample)
        for category, count in objects_to_sample:
            valid_placement_rules = []
            for i in range(count):
                object_id = random.choice(object_id_dict[category])
                if len(object_id_dict[category]) > 1:
                    object_id_dict[category].remove(object_id)
                urdf_path = "%s/%s/%s.urdf" % (object_cat_dirs[category], object_id, object_id)
                while not os.path.isfile(urdf_path):
                    object_id = random.choice(object_id_dict[category])
                    if len(object_id_dict[category]) > 1:
                        object_id_dict[category].remove(object_id)
                    else:
                        break
                    urdf_path = "%s/%s/%s.urdf" % (object_cat_dirs[category], object_id, object_id)
                if not os.path.isfile(urdf_path):
                    break

                name = "%s|%s|%d" % (category, object_id, i)
                urdf_object = URDFObject(
                    urdf_path,
                    name=name,
                    category=category,
                    overwrite_inertial=True,
                    avg_obj_dims=avg_category_spec.get(category),
                    fit_avg_dim_volume=True,
                )
                simulator.import_object(urdf_object)
                for attempt in range(args.num_attempts):
                    urdf_path = "%s/%s/%s.urdf" % (object_cat_dirs[category], object_id, object_id)
                    placement_rules_path = os.path.join(urdf_object.model_path, "misc", "placement_probs.json")
                    if not os.path.isfile(placement_rules_path):
                        break
                    with open(placement_rules_path, "r") as f:
                        placement_rules = json.load(f)
                    valid_placement_rules = {}
                    for placement_rule in placement_rules.keys():
                        support_obj_cat, room, predicate = placement_rule.split("-")
                        if (support_obj_cat, room) in category_supporting_objects:
                            valid_placement_rules[placement_rule] = placement_rules[placement_rule]
                    if len(valid_placement_rules) == 0:
                        print("No valid rules for %s" % category)
                        print(placement_rules)
                        break
                    placement_rule = random.choices(
                        list(valid_placement_rules.keys()), weights=valid_placement_rules.values(), k=1
                    )[0]
                    support_obj_cat, room, predicate = placement_rule.split("-")
                    if predicate == "ontop":
                        predicate = "onTop"
                    support_objs = category_supporting_objects[(support_obj_cat, room)]
                    min_obj = None
                    min_obj_count = None
                    for obj in support_objs:
                        if min_obj is None or obj_counts[obj] < min_obj_count:
                            min_obj = obj
                            min_obj_count = obj_counts[obj]
                    if attempt < 2:
                        chosen_support_obj = min_obj
                    else:
                        chosen_support_obj = random.choice(support_objs)
                    print("Sampling %s %s %s %s in %s" % (category, object_id, predicate, support_obj_cat, room))
                    result = sample_kinematics(predicate, urdf_object, chosen_support_obj, True)
                    if not result:
                        print("Failed kinematic sampling! Attempt %d" % attempt)
                        continue
                    for i in range(10):
                        s.step()
                    obj_counts[chosen_support_obj] += 1
                    placement_count += 1
                    room_placements_counts[room] += 1
                    placements_counts[(support_obj_cat, room)] += 1

                    if args.save_images:
                        simulator.sync()
                        scene.open_one_obj(chosen_support_obj.get_body_ids()[chosen_support_obj.main_body], "max")
                        pos = urdf_object.get_position()
                        offsets = [[-0.6, 0], [0.0, -0.6], [0.6, 0.0], [0.0, 0.6]]
                        for i in range(4):
                            camera_pos = np.array([pos[0] - offsets[i][0], pos[1] - offsets[i][1], pos[2] + 0.1])
                            renderer.set_camera(camera_pos, pos, [0, 0, 1])
                            frame = renderer.render(modes=("rgb"))[0]
                            plt.imshow(frame)
                            plt.savefig("placement_imgs/%s_placement_%d_%d.png" % (scene_name, placement_count, i))
                            plt.close()
                        scene.open_one_obj(chosen_support_obj.get_body_ids()[chosen_support_obj.main_body], "zero")

                    urdf_object.in_rooms = chosen_support_obj.in_rooms
                    break
                if len(valid_placement_rules) == 0:
                    break

        print("Total %d objects placed" % placement_count)
        if args.urdf_name:
            scene.save("%s_%s" % (scene_name, args.urdf_name))
        if args.save_placement_txt:
            with open("%s_placements.txt" % scene_name, "w") as f:
                for room in room_placements_counts:
                    f.write("%s: %d\n" % (room, room_placements_counts[room]))
                    for support_cat in [cat for cat, r in placements_counts if r == room]:
                        f.write("\t%s: %d\n" % (support_cat, placements_counts[(support_cat, room)]))

        s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure which surfaces and containers in a scene an object might go in."
    )
    parser.add_argument("scene_name", type=str)
    parser.add_argument("csv_name", type=str)
    parser.add_argument("--urdf_name", type=str)
    parser.add_argument("--num_attempts", type=int, default=10)
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--save_placement_txt", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
