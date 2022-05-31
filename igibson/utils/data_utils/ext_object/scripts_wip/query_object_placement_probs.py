import argparse
import json
import os

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def input_number_or_name(selection_name, options):
    if len(options) == 1:
        print("Only %s option is %s" % (selection_name, options[0]))
        return options[0]

    for i, option in enumerate(options):
        print("%s option %d: %s" % (selection_name, i, option))

    user_input = input("Input number or name to assign probability, or done to finish\n")
    if user_input == "done":
        return "done"

    try:
        obj_num = int(user_input)
        choice = options[obj_num]
    except:
        if user_input.lower() not in [option.lower() for option in options]:
            print("Input %s is not valid, try again" % user_input)
            return False
        choice = [option for option in options if option.lower() == user_input.lower()][0]

    return choice


def main(args):
    object_cat_dir = "data/ig_dataset/objects/%s" % (args.object_cat)

    if not os.path.isdir(object_cat_dir):
        print("%s is not a valid object" % (args.object_cat))
        return

    if args.object_id:
        object_ids = [args.object_id]
    else:
        object_ids = os.listdir(object_cat_dir)

    obj_json_paths = []
    existing_placement_rules = {}
    obj_probs = {}
    total_prob = 0.0
    for object_id in object_ids:
        obj_dir = "%s/%s/misc/" % (object_cat_dir, object_id)
        if not os.path.isdir(obj_dir):
            print("%s %s is not a valid object" % (args.object_cat, object_id))
            return
        obj_json_path = "%s/placement_probs.json" % (obj_dir)
        if os.path.isfile(obj_json_path) and not args.overwrite:
            if args.add:
                if total_prob == 0.0:
                    with open(obj_json_path, "r") as f:
                        obj_probs = json.load(f)
                    total_prob = 1.0
            elif not args.overwrite:
                print("%s exists and overwrite false, quitting" % obj_json_path)
                return
        obj_json_paths.append(obj_json_path)

    scene_names = [
        "Beechwood_1_int",
        "Benevolence_1_int",
        "Ihlen_0_int",
        "Merom_0_int",
        "Pomaria_0_int",
        "Pomaria_2_int",
        "Wainscott_0_int",
        "Beechwood_0_int",
        "Benevolence_0_int",
        "Benevolence_2_int",
        "Ihlen_1_int",
        "Merom_1_int",
        "Pomaria_1_int",
        "Rs_int",
        "Wainscott_1_int",
    ]
    support_obj_dicts = []
    for scene_name in scene_names:
        support_objs_json = "data/ig_dataset/scenes/%s/misc/all_support_objs.json" % scene_name
        if os.path.isfile(support_objs_json):
            with open(support_objs_json, "r") as f:
                support_obj_dicts += json.load(f)
        else:
            settings = MeshRendererSettings(enable_shadow=False, msaa=False, enable_pbr=False)
            s = Simulator(mode="headless", image_width=800, image_height=800, rendering_settings=settings)
            simulator = s
            scene = InteractiveIndoorScene(scene_name, texture_randomization=False, object_randomization=False)
            s.import_scene(scene)

            for obj_name in scene.objects_by_name:
                obj = scene.objects_by_name[obj_name]
                if not obj.supporting_surfaces:
                    continue
                info_dict = {}
                info_dict["name"] = obj_name
                info_dict["category"] = obj.category
                info_dict["room"] = obj.in_rooms[0]
                info_dict["supporting_surface_types"] = list(obj.supporting_surfaces.keys())
                support_obj_dicts.append(info_dict)

            with open(support_objs_json, "w") as f:
                json.dump(support_obj_dicts, f)

            s.disconnect()

    unique_categories = set()
    unique_rooms = set()
    room_category_support_types = {}
    for support_obj_dict in support_obj_dicts:
        obj_category = support_obj_dict["category"]
        unique_categories.add(obj_category)
        obj_room = support_obj_dict["room"][:-2]
        unique_rooms.add(obj_room)
        room_category_support_types[(obj_category, obj_room)] = support_obj_dict["supporting_surface_types"]

    unique_categories = list(unique_categories)
    unique_rooms = list(unique_rooms)
    room_categories = {room: set() for room in unique_rooms}
    for support_obj_dict in support_obj_dicts:
        obj_category = support_obj_dict["category"]
        obj_room = support_obj_dict["room"][:-2]
        room_categories[obj_room].add(obj_category)

    for room in room_categories:
        room_categories[room] = list(room_categories[room])

    done = False
    while not done:
        room = input_number_or_name("room", unique_rooms)
        while not room:
            room = input_number_or_name("room", unique_rooms)
        if room == "done":
            break

        categories = room_categories[room]
        obj_category = input_number_or_name("object category", categories)
        while not obj_category:
            obj_category = input_number_or_name("object category", categories)
        if obj_category == "done":
            break

        support_types = room_category_support_types[(obj_category, room)]
        support_type = input_number_or_name("support_type", support_types)
        while not support_type:
            support_type = input_number_or_name("support_type", support_types)
        if support_type == "done":
            break

        prob = float(
            input(
                "Enter probability for object %s being %s %s in room %s\n"
                % (args.object_cat, support_type, obj_category, room)
            )
        )
        obj_probs["%s-%s-%s" % (obj_category, room, support_type)] = prob
        total_prob += prob

        for key in obj_probs:
            obj_probs[key] /= total_prob

    for obj_json_path in obj_json_paths:
        with open(obj_json_path, "w") as f:
            json.dump(obj_probs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure which surfaces and containers in a scene an object might go in."
    )
    parser.add_argument("object_cat", type=str, default=None)
    parser.add_argument("--object_id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--add", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
