import argparse
import json
import os

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def get_scene_support_objs(scene_name):
    support_objs_json = "data/ig_dataset/scenes/%s/misc/all_support_objs.json" % scene_name
    if os.path.isfile(support_objs_json):
        with open(support_objs_json, "r") as f:
            support_obj_dicts = json.load(f)
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
    return support_obj_dicts


def main(args):
    scene_name = args.scene_name
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
    if scene_name not in scene_names:
        print("%s is not a valid scene name" % scene_name)
        return

    support_obj_dicts = get_scene_support_objs(scene_name)

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

    print("-" * 80)
    print("Rooms:")
    for room in unique_rooms:
        print(room)
    print("-" * 80)

    print("Room furniture:")
    for room in unique_rooms:
        print("%s:" % room)
        for category in room_categories[room]:
            print("\t%s" % category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure which surfaces and containers in a scene an object might go in."
    )
    parser.add_argument("scene_name", type=str)
    args = parser.parse_args()
    main(args)
