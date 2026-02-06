import json
import pathlib

def main():
    scenes = []
    # with open("/checkpoint/clear/cgokmen/procthor/houses/houses_2023_07_28/train.jsonl", "r") as f:
    #     for i, line in enumerate(f):
    #         if i != 125565:
    #             continue
    #         scenes.append(json.loads(line))
    #         break

    scene_data = json.loads(pathlib.Path("/checkpoint/clear/cgokmen/behavior-data2/spoc/scenes/train_125565/json/train_125565_best.json").read_text())
    floor_keys = {k for k, v in scene_data["objects_info"]["init_info"].items() if v["args"]["category"] == "floors"}
    for k in floor_keys:
        in_rooms = scene_data["objects_info"]["init_info"][k]["args"]["in_rooms"]
        print(k, in_rooms)
    floor_positions = {k: scene_data["state"]["registry"]["object_registry"][k]["root_link"]["pos"] for k in floor_keys}

    breakpoint()

    for scene in scenes:
        for room in scene["rooms"]:
            for floor in room["floors"]:
                print(floor)

if __name__ == "__main__":
    main()