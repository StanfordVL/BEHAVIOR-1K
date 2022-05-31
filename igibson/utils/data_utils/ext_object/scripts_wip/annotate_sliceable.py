import json
import os


from IPython import embed


def set_pos_orn(body_id, pos, orn):
    p.resetBasePositionAndOrientation(body_id, pos, orn)


def get_pos_orn(body_id):
    return p.getBasePositionAndOrientation(body_id)


p.connect(p.GUI)
obj_root_dir = "/cvgl2/u/chengshu/gibsonv2/igibson/data/ig_dataset/objects"

prematched_category = ["apple", "vidalia_onion"]
for obj_cat in sorted(os.listdir(obj_root_dir)):
    if "half_" not in obj_cat:
        continue

    whole_obj_cat = obj_cat.replace("half_", "")
    whole_obj_dir = os.path.join(obj_root_dir, whole_obj_cat)
    half_obj_dir = os.path.join(obj_root_dir, obj_cat)
    for obj_inst in sorted(os.listdir(whole_obj_dir)):
        obj_inst_dir = os.path.join(whole_obj_dir, obj_inst)
        json_file = os.path.join(obj_inst_dir, "misc", "metadata.json")
        with open(json_file) as f:
            metadata = json.load(f)
        if whole_obj_cat in prematched_category:
            assert "object_parts" in metadata
            object_parts = metadata["object_parts"]
        else:
            half_names = os.listdir(half_obj_dir)
            num_halfs = len(half_names)
            assert num_halfs == 1 or num_halfs == 2
            if num_halfs == 1:
                object_parts = [
                    {"category": obj_cat, "model": half_names[0]},
                    {"category": obj_cat, "model": half_names[0]},
                ]
            else:
                object_parts = [
                    {"category": obj_cat, "model": half_names[0]},
                    {"category": obj_cat, "model": half_names[1]},
                ]

        whole_obj_urdf = os.path.join(obj_inst_dir, obj_inst + ".urdf")
        obj_part_ids = []
        already_annotated = False
        for part in object_parts:
            if "pos" in part:
                already_annotated = True
                break
            obj_part_urdf = os.path.join(obj_root_dir, part["category"], part["model"], part["model"] + ".urdf")
            obj_part_ids.append(p.loadURDF(obj_part_urdf))

        if already_annotated:
            print("already annotated:", obj_inst_dir)
            continue

        whole_obj_id = p.loadURDF(whole_obj_urdf)

        embed()
        for obj_part_id, part in zip(obj_part_ids, object_parts):
            pos, orn = get_pos_orn(obj_part_id)
            part["pos"] = pos
            part["orn"] = orn

        for body_id in obj_part_ids + [whole_obj_id]:
            p.removeBody(body_id)

        metadata["object_parts"] = object_parts
        with open(json_file, "w+") as f:
            json.dump(metadata, f, indent=4)
