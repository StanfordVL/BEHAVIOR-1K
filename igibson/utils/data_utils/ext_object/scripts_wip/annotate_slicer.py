import json
import os


from bddl.object_taxonomy import ObjectTaxonomy
from IPython import embed


def set_pos_orn(body_id, pos, orn):
    p.resetBasePositionAndOrientation(body_id, pos, orn)


def get_pos_orn(body_id):
    return p.getBasePositionAndOrientation(body_id)


def save_box(obj_inst_dir, box_id, sizes):
    pos, orn = get_pos_orn(box_id)

    metadata_file = os.path.join(obj_inst_dir, "misc", "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)
    if "links" not in metadata:
        metadata["links"] = {}

    key = "slicer"
    metadata["links"][key] = {}
    metadata["links"][key]["geometry"] = "box"
    metadata["links"][key]["size"] = sizes
    metadata["links"][key]["xyz"] = pos
    metadata["links"][key]["rpy"] = p.getEulerFromQuaternion(orn)

    with open(metadata_file, "w+") as f:
        json.dump(metadata, f)


slicer_synsets = []
taxonomy = ObjectTaxonomy()
for node in taxonomy.taxonomy.nodes:
    if taxonomy.is_leaf(node) and taxonomy.has_ability(node, "slicer") and node != "lawn_mower.n.01":
        slicer_synsets.append(node)

p.connect(p.GUI)
obj_root_dir = "/cvgl2/u/chengshu/gibsonv2/igibson/data/ig_dataset/objects"

for synset in slicer_synsets:
    for obj_cat in taxonomy.get_subtree_igibson_categories(synset):
        obj_dir = os.path.join(obj_root_dir, obj_cat)
        for obj_inst in sorted(os.listdir(obj_dir)):
            obj_inst_dir = os.path.join(obj_dir, obj_inst)
            obj_urdf = os.path.join(obj_inst_dir, obj_inst + ".urdf")
            path_to_urdf = obj_urdf.replace(".urdf", "_slicer.urdf")
            if os.path.isfile(path_to_urdf):
                print("already annotated")
                continue
            obj_id = p.loadURDF(obj_urdf)
            embed()
            p.removeBody(obj_id)

            # # Example of adding a box and setting its pose
            # sizes = np.array([0.03, 0.002, 0.00035])
            # visual_id = p.createVisualShape(
            #     p.GEOM_BOX, halfExtents=sizes / 2)
            # collision_id = p.createVisualShape(
            #     p.GEOM_BOX, halfExtents=sizes / 2)
            # sizes = sizes.tolist()
            # box_id = p.createMultiBody(
            #     baseVisualShapeIndex=visual_id, baseCollisionShapeIndex=collision_id)
            # set_pos_orn(box_id, pos, orn)

            # # After the adjustment is done
            # save_box(obj_inst_dir, box_id, sizes)
            # p.removeBody(box_id)
