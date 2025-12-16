from collections import defaultdict
import glob
import pathlib

import sys

import numpy as np
import tqdm

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import parse_name

# Items to merge
MERGES_BY_TARGET = {
    # "objects/batch-04": [
    #     "oedkgo",
    #     "wrtgbv",
    #     "pfapuv",
    #     "xzizzj",
    #     "zseasm",
    #     "onpcoz",
    #     "rlcnca",
    #     "diwmzn",
    #     "voawxd",
    #     "dwnzxq",
    #     "lderti",
    #     "hhtouu",
    #     "zrerox",
    #     "qjkibp",
    #     "knqyja",
    # ],
    # "objects/batch-07": [
    #     "goprkj",
    #     "nkzvwy",
    #     "tgekwx",
    #     "ffmpvs",
    #     "mqxqgs",
    #     "detbvf",
    #     "vjruxb",
    #     "dwxvxp",
    #     "ptcdjs",
    #     "mkvtje",
    #     "izavdu",
    #     "sspnhq",
    #     "pxsxoj",
    #     "ppwcrm",
    #     "wlktgt",
    #     "wzbjba",
    #     "bdsaeg",
    #     "vnvhzd",
    #     "iobqqp",
    #     "crltjn",
    #     "mbvbqt",
    #     "lfkjfd",
    #     "qlrgsg",
    #     "ywnfer",
    #     "uenxiu",
    #     "kjjkii",
    #     "nhljib",
    #     "vskces",
    #     "gkmvqk",
    #     "xvvsij",
    #     "jjxhmf",
    #     "axloyp",
    #     "kkckpq",
    #     "puxhag",
    #     "nizghj",
    #     "zdcjqz",
    #     "qbfkpo",
    #     "xkpcoi",
    #     "yujqba",
    #     "nvtnsi",
    #     "qctxuz",
    #     "qyglnm",
    #     "wodmnd",
    #     "lceujn",
    #     "vmnhqj",
    #     "wywdvp",
    # ],
    # "objects/batch-01": ["kqhokv", "ozvqdx"],
    # "objects/batch-08": [
    #     "etejri",
    #     "hgswfr",
    #     "bajppk",
    #     "ncvtfs",
    #     "jwqrer",
    #     "pqlafi",
    #     "huevxh",
    #     "bmnejm",
    # ],
    # "objects/legacy_batch-10": [
    #     "fjafda",
    #     "fjdkah",
    #     "fkasda",
    #     "iwoaoo",
    #     "rtuiww",
    #     "lwuaas",
    # ],
    # "objects/task_relevant-xy": ["prxfhj", "gqmvyh", "fayhga", "hppsgv"],
    # "objects/batch-02": [
    #     "hpnnry",
    #     "uyqrik",
    #     "fdmgww",
    #     "kiwkye",
    #     "wfhrzf",
    #     "zwkoax",
    #     "vgfyvo",
    #     "sasyin",
    #     "wvqiai",
    #     "dcosbg",
    # ],
    # "objects/batch-06": ["gltrdw", "wocjtj", "fwupto", "wufirc", "fyvbms"],
    # "objects/batch-00": [
    #     "bceugv",
    #     "cdirwq",
    #     "wsitbh",
    #     "absogx",
    #     "svozzf",
    #     "jvkhxe",
    #     "oqfzwb",
    #     "akpsld",
    #     "eillnh",
    #     "gikvpd",
    #     "ucrnic",
    #     "ukbqku",
    #     "unfdjb",
    #     "ypyicv",
    #     "ujyxaz",
    #     "tmsvpt",
    #     "dougsx",
    #     "hzlhrs",
    #     "xkewvu",
    #     "wyswqr",
    #     "duigmh",
    #     "fkhgbp",
    #     "gapzju",
    #     "rravgd",
    #     "lprbcf",
    #     "tweqol",
    #     "ahhnsg",
    #     "nldcwc",
    #     "tjbgmz",
    # ],
    # "objects/batch-03": ["cfmerb", "yrjoai", "numpzy", "glalke", "hkujlj", "abcedi"],
    # "objects/batch-05": [
    #     "scegcm",
    #     "evzqom",
    #     "tjrevb",
    #     "rxnxmg",
    #     "ohhozd",
    #     "irgjqx",
    #     "wbgews",
    #     "jbfabl",
    #     "kdwqco",
    #     "deftjq",
    #     "uwdwzw",
    #     "clvjuy",
    # ],
    "objects/batch-09": ["acjxoq"],
    "objects/legacy_batch-05": ["acooui"],
    "objects/legacy_batch-04": ["mfkwzi"],
    "objects/batch-11": ["rxefdu"],
    "objects/batch-12": ["avvshq"],
    "objects/legacy_batch-11": ["bhirar"],
}


def merge_files():
    x_so_far = 19000

    # Merge in each file
    for target, ids in tqdm.tqdm(MERGES_BY_TARGET.items()):
        p = (
            pathlib.Path(r"D:\BEHAVIOR-1K\asset_pipeline\cad")
            / target
            / "processed.max"
        )
        filename = str(p)

        # Get the right object names from the file
        f_objects = rt.getMAXFileObjectNames(filename, quiet=True)
        visual_objects_to_merge = defaultdict(set)
        for obj in f_objects:
            m = parse_name(obj)
            if not m:
                continue
            if m.group("bad") or m.group("instance_id") != "0":
                continue

            visual_objects_to_merge[m.group("model_id")].add(obj)

        model_ids_to_import = set(visual_objects_to_merge.keys()) & set(ids)
        print(
            f"Importing {len(model_ids_to_import)} models from {filename}: {sorted(model_ids_to_import)}"
        )

        # Clean up all the objects in the current file belonging to that model ID
        deletion_queue = []
        for obj in rt.objects:
            m = parse_name(obj.name)
            if not m:
                continue
            if m.group("model_id") in model_ids_to_import:
                deletion_queue.append(obj)
        assert not deletion_queue, f"Objects to delete: {deletion_queue}"
        # for obj in deletion_queue:
        #     rt.delete(obj)

        # Concatenate the list of all the objects that we need to import
        objects_to_import = [
            object_name
            for model_id in model_ids_to_import
            for object_name in visual_objects_to_merge[model_id]
        ]

        success, imported_meshes = rt.mergeMaxFile(
            filename,
            objects_to_import,
            rt.Name("select"),
            rt.Name("autoRenameDups"),
            rt.Name("useSceneMtlDups"),
            rt.Name("neverReparent"),
            rt.Name("noRedraw"),
            quiet=True,
            mergedNodes=pymxs.byref(None),
        )
        assert success, f"Failed to import"
        imported_objs_by_name = {obj.name: obj for obj in imported_meshes}
        assert set(objects_to_import) == set(imported_objs_by_name.keys()), (
            "Not all objects were imported. Missing: "
            + str(set(objects_to_import) - set(imported_objs_by_name.keys()))
        )

        # Unhide everything
        for x in imported_meshes:
            x.isHidden = False

        # Move the root-level objects to the right place
        for x in imported_meshes:
            if x.parent is not None:
                continue

            bbox_min, bbox_max = rt.NodeGetBoundingBox(x, rt.Matrix3(1))
            bbox_min = np.array(bbox_min)
            bbox_max = np.array(bbox_max)
            bbox_extent = bbox_max - bbox_min
            bbox_min_wrt_pos = np.array(x.position) - bbox_min
            target_bbox_min = np.array([x_so_far, 4000, 0])
            x.position = rt.Point3(*(target_bbox_min + bbox_min_wrt_pos).tolist())
            x_so_far += bbox_extent[0] + 100

    print("Done!")


if __name__ == "__main__":
    merge_files()
