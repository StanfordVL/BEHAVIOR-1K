from collections import defaultdict
import glob
import pathlib

import sys

import tqdm

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import parse_name

# Items to merge
IDS_TO_MERGE = {
    "aknswb",
    "arbekb",
    "ayuubt",
    "bftrar",
    "bgpvtb",
    "bpwjxr",
    "cgvoho",
    "ebtgyg",
    "ehwmol",
    "enfumt",
    "ertfpr",
    "ewqpfv",
    "exaqtm",
    "fdxevw",
    "fjytro",
    "fkosow",
    "flctdz",
    "ftvzzg",
    "gemgfz",
    "gvkgsi",
    "gxajos",
    "hfianz",
    "hlktkb",
    "hocerb",
    "hvjwvj",
    "hwhisw",
    "hxbrop",
    "hylkmj",
    "inkwmw",
    "izhouj",
    "jchomn",
    "jwagmm",
    "khxpgc",
    "kjxpdi",
    "kkrlrc",
    "ldbalq",
    "ldrjoi",
    "lufqkq",
    "mbfgxr",
    "mcukuh",
    "mgzyjy",
    "mstxjw",
    "muqeud",
    "myaxie",
    "nbwrns",
    "neqiyu",
    "nwlbit",
    "oedwla",
    "oynrtw",
    "paaegg",
    "pktarr",
    "pmtynn",
    "qebtel",
    "qemval",
    "qvvvxz",
    "qzrpiw",
    "rgmujm",
    "rpprcc",
    "rwmofs",
    "sdsfzw",
    "shnohg",
    "skcdiz",
    "tbzucx",
    "tfgypq",
    "tfrisv",
    "tiscsa",
    "tjawkk",
    "tmluxa",
    "tqyiso",
    "uztisk",
    "vaoegv",
    "vgsgxk",
    "wraizn",
    "wttqrk",
    "wxkuww",
    "xijidt",
    "xjupjc",
    "xkfmjp",
    "xnjodq",
    "xxipyh",
    "ydfgju",
    "yhthgr",
    "zoolef",
    "zrsytt",
    "zughoy",
    "zywgbh",
}


def merge_files():
    max_files = glob.glob(r"D:\BEHAVIOR-1K\asset_pipeline\cad\scenes\*\processed.max")
    school_files = sorted(
        [pathlib.Path(x) for x in max_files if "scenes" in x and "school_" in x]
    )

    # Merge in each file
    for p in tqdm.tqdm(school_files):
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

        model_ids_to_import = set(visual_objects_to_merge.keys()) & IDS_TO_MERGE
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
        for obj in deletion_queue:
            rt.delete(obj)

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

    print("Done!")


if __name__ == "__main__":
    merge_files()
