import json
import pathlib
from deepdiff import DeepDiff

from b1k_pipeline.utils import parse_name


def model_ids_from_objects(objs):
    model_ids = set()
    for obj in objs:
        pn = parse_name(obj)
        if not pn:
            continue
        model_ids.add(pn.group("model_id"))

    return model_ids


def get_meta_type(obj):
    """
    Get the meta type of an object.
    :param obj: The object to get the meta type from.
    :return: The meta type of the object.
    """
    pn = parse_name(obj)
    assert pn, f"Could not parse object name {obj}"
    if pn.group("meta_type"):
        return pn.group("meta_type")
    elif pn.group("light_id"):
        return "light"
    else:
        return "None"


def main():
    deep_glob = "cad/*/*/artifacts/file_manifest.json"  # maybe use _deep ?
    base_files = pathlib.Path("base").glob(deep_glob)
    base_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in base_files}
    pr_files = pathlib.Path("pr").glob(deep_glob)
    pr_manifest_by_target = {"/".join(f.parts[-4:-2]): f for f in pr_files}
    all_targets = set(base_manifest_by_target.keys()) | set(
        pr_manifest_by_target.keys()
    )

    full_diffs = {}
    object_diffs = {}
    for target in sorted(all_targets):
        # Load the manifests
        base_manifest = (
            json.loads(base_manifest_by_target[target].read_text())
            if target in base_manifest_by_target
            else []
        )
        pr_manifest = (
            json.loads(pr_manifest_by_target[target].read_text())
            if target in pr_manifest_by_target
            else []
        )

        # Convert the manifests to dicts based on the object name
        base_manifest_dict = {v["name"]: v for v in base_manifest}
        pr_manifest_dict = {v["name"]: v for v in pr_manifest}

        # Discard the mtl_hash for now. Should not matter until we start material work.
        for val in base_manifest_dict.values():
            del val["mtl_hash"]
        for val in pr_manifest_dict.values():
            del val["mtl_hash"]

        diff = DeepDiff(base_manifest_dict, pr_manifest_dict)
        object_diffs[target] = sorted(diff.affected_root_keys)
        full_diffs[target] = diff.pretty()

    meta_type_diffs = sorted(
        {
            get_meta_type(x)
            for target_object_diffs in object_diffs.values()
            for x in target_object_diffs
            if parse_name(x)
        }
    )

    print("-------------------------------------------------")
    print("OBJECT DIFFS")
    print("Unique edited obj names:", sum(len(vals) for vals in object_diffs.values()))
    print(
        "Unique edited model IDs:",
        len(
            {
                mid
                for objs in object_diffs.values()
                for mid in model_ids_from_objects(objs)
            }
        ),
    )
    print("Unique edited meta types:", len(meta_type_diffs), meta_type_diffs)
    print("-------------------------------------------------")
    print("All edited models:")
    for mid in sorted(
        {mid for objs in object_diffs.values() for mid in model_ids_from_objects(objs)}
    ):
        print("    " + str(mid))
    print()
    print("BY TARGET:")
    for target in sorted(all_targets):
        target_objs = sorted(object_diffs[target])
        target_meta_types = sorted(
            {get_meta_type(x) for x in target_objs if parse_name(x)}
        )
        target_mids = sorted(model_ids_from_objects(target_objs))

        print(f"\n\n-------------------------\n{target}")
        print("Unique edited obj names:", len(target_objs))
        print("Unique edited model IDs:", len(target_mids))
        print("Unique edited meta types:", len(target_meta_types), target_meta_types)

        print("Objects:")
        for root in target_objs:
            print("    " + str(root))

        print("Models:")
        for mid in target_mids:
            print("    " + str(mid))

        print()

    print("-------------------------------------------------")
    print("FULL DIFFS")
    print("-------------------------------------------------")
    for target in sorted(all_targets):
        print(f"\n\n-------------------------\n{target}")
        print("    " + full_diffs[target].replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
