import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import b1k_pipeline.utils

import json
import pathlib
import pymxs
import textwrap

rt = pymxs.runtime


def should_get_complaint(complaint):
    if complaint["processed"]:
        return False

    if complaint["type"] in ("material", "clothappearance"):
        return False

    return True


def main():
    current_max_dir = pathlib.Path(rt.maxFilePath).resolve()
    complaint_path = current_max_dir / "complaints.json"

    if not complaint_path.exists():
        print("No complaints!")
        return

    selected_objs = list(rt.selection) if len(rt.selection) > 0 else list(rt.objects)
    selected_names = [obj.name for obj in selected_objs]
    selected_obj_matches = [
        b1k_pipeline.utils.parse_name(name) for name in selected_names
    ]
    selected_keys = {
        match.group("model_id") for match in selected_obj_matches if match is not None
    }

    with open(complaint_path, "r") as f:
        all_complaints = json.load(f)

    complaints = [
        complaint
        for complaint in all_complaints
        if complaint["object"].split("-")[-1] in selected_keys
        and should_get_complaint(complaint)
    ]
    if len(complaints) == 0:
        print("No complaints for objects", ", ".join(selected_keys))
    else:
        for complaint in complaints:
            print("\n".join(textwrap.wrap(str(complaint))))
            print("")


if __name__ == "__main__":
    main()
