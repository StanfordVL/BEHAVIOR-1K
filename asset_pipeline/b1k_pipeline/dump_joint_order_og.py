"""Dump OmniGibson's joint order for the articulated models that task instances use.

Task instance files store each object's `joint_pos` as an unnamed list, in the
order of OmniGibson's `obj.joints` (the order `get_joint_positions` uses). This
script loads every model whose instances have two or more joint values into an
empty OmniGibson scene, records `list(obj.joints)`, checks the names against the
model's URDF, and writes metadata/mujoco_joint_order.json, which
export_tasks_mujoco.py uses to map `joint_pos` onto named MuJoCo joints.

It needs OmniGibson and its dataset:

    OMNIGIBSON_HEADLESS=1 python -m b1k_pipeline.dump_joint_order_og \\
        --instances_zip 2026-challenge-task-instances.zip
"""

import argparse
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET

import b1k_pipeline.utils

OUTPUT = b1k_pipeline.utils.PIPELINE_ROOT / "metadata" / "mujoco_joint_order.json"
DEFAULT_INSTANCES_ZIP = b1k_pipeline.utils.PIPELINE_ROOT / "2026-challenge-task-instances.zip"
OBJECTS_ZIP = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "parallels" / "objects.zip"
TEMPLATE_PATTERN = re.compile(r"^scenes/[^/]+/json/.+(_template|_stable)\.json$")


def articulated_models(instances_zip):
    """(category, model) -> number of joint values, for models with >= 2."""
    models = {}
    with zipfile.ZipFile(instances_zip) as zf:
        for member in zf.namelist():
            if not TEMPLATE_PATTERN.match(member):
                continue
            state = json.loads(zf.read(member))
            init_info = state["objects_info"]["init_info"]
            registry = state["state"]["registry"]["object_registry"]
            for name, info in init_info.items():
                args = info["args"]
                if "category" not in args or "model" not in args:
                    continue
                joint_pos = registry.get(name, {}).get("joint_pos") or []
                if len(joint_pos) >= 2:
                    key = (args["category"], args["model"])
                    previous = models.setdefault(key, len(joint_pos))
                    assert previous == len(joint_pos), f"{key}: inconsistent joint counts"
    return models


def urdf_joint_names(category, model):
    """Movable joint names of a model's URDF in objects.zip, or None if unavailable."""
    if not OBJECTS_ZIP.exists():
        return None
    with zipfile.ZipFile(OBJECTS_ZIP) as zf:
        member = f"objects/{category}/{model}/urdf/{model}.urdf"
        if member not in zf.namelist():
            return None
        root = ET.parse(io.BytesIO(zf.read(member))).getroot()
    return {j.attrib["name"] for j in root.findall("joint") if j.attrib.get("type") != "fixed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instances_zip", default=str(DEFAULT_INSTANCES_ZIP))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    models = articulated_models(args.instances_zip)
    print(f"{len(models)} articulated models with >= 2 joint values")

    import omnigibson as og
    from omnigibson.objects.dataset_object import DatasetObject

    env = og.Environment(configs={"scene": {"type": "Scene", "floor_plane_visible": False}})
    order, problems = {}, {}
    for i, ((category, model), count) in enumerate(sorted(models.items())):
        og.sim.stop()
        obj = DatasetObject(name=f"obj_{i}", category=category, model=model, fixed_base=True)
        env.scene.add_object(obj)
        og.sim.play()
        names = list(obj.joints.keys())
        key = f"{category}/{model}"
        order[key] = names
        if len(names) != count:
            problems.setdefault(key, []).append(f"OmniGibson has {len(names)} joints, instances have {count}")
        urdf_names = urdf_joint_names(category, model)
        if urdf_names is not None and set(names) != urdf_names:
            problems.setdefault(key, []).append(f"joint names differ from the URDF: {sorted(set(names) ^ urdf_names)}")
        og.sim.stop()
        env.scene.remove_object(obj)

    with open(args.output, "w") as f:
        json.dump({"format": "b1k_joint_order/1", "models": order, "problems": problems}, f, indent=1, sort_keys=True)
    print(f"Wrote {len(order)} models to {args.output}; {len(problems)} with problems")
    og.shutdown()


if __name__ == "__main__":
    main()
