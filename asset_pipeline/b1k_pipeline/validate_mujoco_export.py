"""Validate a MuJoCo export of the BEHAVIOR-1K dataset.

Runs over the packed layout (objects/, scenes/, skybox.png), given as a
directory or a zip, and checks every base scene:

* each part and each composed scene compiles;
* every object has its root body `N_root`, loose objects have a free joint
  `N_freejoint` and fixed ones have none, and every `N_b1k_metadata` text
  parses;
* no `<attach>` carries a `scale` attribute (scales live in frames);
* contacts at qpos0: the deepest penetration, flagged below -5 mm;
* stability: stepping from qpos0 without a robot, loose objects that move
  more than 5 cm are flagged;
* per scene: compile time, nmesh, nmeshvert, ntex and the bytes of tex_data.

With --check_objects it also compiles every referenced object archive under an
anisotropic frame scale, which is how scenes scale them.

The script only needs MuJoCo and NumPy.

Example:
    python validate_mujoco_export.py --root og_dataset_mujoco --scenes house_single_floor \\
        --report validation.json
"""

import argparse
import glob
import json
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

METADATA_TEXT = "b1k_metadata"
METADATA_FORMAT = "b1k_metadata/1"
PENETRATION_FLAG = -0.005
DRIFT_FLAG = 0.05
TEST_SCALE = "1.2 0.8 1.1"


def _text(model, name):
    text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if text_id < 0:
        return None
    adr, size = model.text_adr[text_id], model.text_size[text_id]
    return bytes(model.text_data[adr : adr + size - 1]).decode()


def _compile(path):
    start = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(path)
    return model, time.perf_counter() - start


def model_stats(model, compile_seconds):
    return {
        "compile_seconds": round(compile_seconds, 3),
        "nq": int(model.nq),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "nmeshvert": int(model.nmeshvert),
        "ntex": int(model.ntex),
        "tex_bytes": int(model.tex_data.nbytes),
    }


def check_xml_files(paths):
    """Raw-XML checks: no scale attribute on attach."""
    problems = []
    for path in paths:
        for attach in ET.parse(path).getroot().iter("attach"):
            if "scale" in attach.attrib:
                problems.append(f"{path}: attach {attach.attrib.get('prefix')} has a scale attribute")
    return problems


def check_objects_in_model(model, objects):
    """Root bodies, free joints and metadata texts of the manifest's objects."""
    problems = []
    for name, info in objects.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_root")
        if body_id < 0:
            problems.append(f"{name}: no {name}_root body")
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
        if info["fixed"]:
            if model.body_jntnum[body_id] != 0:
                problems.append(f"{name}: fixed object has a joint on its root")
        elif joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            problems.append(f"{name}: loose object has no free joint {name}_freejoint")
        elif model.jnt_bodyid[joint_id] != body_id:
            problems.append(f"{name}: {name}_freejoint isn't on {name}_root")
        text = _text(model, f"{name}_{METADATA_TEXT}")
        if text is None:
            problems.append(f"{name}: no {name}_{METADATA_TEXT} text")
            continue
        try:
            metadata = json.loads(text)
        except json.JSONDecodeError as e:
            problems.append(f"{name}: {METADATA_TEXT} doesn't parse: {e}")
            continue
        if metadata.get("format") != METADATA_FORMAT:
            problems.append(f"{name}: unexpected metadata format {metadata.get('format')}")
        if metadata.get("model") != info["model"]:
            problems.append(f"{name}: metadata model {metadata.get('model')} != {info['model']}")
    return problems


def contact_report(model, data, limit=10):
    """Deepest contacts at the current state."""
    mujoco.mj_forward(model, data)
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        contacts.append((float(c.dist), b1, b2))
    contacts.sort()
    deepest = contacts[0][0] if contacts else 0.0
    flagged = [c for c in contacts if c[0] < PENETRATION_FLAG]
    return {
        "ncon": int(data.ncon),
        "deepest": deepest,
        "num_flagged": len(flagged),
        "flagged": [{"dist": d, "body1": b1, "body2": b2} for d, b1, b2 in flagged[:limit]],
    }


def stability_report(model, objects, steps, timestep, key=None):
    """Displacement of loose objects after stepping without a robot."""
    data = mujoco.MjData(model)
    if key is not None:
        mujoco.mj_resetDataKeyframe(model, data, key)
    model.opt.timestep = timestep
    loose = {}
    for name, info in objects.items():
        if info["fixed"]:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
        if joint_id >= 0:
            loose[name] = model.jnt_qposadr[joint_id]
    start = {n: data.qpos[a : a + 3].copy() for n, a in loose.items()}
    unstable = False
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            unstable = True
            break
    moved = []
    for name, adr in loose.items():
        d = float(np.linalg.norm(data.qpos[adr : adr + 3] - start[name]))
        if not np.isfinite(d) or d > DRIFT_FLAG:
            moved.append({"object": name, "displacement": d})
    moved.sort(key=lambda m: -m["displacement"] if np.isfinite(m["displacement"]) else -np.inf)
    return {"steps": steps, "timestep": timestep, "nan": unstable, "num_moved": len(moved), "moved": moved}


def check_object_archive(root, archive):
    """Compile an object archive, under an anisotropic frame scale unless its
    metadata says it can't be stretched (scenes then only scale it uniformly)."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(os.path.join(root, archive)) as zf:
            zf.extractall(tmp)
        metadata_text = None
        for text in ET.parse(os.path.join(tmp, "model.xml")).getroot().iter("text"):
            if text.attrib.get("name") == METADATA_TEXT:
                metadata_text = text.attrib["data"]
        scalable = json.loads(metadata_text).get("anisotropic_scaling", {}).get("ok", True) if metadata_text else True
        scale = TEST_SCALE if scalable else "1.1"
        wrapper = os.path.join(tmp, "_scaled.xml")
        with open(wrapper, "w") as f:
            f.write(
                '<mujoco><asset><model name="o" file="model.xml"/></asset><worldbody>'
                f'<frame scale="{scale}" scaling="geometry" massscale="1">'
                '<attach model="o" body="base_link" prefix="o_"/></frame></worldbody></mujoco>'
            )
        model = mujoco.MjModel.from_xml_path(wrapper)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(data.geom_xpos)):
            raise ValueError("non-finite geom positions")


def validate_scene(root, manifest_path, args):
    scene_dir = os.path.dirname(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    report = {"scene": manifest["scene"], "parts": {}, "scenes": {}, "problems": []}
    xml_files = [os.path.join(scene_dir, part["file"]) for part in manifest["parts"].values()]
    xml_files += [os.path.join(scene_dir, f) for f in manifest["scenes"].values()]
    report["problems"] += check_xml_files(xml_files)

    for part_name, part in manifest["parts"].items():
        try:
            model, seconds = _compile(os.path.join(scene_dir, part["file"]))
            report["parts"][part_name] = model_stats(model, seconds)
            part_objects = {n: manifest["objects"][n] for n in part["objects"] + part["inlined"]}
            report["problems"] += [f"[{part_name}] {p}" for p in check_objects_in_model(model, part_objects)]
        except Exception as e:  # pylint: disable=broad-except
            report["problems"].append(f"[{part_name}] doesn't compile: {e}")

    for kind, file in manifest["scenes"].items():
        entry = {}
        try:
            model, seconds = _compile(os.path.join(scene_dir, file))
        except Exception as e:  # pylint: disable=broad-except
            report["problems"].append(f"[{kind}] doesn't compile: {e}")
            continue
        entry.update(model_stats(model, seconds))
        objects = manifest["objects"]
        report["problems"] += [f"[{kind}] {p}" for p in check_objects_in_model(model, objects)]
        data = mujoco.MjData(model)
        entry["contacts"] = contact_report(model, data)
        if args.stability_steps:
            entry["stability"] = stability_report(model, objects, args.stability_steps, args.timestep)
        report["scenes"][kind] = entry

    if args.check_objects:
        archives = sorted({o["archive"] for o in manifest["objects"].values() if o["archive"]})
        failures = {}
        for archive in archives:
            try:
                check_object_archive(root, archive)
            except Exception as e:  # pylint: disable=broad-except
                failures[archive] = str(e)
        report["object_archives"] = {"checked": len(archives), "failures": failures}
        report["problems"] += [f"[object] {a}: {e}" for a, e in failures.items()]
    return report


def _key_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)


def check_task_xml_order(path):
    """Robot frames come after every body and attachment in the worldbody."""
    worldbody = ET.parse(path).getroot().find("worldbody")
    seen_frame = False
    for child in worldbody:
        if child.tag == "frame" and child.attrib.get("name", "").startswith("robot_"):
            seen_frame = True
        elif seen_frame and child.tag in ("body", "attach"):
            return [f"{path}: <{child.tag}> after robot frames"]
    return []


def validate_task(root, sidecar_path, args, instances_zf):
    """Check a task sidecar's scenes: compile, layout, keyframes, frames, states."""
    scene_dir = os.path.dirname(sidecar_path)
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    report = {"activity": sidecar["activity"], "scene": sidecar["scene"], "modes": {}, "problems": []}
    instances = sidecar["instances"]
    loose_scope = {
        inst: e["object"]
        for inst, e in sidecar["scope"].items()
        if e["kind"] == "object" and sidecar["objects"][e["object"]]["freejoint"]
    }
    for mode_name, mode in sidecar["modes"].items():
        path = os.path.join(scene_dir, mode["xml"])
        entry = {}
        report["problems"] += check_task_xml_order(path)
        try:
            model, seconds = _compile(path)
        except Exception as e:  # pylint: disable=broad-except
            report["problems"].append(f"[{mode_name}] doesn't compile: {e}")
            continue
        entry.update(model_stats(model, seconds))
        if model.nq != mode["nq"]:
            report["problems"].append(f"[{mode_name}] nq {model.nq} != sidecar {mode['nq']}")
        for name, jtype, adr, size in mode["qpos_layout"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or model.jnt_qposadr[jid] != adr:
                report["problems"].append(f"[{mode_name}] qpos_layout: {name} at {adr}, model has {jid}")
                break
        labels = ["template"] + [i["keyframe"] for i in instances]
        xml_frames = {
            f.attrib.get("name") for f in ET.parse(path).getroot().find("worldbody").findall("frame")
        }
        for label in labels:
            if _key_id(model, label) < 0:
                report["problems"].append(f"[{mode_name}] no keyframe {label}")
            if f"robot_{label}" not in xml_frames:
                report["problems"].append(f"[{mode_name}] no robot frame robot_{label}")

        # Keyframes reproduce the instance states of loose scope objects.
        data = mujoco.MjData(model)
        if instances_zf is not None:
            worst = 0.0
            for inst in instances:
                state = json.loads(instances_zf.read(inst["source"]))
                mujoco.mj_resetDataKeyframe(model, data, _key_id(model, inst["keyframe"]))
                for scope_name, obj in loose_scope.items():
                    if scope_name not in state:
                        continue
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_freejoint")
                    adr = model.jnt_qposadr[jid]
                    pos = np.asarray(state[scope_name]["root_link"]["pos"])
                    x, y, z, w = state[scope_name]["root_link"]["ori"]
                    quat = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
                    err = max(
                        float(np.max(np.abs(data.qpos[adr : adr + 3] - pos))),
                        float(min(np.max(np.abs(data.qpos[adr + 3 : adr + 7] - quat)),
                                  np.max(np.abs(data.qpos[adr + 3 : adr + 7] + quat)))),
                    )
                    worst = max(worst, err)
            entry["keyframe_max_error"] = worst
            if worst > 1e-6:
                report["problems"].append(f"[{mode_name}] keyframes differ from instance states by {worst:.2e}")

        # Contacts and stability from the template and a few instances.
        mujoco.mj_resetDataKeyframe(model, data, _key_id(model, "template"))
        entry["contacts"] = contact_report(model, data)
        objects = {n: {"fixed": o["freejoint"] is None} for n, o in sidecar["objects"].items()}
        if args.stability_steps:
            stab = {}
            for label in labels[: 1 + args.task_stability_samples]:
                stab[label] = stability_report(model, objects, args.stability_steps, args.timestep, _key_id(model, label))
            entry["stability"] = stab
        report["modes"][mode_name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="Packed dataset directory or zip.")
    parser.add_argument("--scenes", nargs="*", help="Only these scenes.")
    parser.add_argument("--stability_steps", type=int, default=100)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--check_objects", action="store_true")
    parser.add_argument("--tasks", action="store_true", help="Also validate task scenes and their sidecars.")
    parser.add_argument("--activities", nargs="*", help="Only these activities.")
    parser.add_argument("--instances_zip", help="Task-instance archive, to check keyframes against instance states.")
    parser.add_argument("--task_stability_samples", type=int, default=2)
    parser.add_argument("--report", help="Write the JSON report here.")
    args = parser.parse_args()

    root, tmp = args.root, None
    if zipfile.is_zipfile(root):
        tmp = tempfile.mkdtemp(prefix="b1k_mujoco_validate_")
        with zipfile.ZipFile(root) as zf:
            zf.extractall(tmp)
        root = tmp
    try:
        manifests = sorted(glob.glob(os.path.join(root, "scenes", "*", "mjcf", "*_parts.json")))
        if args.scenes:
            manifests = [m for m in manifests if os.path.basename(m)[: -len("_parts.json")] in args.scenes]
        reports = []
        for manifest in manifests:
            report = validate_scene(root, manifest, args)
            reports.append(report)
            print(f"== {report['scene']}: {len(report['problems'])} problems")
            for kind, entry in report["scenes"].items():
                stab = entry.get("stability", {})
                print(
                    f"  {kind}: compile {entry['compile_seconds']}s nq={entry['nq']} nmesh={entry['nmesh']} "
                    f"nmeshvert={entry['nmeshvert']} ntex={entry['ntex']} tex={entry['tex_bytes'] / 1e9:.2f}GB "
                    f"deepest={entry['contacts']['deepest'] * 1000:.1f}mm "
                    f"penetrating={entry['contacts']['num_flagged']} moved={stab.get('num_moved')} "
                    f"nan={stab.get('nan')}"
                )
            for problem in report["problems"][:20]:
                print("  !", problem)
        if args.tasks:
            instances_zf = zipfile.ZipFile(args.instances_zip) if args.instances_zip else None
            sidecars = sorted(glob.glob(os.path.join(root, "scenes", "*", "mjcf", "*_task_*.json")))
            for sidecar in sidecars:
                scene = os.path.basename(os.path.dirname(os.path.dirname(sidecar)))
                activity = json.load(open(sidecar))["activity"]
                if (args.scenes and scene not in args.scenes) or (args.activities and activity not in args.activities):
                    continue
                report = validate_task(root, sidecar, args, instances_zf)
                reports.append(report)
                print(f"== task {report['activity']} ({report['scene']}): {len(report['problems'])} problems")
                for mode, entry in report["modes"].items():
                    moved = {k: v["num_moved"] for k, v in entry.get("stability", {}).items()}
                    print(
                        f"  {mode}: compile {entry['compile_seconds']}s nq={entry['nq']} "
                        f"tex={entry['tex_bytes'] / 1e9:.2f}GB deepest={entry['contacts']['deepest'] * 1000:.1f}mm "
                        f"penetrating={entry['contacts']['num_flagged']} key_err={entry.get('keyframe_max_error')} "
                        f"moved={moved}"
                    )
                for problem in report["problems"][:20]:
                    print("  !", problem)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(reports, f, indent=1)
        if any(r["problems"] for r in reports):
            raise SystemExit(1)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
