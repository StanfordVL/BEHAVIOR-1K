"""Export every BEHAVIOR challenge activity as MuJoCo task scenes plus a JSON sidecar.

For each activity of the task-instance archive (a template per activity, 300
train instances and 20 public test instances), this writes two task scenes
next to the scene's base-scene parts (export_scenes_mujoco.py):

* `{scene}_task_{activity}_partial.xml`: the rooms the official evaluator loads
  (the BDDL's `inroom` room types, one level of ROOM_DEPENDENCIES, and
  TASK_SPECIFIC_EXTRA_ROOMS), i.e. the `shell` part plus every part whose rooms
  intersect them;
* `{scene}_task_{activity}_full.xml`: every part.

Each attaches its parts with prefix `{part}__`, adds the objects that the task
template adds to the settled scene (as `N_root`, with frame scales), then the
robot frames, then keyframes: `template`, `train_0007`, `test_0301`, ... Each
key holds the scene's full qpos. Task semantics live in the sidecar
`{scene}_task_{activity}.json` (see docs/mujoco_export.md), never in the XML.

    python -m b1k_pipeline.export_tasks_mujoco [--activities turning_on_radio ...]
"""

import argparse
import ast
import collections
import hashlib
import io
import json
import os
import re
import subprocess
import traceback
import xml.etree.ElementTree as ET
import zipfile
from concurrent import futures
from xml.dom import minidom

import numpy as np
import tqdm
import yaml
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils
from b1k_pipeline import mujoco_common
from b1k_pipeline import mujoco_scene_utils

ROOT = b1k_pipeline.utils.PIPELINE_ROOT
PARALLELS = ROOT / "artifacts" / "parallels"
DEFAULT_INSTANCES_ZIP = ROOT / "2026-challenge-task-instances.zip"
DEFAULT_INSTRUCTIONS = ROOT / "2026-challenge-tasks.jsonl"
JOINT_ORDER_FILE = ROOT / "metadata" / "mujoco_joint_order.json"
ROBOT_FRAME_OFFSETS_FILE = ROOT / "metadata" / "mujoco_robot_frame_offsets.json"
OG_TELEOP_CFG = ROOT.parent / "joylo" / "gello" / "utils" / "og_teleop_cfg.py"

SIDECAR_FORMAT = "b1k_mujoco_task/1"
EVAL_TIMEOUT_MULTIPLIER = 1.5  # omnigibson/eval/utils/eval_utils.py
CONTROL_HZ = 30.0
TEMPLATE_PATTERN = re.compile(r"^scenes/(?P<scene>[^/]+)/json/(?P=scene)_task_(?P<activity>.+)_0_0_template\.json$")
INSTANCE_PATTERN = re.compile(r"_0_(?P<id>\d+)_template-tro_state\.json$")
ROOM_INSTANCE_PATTERN = re.compile(r"^(?P<type>.+)_(?P<index>\d+)$")
FIXED_POS_TOLERANCE = 1e-3
FIXED_ANGLE_TOLERANCE_DEG = 0.1
QPOS_SIZES = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _fmt(values):
    return mujoco_common.fmt(values)


def _quat_wxyz(xyzw):
    q = mujoco_common.xyzw_to_wxyz(xyzw)
    return q / np.linalg.norm(q)


def load_room_tables(path=OG_TELEOP_CFG):
    """ROOM_DEPENDENCIES and TASK_SPECIFIC_EXTRA_ROOMS, read without importing
    the module (it imports OmniGibson and torch)."""
    tables = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) in ("ROOM_DEPENDENCIES", "TASK_SPECIFIC_EXTRA_ROOMS"):
                    tables[target.id] = ast.literal_eval(node.value)
    return tables["ROOM_DEPENDENCIES"], tables["TASK_SPECIFIC_EXTRA_ROOMS"]


def eval_room_types(init_conditions, scene, activity, room_dependencies, extra_rooms):
    """Room types the official evaluator loads (og_teleop_utils.augment_rooms)."""
    types = sorted({c[2] for c in init_conditions if len(c) == 3 and c[0] == "inroom"})
    augmented = list(types)
    for room in types:
        for dependent in room_dependencies.get(scene, {}).get(room, []):
            if dependent not in augmented:
                augmented.append(dependent)
    augmented += extra_rooms.get(activity, {}).get(scene, [])
    return sorted(set(augmented))


def room_type(instance):
    match = ROOM_INSTANCE_PATTERN.match(instance)
    return match["type"] if match else instance


class KnowledgeBase:
    """bddl's knowledge base, as OmniGibson's BehaviorTask uses it."""

    def __init__(self):
        from bddl.knowledge_base import KnowledgeBase as BddlKnowledgeBase

        self.kb = BddlKnowledgeBase(verbose=False)

    def is_substance(self, synset):
        synset_obj = self.kb.get_synset(synset)
        return synset_obj is not None and synset_obj.is_substance

    def categories(self, synset):
        """OmniGibson categories of a synset (bddl_utils.og_categories_from_bddl_inst)."""
        synset_obj = self.kb.get_synset(synset)
        synsets = [synset_obj] + sorted(synset_obj.descendants, key=lambda s: s.name)
        return [c.name for s in synsets if s.is_leaf for c in s.categories]

    def compile(self, activity, inst_to_name, init_info, loaded_order):
        """Compile the activity like BehaviorTask with cached scope, expanding wildcards.

        Wildcards expand to the counts of each category in the room instances
        of the base scope's inroom objects (_build_scene_layout_from_rooms),
        and expanded instances not in inst_to_name go to the first unassigned
        loaded object of a matching category (_assign_wildcard_instances).

        Returns:
            (compiled task, binding: instance -> object name, wildcard info)
        """
        task = self.kb.get_task(f"{activity}-0")
        _, _, inroom = task.parse_base_scope()
        binding = dict(inst_to_name)
        wildcard = None
        if task.has_wildcards:
            room_instances = {}
            for inst, rtype in inroom.items():
                name = inst_to_name.get(inst)
                if name not in init_info:
                    continue
                for room in init_info[name]["args"].get("in_rooms") or []:
                    if room.rsplit("_", 1)[0] == rtype:
                        room_instances.setdefault(rtype, room)
                        break
            layout = {
                rtype: dict(
                    collections.Counter(
                        init_info[n]["args"]["category"]
                        for n in loaded_order
                        if room in (init_info[n]["args"].get("in_rooms") or [])
                    )
                )
                for rtype, room in room_instances.items()
            }
            compiled = task.compile(scene_layout=layout)
            expanded = {}
            assigned = set(binding.values())
            for inst in sorted(compiled.object_scope):
                if inst in binding or inst.startswith("agent.n."):
                    continue
                synset = "_".join(inst.split("_")[:-1])
                if self.is_substance(synset):
                    continue
                categories = set(self.categories(synset))
                for name in loaded_order:
                    if init_info[name]["args"]["category"] in categories and name not in assigned:
                        binding[inst] = expanded[inst] = name
                        assigned.add(name)
                        break
            wildcard = {"room_instances": room_instances, "scene_layout": layout, "expanded": expanded}
        else:
            compiled = task.compile(scene_layout={})
        return compiled, binding, wildcard


def classify_scope(conditions, binding, init_info, kb):
    """Scope entries as object / agent / system / future."""
    future = {c[1] for c in conditions.parsed_initial_conditions if len(c) == 2 and c[0] == "future"}
    scope = {}
    for synset, instances in conditions.parsed_objects.items():
        for inst in instances:
            name = binding.get(inst)
            if synset.startswith("agent.n."):
                scope[inst] = {"kind": "agent"}
            elif inst in future:
                scope[inst] = {"kind": "future", "synset": synset}
            elif kb.is_substance(synset):
                scope[inst] = {"kind": "system", "system": name, "synset": synset}
            else:
                if name not in init_info:
                    raise ValueError(f"Scope object {inst} ({name}) is not bound to a template object")
                scope[inst] = {"kind": "object", "object": name, "synset": synset}
    return scope


def _archive_joints(objects_mujoco_zf, archive, cache):
    """Movable joints of an object archive's base_link tree, in qpos order."""
    if archive not in cache:
        with zipfile.ZipFile(io.BytesIO(objects_mujoco_zf.read(archive))) as zf:
            root = ET.fromstring(zf.read(mujoco_common.ARCHIVE_ROOT_FILE))
        cache[archive] = [(j.attrib["name"], j.attrib.get("type", "hinge")) for j in _iter_tree_joints(root)]
    return cache[archive]


def _iter_tree_joints(root):
    """Joints of an MJCF in compiled (depth-first, document) order."""
    worldbody = root.find("worldbody")

    def visit(el):
        for child in el:
            if child.tag in ("joint", "freejoint"):
                yield child
            elif child.tag in ("body", "frame"):
                yield from visit(child)

    yield from visit(worldbody)


def qpos_layout(xml_root, resolve):
    """[(joint name, type, qpos address, size)] of a scene, without compiling it.

    MuJoCo orders joints depth-first in document order; attachments insert the
    child's subtree at the <attach>, with the prefix prepended to names.

    Args:
        xml_root: The scene's <mujoco> element.
        resolve: Function (model asset name) -> ("part", <mujoco> element) or
            ("object", [(joint name, type)]).
    """
    layout = []

    def add(name, jtype):
        size = QPOS_SIZES[jtype]
        adr = layout[-1][2] + layout[-1][3] if layout else 0
        layout.append((name, jtype, adr, size))

    def visit(el, prefix):
        for child in el:
            if child.tag == "freejoint":
                add(prefix + child.attrib["name"], "free")
            elif child.tag == "joint":
                add(prefix + child.attrib["name"], child.attrib.get("type", "hinge"))
            elif child.tag in ("body", "frame"):
                visit(child, prefix)
            elif child.tag == "attach":
                kind, payload = resolve(child.attrib["model"])
                child_prefix = prefix + child.attrib["prefix"]
                if kind == "part":
                    visit(payload.find("worldbody"), child_prefix)
                else:
                    for name, jtype in payload:
                        add(child_prefix + name, jtype)

    visit(xml_root.find("worldbody"), "")
    return layout


def robot_pose(poses, key):
    """The first pose under `robot`, else under the robot model (case-insensitive)."""
    by_lower = {k.lower(): k for k in poses}
    for candidate in ("robot", key.lower()):
        if candidate in by_lower and poses[by_lower[candidate]]:
            return by_lower[candidate], poses[by_lower[candidate]][0]
    raise KeyError(f"No robot pose under 'robot' or '{key}' in {sorted(poses)}")


def yaw_frame(pose):
    """Position, yaw-only quaternion (wxyz) and tilt in degrees of a robot pose."""
    rotation = R.from_quat(pose["orientation"])
    z = rotation.apply([0, 0, 1])
    tilt = float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))
    x = rotation.apply([1, 0, 0])
    yaw = float(np.arctan2(x[1], x[0]))
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return np.asarray(pose["position"], dtype=np.float64), np.asarray(quat), tilt


def shift_frame(pos, quat_wxyz, shift):
    """A yaw-only frame shifted by [dx, dy, dyaw_deg] in its own coordinates.

    x is the frame's forward and y its left; the frame turns about its own z.
    """
    dx, dy, dyaw_deg = shift
    yaw = 2.0 * np.arctan2(quat_wxyz[3], quat_wxyz[0])
    new_pos = np.asarray(pos, dtype=np.float64) + [
        np.cos(yaw) * dx - np.sin(yaw) * dy,
        np.sin(yaw) * dx + np.cos(yaw) * dy,
        0.0,
    ]
    new_yaw = yaw + np.radians(dyaw_deg)
    return new_pos, np.array([np.cos(new_yaw / 2), 0.0, 0.0, np.sin(new_yaw / 2)])


def load_robot_frame_offsets(path=ROBOT_FRAME_OFFSETS_FILE):
    """activity -> {scene, frames: keyframe label -> [dx, dy, dyaw_deg]}.

    See metadata/mujoco_robot_frame_offsets.json and docs/mujoco_export.md.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        activities = json.load(f)["activities"]
    for activity, entry in activities.items():
        for label, shift in entry["frames"].items():
            if len(shift) != 3:
                raise ValueError(f"{activity} {label}: a frame shift is [dx, dy, dyaw_deg], got {shift}")
    return activities


def _pose_delta(a, b):
    dpos = float(np.linalg.norm(np.asarray(a["pos"]) - np.asarray(b["pos"])))
    rel = R.from_quat(a["ori"]).inv() * R.from_quat(b["ori"])
    return dpos, float(np.degrees(rel.magnitude()))


class ActivityExporter:
    def __init__(self, args, scene_manifests, joint_order):
        self.args = args
        self.scene_manifests = scene_manifests
        self.joint_order = joint_order
        self.room_dependencies, self.extra_rooms = load_room_tables()
        self.kb = KnowledgeBase()
        self.archive_joint_cache = {}
        self.urdf_joint_cache = {}
        self.robot_frame_offsets = load_robot_frame_offsets(args.robot_frame_offsets)

    def og_joint_names(self, objects_zf, category, model):
        """OmniGibson's joint order for a model, and whether it's known."""
        key = f"{category}/{model}"
        if key in self.joint_order:
            return self.joint_order[key], True
        if key not in self.urdf_joint_cache:
            urdf = ET.fromstring(objects_zf.read(f"objects/{category}/{model}/urdf/{model}.urdf"))
            self.urdf_joint_cache[key] = [
                j.attrib["name"] for j in urdf.findall("joint") if j.attrib["type"] in ("revolute", "continuous", "prismatic")
            ]
        return self.urdf_joint_cache[key], False

    def export(self, scene, activity, instances_zf, objects_mujoco_zf, objects_zf, scenes_mujoco_zf, out_root):
        issues, warnings = [], []
        template_member = f"scenes/{scene}/json/{scene}_task_{activity}_0_0_template.json"
        template = json.loads(instances_zf.read(template_member))
        manifest = self.scene_manifests[scene]
        init_info = {
            n: i for n, i in template["objects_info"]["init_info"].items() if "category" in i["args"] and "model" in i["args"]
        }
        registry = template["state"]["registry"]["object_registry"]
        inst_to_name = template["metadata"]["task"]["inst_to_name"]

        import bddl

        problem = (
            pathlib_path(bddl.__file__).parent / "activity_definitions" / activity / "problem0.bddl"
        ).read_bytes()
        # The evaluator's room types come from the base (wildcard-free) scope
        # (og_teleop_utils.get_task_relevant_room_types).
        conditions = self.kb.kb.get_task(f"{activity}-0").parse_base_scope()[0]

        # Objects: the settled scene's (in parts) plus those the template adds.
        base = manifest["objects"]
        added = {}
        for name, info in init_info.items():
            args = info["args"]
            if name in base:
                ref = base[name]
                if (args["category"], args["model"]) != (ref["category"], ref["model"]):
                    raise ValueError(f"{name}: template model differs from the base scene")
                if args.get("scale") and np.max(np.abs(np.asarray(args["scale"]) - ref["scale"])) > 1e-4:
                    raise ValueError(f"{name}: template scale differs from the base scene")
                continue
            if args["category"] in mujoco_common.STRUCTURE_CATEGORIES or args.get("visual_only"):
                raise ValueError(f"{name}: task-added structure or visual-only objects aren't supported")
            added[name] = {
                "category": args["category"],
                "model": args["model"],
                "scale": np.asarray(args.get("scale") or [1, 1, 1], dtype=np.float64),
                "fixed": bool(args.get("fixed_base")),
                "visual_only": False,
                "rooms": list(args.get("in_rooms") or []),
                "archive": mujoco_common.object_archive_path(args["category"], args["model"]),
            }
        missing = set(base) - set(init_info)
        if missing:
            raise ValueError(f"Template lacks base-scene objects: {sorted(missing)[:5]}")

        # Rooms.
        all_parts = sorted(manifest["parts"])
        scene_rooms = sorted({r for p in all_parts for r in mujoco_scene_utils.part_rooms(p)})
        types = eval_room_types(
            conditions.parsed_initial_conditions, scene, activity, self.room_dependencies, self.extra_rooms
        )
        instances = [r for r in scene_rooms if room_type(r) in types]
        unknown_types = sorted(set(types) - {room_type(r) for r in scene_rooms})
        if unknown_types:
            warnings.append(f"room types not in the scene: {unknown_types}")
        modes = {
            "partial": {"parts": mujoco_scene_utils.parts_for_rooms(all_parts, instances), "rooms": instances},
            "full": {"parts": all_parts, "rooms": None},
        }
        for mode in modes.values():
            loaded = {n for p in mode["parts"] for n in manifest["parts"][p]["objects"] + manifest["parts"][p]["inlined"]}
            mode["added"] = sorted(
                n
                for n, o in added.items()
                if mode["rooms"] is None
                or o["category"] in mujoco_common.SHELL_CATEGORIES
                or set(o["rooms"]) & set(mode["rooms"])
            )
            mode["objects"] = loaded | set(mode["added"])
        # Compile the activity (expanding wildcards) over what the evaluator loads.
        loaded_order = [n for n in init_info if n in modes["partial"]["objects"]]
        compiled, binding, wildcard = self.kb.compile(activity, inst_to_name, init_info, loaded_order)
        conditions = compiled.conditions
        scope = classify_scope(conditions, binding, init_info, self.kb)
        for inst, entry in scope.items():
            if entry["kind"] == "object" and entry["object"] not in modes["partial"]["objects"]:
                raise ValueError(f"Scope object {inst} ({entry['object']}) isn't loaded with the eval rooms")

        # Instances.
        test_dir = f"scene_test/public/{scene}/json/{scene}_task_{activity}_instances/"
        train_dir = f"scenes/{scene}/json/{scene}_task_{activity}_instances/"
        instance_files = []
        names = instances_zf.namelist()
        for split, prefix in (("train", train_dir), ("test", test_dir)):
            if split not in self.args.splits:
                continue
            members = sorted(
                (int(INSTANCE_PATTERN.search(m)["id"]), m) for m in names if m.startswith(prefix) and INSTANCE_PATTERN.search(m)
            )
            if self.args.max_instances_per_split:
                members = members[: self.args.max_instances_per_split]
            instance_files += [(split, i, m) for i, m in members]
        test_template_member = f"scene_test/public/{scene}/json/{scene}_task_{activity}_0_0_template.json"

        # Object metadata for joints.
        def object_joints(name):
            record = base.get(name) or added[name]
            if record["archive"] is None:
                return None
            return _archive_joints(objects_mujoco_zf, record["archive"], self.archive_joint_cache)

        def fixed_joints(category, model):
            with zipfile.ZipFile(
                io.BytesIO(objects_mujoco_zf.read(mujoco_common.object_archive_path(category, model)))
            ) as zf:
                root = ET.fromstring(zf.read(mujoco_common.ARCHIVE_ROOT_FILE))
            text = root.find(f"custom/text[@name='{mujoco_common.METADATA_TEXT}']")
            md = json.loads(text.attrib["data"])
            return {n for n, f in md.get("joint_fixes", {}).items() if f["action"] == "fix"}

        fixed_cache = {}
        joint_order_assumed = set()
        fixed_with_values = set()

        def object_part_prefix(record):
            return mujoco_common.part_attach_prefix(record["part"]) if record.get("part") else ""

        def object_prefix(name, record):
            return object_part_prefix(record) + mujoco_common.attach_prefix(name)

        def object_root(name, record):
            return object_part_prefix(record) + mujoco_common.root_body_name(name)

        def object_freejoint(name, record):
            return object_part_prefix(record) + mujoco_common.freejoint_name(name)

        def joint_values(name, record, state, qpos_index, qpos):
            """Write an object's joint positions from an OmniGibson state."""
            values = state.get("joint_pos") or []
            if not values:
                return
            og_names, known = self.og_joint_names(objects_zf, record["category"], record["model"])
            if len(og_names) != len(values):
                raise ValueError(f"{name}: {len(values)} joint values for {len(og_names)} joints")
            nonzero = [abs(v) > 1e-9 for v in values]
            if len(values) >= 2 and any(nonzero) and not known:
                joint_order_assumed.add(f"{record['category']}/{record['model']}")
            key = (record["category"], record["model"])
            if key not in fixed_cache:
                fixed_cache[key] = fixed_joints(*key) if record["archive"] else set()
            for jn, v in zip(og_names, values):
                if jn in fixed_cache[key]:
                    if abs(v) > 1e-3:
                        fixed_with_values.add(f"{name}:{jn}")
                    continue
                full = object_prefix(name, record) + jn
                if full not in qpos_index:
                    raise ValueError(f"{name}: no joint {full} in the scene")
                qpos[qpos_index[full][0]] = v

        # Build each mode's XML and keyframes.
        parts_xml = {}

        def part_root(part):
            if part not in parts_xml:
                member = f"scenes/{scene}/mjcf/{manifest['parts'][part]['file']}"
                parts_xml[part] = ET.fromstring(scenes_mujoco_zf.read(member))
            return parts_xml[part]

        scene_dir = f"scenes/{scene}/mjcf"
        base_name = f"{scene}_task_{activity}"
        test_template = json.loads(instances_zf.read(test_template_member)) if test_template_member in names else None

        robot_frames = []
        template_poses = template["metadata"]["task"].get("robot_poses")
        if template_poses:
            tkey, tpose = robot_pose(template_poses, self.args.robot_pose_key)
        else:
            # Fall back to the challenge's start pose (metadata/available_tasks.yaml).
            start = self.args.available_tasks[activity][0]
            tkey = "available_tasks"
            tpose = {"position": start["robot_start_position"], "orientation": start["robot_start_orientation"]}
            warnings.append("the template has no robot poses; its frame uses available_tasks.yaml's start pose")
        robot_frames.append(("template", "robot_template", tkey, tpose))
        instance_states = []
        for split, inst_id, member in instance_files:
            state = json.loads(instances_zf.read(member))
            key, pose = robot_pose(state["robot_poses"], self.args.robot_pose_key)
            label = f"{split}_{inst_id:04d}"
            robot_frames.append((label, f"robot_{label}", key, pose))
            instance_states.append((split, inst_id, member, label, state))

        # Robot frames: BEHAVIOR's R1Pro base poses reduced to their yaw, then
        # shifted where metadata/mujoco_robot_frame_offsets.json says so that
        # Huba fits there.
        offsets = self.robot_frame_offsets.get(activity)
        shifts = {}
        if offsets is not None:
            if offsets["scene"] != scene:
                raise ValueError(f"robot frame offsets are for scene {offsets['scene']}, not {scene}")
            shifts = offsets["frames"]
            unknown = set(shifts) - {label for label, _, _, _ in robot_frames}
            if unknown:
                raise ValueError(f"robot frame offsets for unknown keyframes: {sorted(unknown)}")
        frame_poses = {}
        for label, _, _, pose in robot_frames:
            pos, quat, _ = yaw_frame(pose)
            if label in shifts:
                pos, quat = shift_frame(pos, quat, shifts[label])
            frame_poses[label] = (pos, quat, shifts.get(label))

        sidecar_modes = {}
        for mode_name, mode in modes.items():
            root = ET.Element("mujoco", {"model": f"{base_name}_{mode_name}"})
            g = mujoco_common.COLLISION_GROUP
            ET.SubElement(
                root, "compiler", {"angle": "radian", "autolimits": "true", "inertiagrouprange": f"{g} {g}"}
            )
            asset_xml = ET.SubElement(root, "asset")
            ET.SubElement(
                asset_xml,
                "texture",
                {
                    "name": "skybox",
                    "type": "skybox",
                    "file": mujoco_scene_utils.relative_path(scene_dir, "skybox.png"),
                    "gridsize": "3 4",
                    "gridlayout": ".U..LFRB.D..",
                },
            )
            worldbody_xml = ET.SubElement(root, "worldbody")
            resolve_table = {}
            for part in mode["parts"]:
                asset = f"part-{part}"
                ET.SubElement(asset_xml, "model", {"name": asset, "file": manifest["parts"][part]["file"]})
                ET.SubElement(
                    worldbody_xml,
                    "attach",
                    {"model": asset, "prefix": mujoco_common.part_attach_prefix(part)},
                )
                resolve_table[asset] = ("part", part_root(part))
            declared = set()
            for name in mode["added"]:
                record = added[name]
                mujoco_common.check_scalable(name, self._object_metadata(objects_mujoco_zf, record), record["scale"])
                asset = mujoco_common.model_asset_name(record["category"], record["model"])
                if asset not in declared:
                    ET.SubElement(
                        asset_xml, "model", {"name": asset, "file": mujoco_scene_utils.relative_path(scene_dir, record["archive"])}
                    )
                    declared.add(asset)
                state = registry[name]["root_link"]
                mujoco_scene_utils.add_object(
                    worldbody_xml, name, asset, state["pos"], _quat_wxyz(state["ori"]), record["fixed"], record["scale"]
                )
            for label, frame_name, _, _ in robot_frames:
                pos, quat, _ = frame_poses[label]
                mujoco_scene_utils.add_robot_frame(worldbody_xml, frame_name, pos, quat)

            # Resolve attach models for the qpos layout: parts, and the
            # object archives they and the task attach.
            def resolve(asset_name, _table=resolve_table):
                if asset_name in _table:
                    return _table[asset_name]
                category, model = asset_name.rsplit("-", 1)
                return (
                    "object",
                    _archive_joints(objects_mujoco_zf, mujoco_common.object_archive_path(category, model), self.archive_joint_cache),
                )

            layout = qpos_layout(root, resolve)
            nq = layout[-1][2] + layout[-1][3] if layout else 0
            qpos_index = {name: (adr, size, jtype) for name, jtype, adr, size in layout}

            # qpos0: loose roots at their authored pose. Parts author the
            # settled scene's poses and task objects the template's.
            def qpos_from(states):
                qpos = np.zeros(nq)
                for name in mode["objects"]:
                    record = base.get(name) or added[name]
                    state = states[name]
                    fj = object_freejoint(name, record)
                    if fj in qpos_index:
                        adr = qpos_index[fj][0]
                        qpos[adr : adr + 3] = state["root_link"]["pos"]
                        qpos[adr + 3 : adr + 7] = _quat_wxyz(state["root_link"]["ori"])
                    joint_values(name, record, state, qpos_index, qpos)
                return qpos

            template_states = {n: registry[n] for n in mode["objects"]}
            keys = [("template", qpos_from(template_states))]
            for split, inst_id, member, label, state in instance_states:
                states = dict(template_states)
                for inst, value in state.items():
                    entry = scope.get(inst)
                    if entry is None or entry["kind"] != "object":
                        continue
                    states[entry["object"]] = value
                keys.append((label, qpos_from(states)))
            for label, qpos in keys:
                mujoco_scene_utils.add_keyframe(root, label, qpos)

            out = os.path.join(out_root, scene_dir, f"{base_name}_{mode_name}.xml")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "))
            sidecar_modes[mode_name] = {
                "xml": f"{base_name}_{mode_name}.xml",
                "rooms": mode["rooms"],
                "parts": mode["parts"],
                "added_objects": mode["added"],
                "nq": nq,
                "qpos_layout": [list(entry) for entry in layout],
            }

        # Fixed objects can't move: flag instances that move them.
        for split, inst_id, member, label, state in instance_states:
            inst_issues = []
            for inst, value in state.items():
                entry = scope.get(inst)
                if entry is None or entry["kind"] != "object":
                    continue
                name = entry["object"]
                record = base.get(name) or added[name]
                if record["fixed"]:
                    dpos, dang = _pose_delta(registry[name]["root_link"], value["root_link"])
                    if dpos > FIXED_POS_TOLERANCE or dang > FIXED_ANGLE_TOLERANCE_DEG:
                        inst_issues.append(f"fixed {name} moved {dpos * 1000:.1f} mm / {dang:.2f} deg")
            state["_issues"] = inst_issues

        # Non-kinematic states.
        initial_nonkin = {
            inst: registry[e["object"]].get("non_kin") or {}
            for inst, e in scope.items()
            if e["kind"] == "object" and registry[e["object"]].get("non_kin")
        }
        varies = collections.defaultdict(set)
        sidecar_instances = []
        has_systems = False
        frames = {label: (key, pose) for label, _, key, pose in robot_frames}
        for split, inst_id, member, label, state in instance_states:
            nonkin, systems = {}, {}
            for inst, value in state.items():
                entry = scope.get(inst)
                if entry is None:
                    continue
                if entry["kind"] == "object":
                    diff = {
                        k: v for k, v in (value.get("non_kin") or {}).items() if initial_nonkin.get(inst, {}).get(k) != v
                    }
                    if diff:
                        nonkin[inst] = diff
                        varies[inst] |= set(diff)
                elif entry["kind"] == "system":
                    systems[inst] = value
                    has_systems = True
            key, pose = frames[label]
            _, _, tilt = yaw_frame(pose)
            frame_pos, frame_quat, frame_shift = frame_poses[label]
            sidecar_instances.append(
                {
                    "id": inst_id,
                    "split": split,
                    "keyframe": label,
                    "robot_frame": f"robot_{label}",
                    "source": member,
                    "robot_pose": {
                        "key": key,
                        "position": pose["position"],
                        "orientation_xyzw": pose["orientation"],
                        "tilt_deg": round(tilt, 3),
                    },
                    "robot_frame_pose": {
                        "position": [float(v) for v in mujoco_common.fmt(frame_pos).split()],
                        "quat_wxyz": [float(v) for v in mujoco_common.fmt(frame_quat).split()],
                        "shift": frame_shift,
                    },
                    "nonkin": nonkin,
                    "systems": systems,
                    "issues": state["_issues"],
                }
            )
        if has_systems:
            issues.append("systems (particles) are passed through raw; they aren't simulated")
        if fixed_with_values:
            issues.append(
                "joints made fixed (mujoco_joint_fixes.json) have nonzero values, which are dropped: "
                + ", ".join(sorted(fixed_with_values))
            )
        if joint_order_assumed:
            issues.append(
                "joint order assumed to follow the URDF (no mujoco_joint_order.json entry): "
                + ", ".join(sorted(joint_order_assumed))
            )

        lengths = self.args.human_lengths
        instructions = self.args.instructions
        human_length = lengths.get(activity)
        tkey_pose = robot_frames[0]
        sidecar = {
            "format": SIDECAR_FORMAT,
            "dataset_version": mujoco_common.DATASET_VERSION,
            "source": {
                "task_instances": os.path.basename(self.args.instances_zip),
                "sha256": self.args.instances_sha256,
                "exporter_commit": self.args.exporter_commit,
            },
            "activity": activity,
            "activity_definition": 0,
            "task_index": instructions.get(activity, {}).get("task_index"),
            "scene": scene,
            "modes": sidecar_modes,
            "default_mode": "partial",
            "rooms": {"types": types, "instances": instances},
            "instruction": instructions.get(activity, {}).get("task"),
            "human_length_steps": human_length,
            "eval_timeout_seconds": round(EVAL_TIMEOUT_MULTIPLIER * human_length / CONTROL_HZ, 3)
            if human_length
            else None,
            "bddl": {
                "problem_sha256": _sha256(problem),
                "wildcards": wildcard,
                "init": conditions.parsed_initial_conditions,
                "goal": conditions.parsed_goal_conditions,
            },
            "scope": scope,
            "objects": {
                name: {
                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in (base.get(name) or added[name]).items()},
                    "task_added": name in added,
                    "prefix": object_prefix(name, base.get(name) or added[name]),
                    "root": object_root(name, base.get(name) or added[name]),
                    "freejoint": None
                    if (base.get(name) or added[name])["fixed"]
                    else object_freejoint(name, base.get(name) or added[name]),
                    "joints": [
                        object_prefix(name, base.get(name) or added[name]) + j
                        for j, _ in (object_joints(name) or [])
                    ],
                }
                for name in sorted({e["object"] for e in scope.values() if e["kind"] == "object"} | set(added))
            },
            "template_robot_pose": {"key": tkey_pose[2], **tkey_pose[3]},
            "initial_nonkin": initial_nonkin,
            "nonkin_varies": {k: sorted(v) for k, v in sorted(varies.items())},
            "instances": sidecar_instances,
            "issues": issues,
            "warnings": warnings,
        }
        out = os.path.join(out_root, scene_dir, f"{base_name}.json")
        with open(out, "w") as f:
            json.dump(sidecar, f, indent=1)
        return issues + warnings

    def _object_metadata(self, objects_mujoco_zf, record):
        with zipfile.ZipFile(io.BytesIO(objects_mujoco_zf.read(record["archive"]))) as zf:
            root = ET.fromstring(zf.read(mujoco_common.ARCHIVE_ROOT_FILE))
        text = root.find(f"custom/text[@name='{mujoco_common.METADATA_TEXT}']")
        return json.loads(text.attrib["data"])


def pathlib_path(p):
    import pathlib

    return pathlib.Path(p)


_WORKER = {}


def _init_worker(args, scene_manifests, joint_order):
    _WORKER["exporter"] = ActivityExporter(args, scene_manifests, joint_order)
    _WORKER["instances"] = zipfile.ZipFile(args.instances_zip)
    _WORKER["objects_mujoco"] = zipfile.ZipFile(args.objects_mujoco_zip)
    _WORKER["objects"] = zipfile.ZipFile(PARALLELS / "objects.zip")
    _WORKER["scenes_mujoco"] = zipfile.ZipFile(args.scenes_mujoco_zip)


def _export_one(scene, activity, out_root):
    return _WORKER["exporter"].export(
        scene,
        activity,
        _WORKER["instances"],
        _WORKER["objects_mujoco"],
        _WORKER["objects"],
        _WORKER["scenes_mujoco"],
        out_root,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instances_zip", default=str(DEFAULT_INSTANCES_ZIP))
    parser.add_argument("--instructions_jsonl", default=str(DEFAULT_INSTRUCTIONS))
    parser.add_argument("--scenes_mujoco_zip", default=str(PARALLELS / "scenes_mujoco.zip"))
    parser.add_argument("--objects_mujoco_zip", default=str(PARALLELS / "objects_mujoco.zip"))
    parser.add_argument("--activities", nargs="*")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--max_instances_per_split", type=int, default=0)
    parser.add_argument("--robot_pose_key", default="R1Pro")
    parser.add_argument(
        "--robot_frame_offsets",
        default=str(ROBOT_FRAME_OFFSETS_FILE),
        help="Per-keyframe robot frame shifts (metadata/mujoco_robot_frame_offsets.json)",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.splits = set(args.splits.split(","))

    with open(args.instances_zip, "rb") as f:
        args.instances_sha256 = _sha256(f.read())
    try:
        args.exporter_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pylint: disable=broad-except
        args.exporter_commit = None
    args.instructions = {}
    with open(args.instructions_jsonl) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                args.instructions[entry["task_name"]] = entry
    args.human_lengths = {}
    with zipfile.ZipFile(args.instances_zip) as zf:
        args.available_tasks = yaml.safe_load(zf.read("metadata/available_tasks.yaml"))
        for line in zf.read("metadata/task.jsonl").decode().splitlines():
            if line.strip():
                entry = json.loads(line)
                args.human_lengths[entry["task_name"]] = entry["length"]
        targets = sorted(
            (m["scene"], m["activity"]) for m in (TEMPLATE_PATTERN.match(n) for n in zf.namelist()) if m
        )
    if args.activities:
        targets = [t for t in targets if t[1] in set(args.activities)]

    scene_manifests = {}
    with zipfile.ZipFile(args.scenes_mujoco_zip) as zf:
        for scene in sorted({s for s, _ in targets}):
            member = f"scenes/{scene}/mjcf/{scene}_parts.json"
            if member not in zf.namelist():
                raise SystemExit(f"{args.scenes_mujoco_zip} has no {member}")
            scene_manifests[scene] = json.loads(zf.read(member))
    joint_order = json.loads(JOINT_ORDER_FILE.read_text())["models"] if JOINT_ORDER_FILE.exists() else {}

    out_name = "tasks_mujoco_debug.zip" if args.activities else "tasks_mujoco.zip"
    with b1k_pipeline.utils.ParallelZipFS(out_name, write=True) as archive_fs:
        out_root = archive_fs.getsyspath("/")
        errors, warnings = {}, {}
        with futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(args, scene_manifests, joint_order)
        ) as executor:
            target_futures = {executor.submit(_export_one, s, a, out_root): (s, a) for s, a in targets}
            for future in tqdm.tqdm(futures.as_completed(target_futures), total=len(target_futures)):
                scene, activity = target_futures[future]
                try:
                    found = future.result()
                    if found:
                        warnings[activity] = found
                except Exception:  # pylint: disable=broad-except
                    errors[activity] = traceback.format_exc()
                    traceback.print_exc()

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    out_json = "export_tasks_mujoco_debug.json" if args.activities else "export_tasks_mujoco.json"
    with pipeline_fs.pipeline_output().open(out_json, "w") as f:
        json.dump({"success": not errors, "errors": errors, "warnings": warnings}, f, indent=1)


if __name__ == "__main__":
    main()
