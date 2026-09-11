"""Validate 2026 challenge task instances and show sampled poses on floor plans."""

import argparse
import base64
import csv
import json
import math
import re
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from constants import DATASET_2026_PATH
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils import transform_utils_np as T


SCRIPT_DESCRIPTION = "Validate 2026 challenge task instances and show sampled poses on floor plans."
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = Path(DATASET_2026_PATH)
DEFAULT_ASSET_DATASET_DIR = Path(get_dataset_path("behavior-1k-assets"))
DEFAULT_ASSET_SCENES_DIR = DEFAULT_ASSET_DATASET_DIR / "scenes"
ROOM_CATEGORIES_PATH = DEFAULT_ASSET_DATASET_DIR / "metadata" / "room_categories.txt"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

METADATA_FILENAMES = {"B100_task_misc.csv", "available_tasks.yaml", "task_custom_lists.json"}
INSTANCE_SUFFIX = "_template-tro_state.json"
MAP_RESOLUTION = 0.01
UNKNOWN_TASK_ID_SORT_KEY = 10_000
QUATERNION_NORM_TOLERANCE = 1e-3
POSE_KEYS_TO_SKIP = {"robot_poses"}
STATIC_OBJECT_PREFIXES = ("floor.", "wall.", "ceiling.")
DEFAULT_PALETTE = [
    "#1577d8",
    "#f59e0b",
    "#1f7a4d",
    "#8b5cf6",
    "#ef4444",
    "#14b8a6",
    "#64748b",
    "#e11d48",
    "#0ea5e9",
    "#84cc16",
    "#f97316",
    "#7c3aed",
    "#06b6d4",
    "#be123c",
    "#475569",
    "#b45309",
]

# Data models


@dataclass
class CheckLine:
    text: str
    ok: bool
    detail: str = ""


@dataclass
class TaskPaths:
    task_name: str
    scene: str | None = None
    instance_dir: Path | None = None
    prefix: str | None = None
    template_path: Path | None = None
    partial_rooms_path: Path | None = None
    stable_path: Path | None = None


@dataclass
class PoseStats:
    count: int = 0
    std_x: float = 0.0
    std_y: float = 0.0
    std_xy: float = 0.0


@dataclass
class TaskReport:
    task_name: str
    task_id: int | None = None
    scene: str | None = None
    rooms: list[str] = field(default_factory=list)
    checks: list[CheckLine] = field(default_factory=list)
    robot_stats: PoseStats = field(default_factory=PoseStats)
    object_stats: dict[str, PoseStats] = field(default_factory=dict)
    robot_points: list[dict] = field(default_factory=list)
    object_points: dict[str, list[dict]] = field(default_factory=dict)
    missing_instance_ids: list[int] = field(default_factory=list)
    extra_instance_files: list[str] = field(default_factory=list)
    invalid_json_files: list[str] = field(default_factory=list)
    map_payload: dict | None = None

    @property
    def ok(self):
        return all(check.ok for check in self.checks)


# Loading helpers


def yes_no(value):
    return "Yes" if value else "No"


def read_json(path):
    with path.open("r") as f:
        return json.load(f)


def load_json_file(path, duplicate_keys=None):
    if duplicate_keys is None:
        return read_json(path)

    def hook(pairs):
        seen = set()
        data = {}
        for key, value in pairs:
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            data[key] = value
        return data

    with path.open("r") as f:
        return json.load(f, object_pairs_hook=hook)


def load_yaml(path):
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_or_report(load_fn, path, default, errors):
    if not path.exists():
        errors.append(f"{path.parent.name}/{path.name}: missing")
        return default
    try:
        return load_fn(path)
    except Exception as exc:
        errors.append(f"{path.parent.name}/{path.name}: {exc}")
        return default


def require_mapping(value, label, errors):
    if isinstance(value, dict):
        return value
    errors.append(f"{label}: root should be an object")
    return {}


def find_top_level_yaml_duplicates(path):
    keys = []
    pattern = re.compile(r"^([A-Za-z0-9_][^:#]*):\s*$")
    for line in path.read_text().splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            keys.append(match.group(1).strip())
    counts = Counter(keys)
    return sorted(key for key, count in counts.items() if count > 1)


def load_b100_rows(path):
    rows = []
    duplicate_tasks = []
    seen = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            task_name = (row.get("Task") or "").strip()
            if not task_name:
                continue
            if task_name in seen:
                duplicate_tasks.append(task_name)
            seen.add(task_name)
            raw_rooms = row.get("Rooms to include") or row.get("Rooms to inlcude") or ""
            rooms = [room.strip() for room in raw_rooms.splitlines() if room.strip()]
            try:
                task_id = int(row.get("Task ID", ""))
            except (TypeError, ValueError):
                task_id = None
            rows.append({"task_id": task_id, "task_name": task_name, "rooms": rooms})
    return rows, sorted(duplicate_tasks)


def discover_task_paths(dataset_dir):
    task_paths = {}
    extra_instance_dirs = []
    for instance_dir in sorted((dataset_dir / "scenes").glob("*/json/*_instances")):
        scene = instance_dir.parents[1].name
        expected_start = f"{scene}_task_"
        if not instance_dir.name.startswith(expected_start):
            extra_instance_dirs.append(instance_dir)
            continue
        task_name = instance_dir.name.removeprefix(expected_start).removesuffix("_instances")
        prefix = f"{scene}_task_{task_name}"
        task_paths[task_name] = TaskPaths(
            task_name=task_name,
            scene=scene,
            instance_dir=instance_dir,
            prefix=prefix,
            template_path=instance_dir.parent / f"{prefix}_0_0_template.json",
            partial_rooms_path=instance_dir.parent / f"{prefix}_0_0_template-partial_rooms.json",
            stable_path=instance_dir.parent / f"{scene}_stable.json",
        )
    return task_paths, extra_instance_dirs


def task_order_from_b100(task_names, b100_rows):
    id_by_task = {row["task_name"]: row["task_id"] for row in b100_rows}
    return sorted(
        task_names,
        key=lambda name: (
            id_by_task.get(name) is None,
            id_by_task.get(name, UNKNOWN_TASK_ID_SORT_KEY),
            name,
        ),
    )


def choose_tasks(available_tasks, b100_rows):
    return task_order_from_b100(available_tasks.keys(), b100_rows)


def parse_instance_id(path, prefix):
    match = re.match(rf"^{re.escape(prefix)}_0_(\d+)_template-tro_state\.json$", path.name)
    return int(match.group(1)) if match else None


# Pose helpers


def instance_state_path(paths, instance_id):
    return paths.instance_dir / f"{paths.prefix}_0_{instance_id}_template-tro_state.json"


def template_task_metadata(data):
    if not isinstance(data, dict):
        raise ValueError("template JSON root should be an object")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata should be an object")
    task_metadata = metadata.get("task")
    if not isinstance(task_metadata, dict):
        raise ValueError("metadata.task should be an object")
    return task_metadata


def safe_template_task_metadata(data):
    try:
        return template_task_metadata(data)
    except ValueError:
        return {}


def is_number_sequence(value, min_length):
    return (
        isinstance(value, list)
        and len(value) >= min_length
        and all(
            isinstance(component, (int, float)) and not isinstance(component, bool) and math.isfinite(component)
            for component in value[:min_length]
        )
    )


def is_valid_position(value, min_length=2):
    return is_number_sequence(value, min_length)


def is_valid_unit_quaternion(value):
    if not is_number_sequence(value, 4):
        return False
    norm = math.sqrt(sum(component * component for component in value[:4]))
    return abs(norm - 1.0) <= QUATERNION_NORM_TOLERANCE


def require_dict_field(data, key, context):
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} should be an object")
    return value


def check_robot_pose_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    for poses in value.values():
        if not isinstance(poses, list) or not poses:
            return False
        for pose in poses:
            if not isinstance(pose, dict):
                return False
            position = pose.get("position")
            orientation = pose.get("orientation")
            if not is_valid_position(position):
                return False
            if not is_valid_unit_quaternion(orientation):
                return False
    return True


def nested_template_robot_poses(data):
    return safe_template_task_metadata(data).get("robot_poses")


def first_position_from_robot_poses(robot_poses):
    if not isinstance(robot_poses, dict):
        return None
    for poses in robot_poses.values():
        if not isinstance(poses, list) or not poses:
            continue
        position = poses[0].get("position") if isinstance(poses[0], dict) else None
        if is_valid_position(position):
            return position
    return None


def first_robot_position(data):
    robot_poses = data.get("robot_poses") if isinstance(data, dict) else None
    return first_position_from_robot_poses(robot_poses)


def first_template_robot_position(data):
    return first_position_from_robot_poses(nested_template_robot_poses(data))


def object_root_position(value):
    if not isinstance(value, dict):
        return None
    root_link = value.get("root_link")
    if not isinstance(root_link, dict):
        return None
    position = root_link.get("pos")
    return position if is_valid_position(position) else None


def object_root_pose(value):
    if not isinstance(value, dict):
        return None
    root_link = value.get("root_link")
    if not isinstance(root_link, dict):
        return None
    position = root_link.get("pos")
    orientation = root_link.get("ori")
    if not is_valid_position(position, min_length=3):
        return None
    if not is_valid_unit_quaternion(orientation):
        return None
    return position, orientation


def transform_local_position(position, parent_state):
    pose = object_root_pose(parent_state)
    if pose is None or not is_valid_position(position, min_length=3):
        return position
    parent_position, parent_orientation = pose
    world, _ = T.pose_transform(
        np.asarray(parent_position[:3], dtype=np.float32),
        np.asarray(parent_orientation[:4], dtype=np.float32),
        np.asarray(position[:3], dtype=np.float32),
        np.asarray([0, 0, 0, 1], dtype=np.float32),
    )
    return world.tolist()


def template_scene_to_bddl_names(data):
    task_metadata = safe_template_task_metadata(data)
    inst_to_name = task_metadata.get("inst_to_name") if isinstance(task_metadata, dict) else None
    if not isinstance(inst_to_name, dict):
        return {}
    return {scene_name: bddl_name for bddl_name, scene_name in inst_to_name.items()}


def attached_object_state(group_name, object_states, scene_to_bddl_name):
    if not isinstance(object_states, dict):
        return None
    return object_states.get(group_name) or object_states.get(scene_to_bddl_name.get(group_name))


def particle_positions(value):
    if not isinstance(value, dict):
        return []
    positions = value.get("positions")
    if not isinstance(positions, list):
        return []
    return [position for position in positions if is_valid_position(position)]


def particle_world_positions(value, object_states, scene_to_bddl_name=None):
    positions = particle_positions(value)
    if not positions:
        return []

    groups = value.get("groups") if isinstance(value, dict) else None
    if not isinstance(groups, dict) or not groups:
        return positions

    if scene_to_bddl_name is None:
        scene_to_bddl_name = {}
    transformed = {}
    single_group = len(groups) == 1
    for group_name, group_info in groups.items():
        if not isinstance(group_info, dict):
            continue
        indices = group_info.get("particle_indices")
        if not isinstance(indices, list):
            indices = list(range(len(positions))) if single_group else []
        parent_state = attached_object_state(group_name, object_states, scene_to_bddl_name)
        for index in indices:
            if isinstance(index, int) and 0 <= index < len(positions):
                transformed[index] = transform_local_position(positions[index], parent_state)

    return [transformed.get(index, position) for index, position in enumerate(positions)]


def should_collect_object_pose(object_name):
    return object_name not in POSE_KEYS_TO_SKIP and not object_name.startswith(STATIC_OBJECT_PREFIXES)


def object_world_positions(object_name, object_data, state_data, scene_to_bddl_name):
    if not should_collect_object_pose(object_name):
        return []
    position = object_root_position(object_data)
    if position is not None:
        return [position]
    return particle_world_positions(object_data, state_data, scene_to_bddl_name)


def template_object_positions(data):
    task_metadata = safe_template_task_metadata(data)
    inst_to_name = task_metadata.get("inst_to_name") if isinstance(task_metadata, dict) else None
    if not isinstance(inst_to_name, dict):
        return {}
    state = require_dict_field(data, "state", "template")
    registry = require_dict_field(state, "registry", "template.state")
    object_registry = require_dict_field(registry, "object_registry", "template.state.registry")
    system_registry = require_dict_field(registry, "system_registry", "template.state.registry")

    positions_by_name = defaultdict(list)
    for bddl_name, scene_name in inst_to_name.items():
        if "agent" in bddl_name or not should_collect_object_pose(bddl_name):
            continue
        object_data = object_registry.get(scene_name)
        position = object_root_position(object_data)
        if position is not None:
            positions_by_name[bddl_name].append(position)
            continue
        for particle_position in particle_world_positions(system_registry.get(scene_name), object_registry):
            positions_by_name[bddl_name].append(particle_position)
    return dict(positions_by_name)


def compute_pose_stats(points):
    if len(points) < 2:
        return PoseStats(count=len(points))
    arr = np.asarray(points, dtype=np.float64)
    std_x = float(np.std(arr[:, 0]))
    std_y = float(np.std(arr[:, 1]))
    return PoseStats(count=len(points), std_x=std_x, std_y=std_y, std_xy=float(math.hypot(std_x, std_y)))


def short_object_label(name):
    synset = name.rsplit("_", 1)[0]
    suffix = name.rsplit("_", 1)[1] if "_" in name else ""
    base = synset.split(".")[0].replace("__", " ").replace("_", " ")
    return f"{base} {suffix}".strip()


def load_room_categories():
    return [line.strip() for line in ROOM_CATEGORIES_PATH.read_text().splitlines() if line.strip()]


# Floor plan helpers


@dataclass
class FloorPlan:
    image_uri: str
    width: int
    height: int
    rooms: list[dict]
    existing_room_names: set[str]
    transform: dict


def parse_room_maps(scene, floor=0):
    layout_dir = DEFAULT_ASSET_SCENES_DIR / scene / "layout"
    ins_path = layout_dir / f"floor_insseg_{floor}.png"
    sem_path = layout_dir / f"floor_semseg_{floor}.png"
    trav_path = layout_dir / f"floor_trav_{floor}.png"
    ins = cv2.imread(str(ins_path), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
    trav = cv2.imread(str(trav_path), cv2.IMREAD_GRAYSCALE)
    if ins is None or sem is None or trav is None:
        missing = [str(path) for path, img in [(ins_path, ins), (sem_path, sem), (trav_path, trav)] if img is None]
        raise FileNotFoundError("Missing floor map files: " + ", ".join(missing))
    if ins.shape != sem.shape or ins.shape != trav.shape:
        raise ValueError(f"Floor map shapes do not match for {scene}")
    if ins.shape[0] != ins.shape[1]:
        raise ValueError(f"Floor maps must be square for {scene}, got shape {ins.shape} for floor {floor}")

    room_cats = load_room_categories()
    sem_id_to_ins_ids = defaultdict(list)
    for ins_id in sorted(int(value) for value in np.unique(ins) if value != 0):
        ys, xs = np.where(ins == ins_id)
        sem_id = int(sem[ys[0], xs[0]])
        sem_id_to_ins_ids[sem_id].append(ins_id)

    ins_id_to_name = {}
    for sem_id, ins_ids in sem_id_to_ins_ids.items():
        sem_name = room_cats[sem_id - 1]
        for index, ins_id in enumerate(sorted(ins_ids)):
            ins_id_to_name[ins_id] = f"{sem_name}_{index}"

    return ins, trav, ins_id_to_name


def transform_map_point(raw_row, raw_col, transform):
    row = (raw_row - transform["crop_y0"]) * transform["scale_y"]
    col = (raw_col - transform["crop_x0"]) * transform["scale_x"]
    row = transform["scaled_h"] - 1 - row
    if transform["rotated"]:
        x = row
        y = transform["scaled_w"] - 1 - col
    else:
        x = col
        y = row
    return float(x), float(y)


def world_to_display_point(position, transform):
    raw_size = transform["raw_size"]
    raw_row = position[1] / MAP_RESOLUTION + raw_size / 2.0
    raw_col = position[0] / MAP_RESOLUTION + raw_size / 2.0
    x, y = transform_map_point(raw_row, raw_col, transform)
    visible = 0 <= x < transform["display_w"] and 0 <= y < transform["display_h"]
    return {"x": x, "y": y, "visible": visible}


def make_floor_plan(scene, chosen_rooms, floor=0, target_size=1200, crop_margin_px=80):
    ins, trav, ins_id_to_name = parse_room_maps(scene, floor=floor)
    ys, xs = np.where(ins > 0)
    if len(xs):
        y0 = max(int(ys.min()) - crop_margin_px, 0)
        y1 = min(int(ys.max()) + crop_margin_px + 1, ins.shape[0])
        x0 = max(int(xs.min()) - crop_margin_px, 0)
        x1 = min(int(xs.max()) + crop_margin_px + 1, ins.shape[1])
    else:
        y0, y1, x0, x1 = 0, ins.shape[0], 0, ins.shape[1]

    ins_crop = ins[y0:y1, x0:x1]
    trav_crop = trav[y0:y1, x0:x1]
    crop_h, crop_w = ins_crop.shape[:2]
    scale = min(1.0, target_size / max(crop_h, crop_w))
    scaled_w = max(1, int(round(crop_w * scale)))
    scaled_h = max(1, int(round(crop_h * scale)))
    scale_x = scaled_w / crop_w
    scale_y = scaled_h / crop_h
    ins_scaled = cv2.resize(ins_crop, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    trav_scaled = cv2.resize(trav_crop, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    ins_display = ins_scaled[::-1, :]
    trav_display = trav_scaled[::-1, :]
    rotated = ins_display.shape[0] > ins_display.shape[1]
    if rotated:
        ins_display = np.rot90(ins_display, k=1)
        trav_display = np.rot90(trav_display, k=1)

    chosen_rooms = set(chosen_rooms)
    canvas = np.full((*ins_display.shape, 3), 244, dtype=np.uint8)
    for ins_id, room_name in ins_id_to_name.items():
        room_mask = ins_display == ins_id
        if room_name in chosen_rooms:
            canvas[room_mask] = np.array([255, 218, 121], dtype=np.uint8)
        else:
            canvas[room_mask] = np.array([221, 228, 226], dtype=np.uint8)

    obstacle_mask = (trav_display == 0) & (ins_display > 0)
    canvas[obstacle_mask] = (canvas[obstacle_mask] * 0.45).astype(np.uint8)
    canvas[(trav_display == 0) & (ins_display == 0)] = np.array([38, 38, 38], dtype=np.uint8)

    success, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f"Could not encode floor plan PNG for {scene}")
    image_uri = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    transform = {
        "raw_size": ins.shape[0],
        "crop_x0": x0,
        "crop_y0": y0,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "rotated": rotated,
        "display_w": int(ins_display.shape[1]),
        "display_h": int(ins_display.shape[0]),
    }

    rooms = []
    for ins_id, name in ins_id_to_name.items():
        room_ys, room_xs = np.where(ins == ins_id)
        if len(room_xs) == 0:
            continue
        raw_row = float(np.mean(room_ys))
        raw_col = float(np.mean(room_xs))
        x, y = transform_map_point(raw_row, raw_col, transform)
        rooms.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "chosen": name in chosen_rooms,
            }
        )

    return FloorPlan(
        image_uri=image_uri,
        width=int(ins_display.shape[1]),
        height=int(ins_display.shape[0]),
        rooms=rooms,
        existing_room_names=set(ins_id_to_name.values()),
        transform=transform,
    )


# Report checks


def validate_templates(paths):
    issues = []
    for path, label in [(paths.template_path, "template"), (paths.partial_rooms_path, "partial rooms")]:
        if path is None or not path.exists():
            issues.append(f"missing {label}")
            continue
        try:
            data = read_json(path)
        except Exception as exc:
            issues.append(f"{label} JSON does not open: {exc}")
            continue
        try:
            task_metadata = template_task_metadata(data)
        except ValueError as exc:
            issues.append(f"{label} metadata is invalid: {exc}")
            continue
        if not check_robot_pose_dict(task_metadata.get("robot_poses")):
            issues.append(f"{label} robot pose is missing or invalid")
    return issues


def analyze_task(paths, task_id, rooms, args):
    report = TaskReport(task_name=paths.task_name, task_id=task_id, scene=paths.scene, rooms=rooms)
    if paths.instance_dir is None or not paths.instance_dir.exists():
        report.checks.append(CheckLine("Task folder exists", False, "missing instance folder"))
        return report

    report.checks.append(CheckLine("Task folder exists", True))
    template_issues = validate_templates(paths)
    report.checks.append(
        CheckLine("Template and partial rooms are there", not template_issues, "; ".join(template_issues))
    )
    report.checks.append(
        CheckLine("Scene stable file is there", paths.stable_path is not None and paths.stable_path.exists())
    )

    instance_pattern = f"*{INSTANCE_SUFFIX}"
    actual_files = sorted(paths.instance_dir.glob(instance_pattern))
    expected_ids = set(range(1, args.expected_instances + 1))
    actual_ids = {}
    report.extra_instance_files = []
    for path in actual_files:
        instance_id = parse_instance_id(path, paths.prefix)
        if instance_id in expected_ids:
            actual_ids[instance_id] = path
        else:
            report.extra_instance_files.append(str(path.relative_to(args.dataset_dir)))
    report.missing_instance_ids = sorted(expected_ids - set(actual_ids))
    report.checks.append(
        CheckLine(
            f"Task has {args.expected_instances} instances",
            not report.missing_instance_ids and not report.extra_instance_files,
            format_instance_detail(report.missing_instance_ids, report.extra_instance_files),
        )
    )

    robot_positions = []
    object_positions = defaultdict(list)
    invalid_robot_pose_ids = []
    valid_json_count = 0
    try:
        template_data = (
            read_json(paths.template_path) if paths.template_path is not None and paths.template_path.exists() else {}
        )
    except Exception:
        template_data = {}
    scene_to_bddl_name = template_scene_to_bddl_names(template_data)
    for instance_id in range(1, args.expected_instances + 1):
        path = instance_state_path(paths, instance_id)
        if not path.exists():
            invalid_robot_pose_ids.append(instance_id)
            continue
        try:
            data = read_json(path)
            valid_json_count += 1
        except Exception as exc:
            invalid_robot_pose_ids.append(instance_id)
            report.invalid_json_files.append(f"{path.relative_to(args.dataset_dir)}: {exc}")
            continue

        if not isinstance(data, dict):
            invalid_robot_pose_ids.append(instance_id)
            report.invalid_json_files.append(
                f"{path.relative_to(args.dataset_dir)}: root should be an object, got {type(data).__name__}"
            )
            continue

        robot_poses = data.get("robot_poses")
        if not check_robot_pose_dict(robot_poses):
            invalid_robot_pose_ids.append(instance_id)
        else:
            position = first_robot_position(data)
            if position is not None:
                robot_positions.append(position[:2])

        for object_name, object_data in data.items():
            for particle_position in object_world_positions(object_name, object_data, data, scene_to_bddl_name):
                object_positions[object_name].append(particle_position[:2])

    report.checks.append(
        CheckLine("JSONs open cleanly", not report.invalid_json_files, format_list(report.invalid_json_files))
    )
    report.checks.append(
        CheckLine(
            "Every instance has a robot pose",
            not invalid_robot_pose_ids,
            format_robot_pose_detail(report.missing_instance_ids, invalid_robot_pose_ids),
        )
    )
    report.robot_stats = compute_pose_stats(robot_positions)
    report.object_stats = {name: compute_pose_stats(points) for name, points in sorted(object_positions.items())}
    robot_std_ok = report.robot_stats.std_xy >= args.min_xy_std
    object_std_values = [stats.std_xy for stats in report.object_stats.values() if stats.count >= 2]
    object_std_ok = bool(object_std_values) and max(object_std_values) >= args.min_xy_std
    report.checks.append(
        CheckLine(
            f"Robot pose std is over {args.min_xy_std:g} m",
            robot_std_ok,
            f"xy std {report.robot_stats.std_xy:.3f} m from {report.robot_stats.count} poses",
        )
    )
    report.checks.append(
        CheckLine(
            f"Object pose std is over {args.min_xy_std:g} m",
            object_std_ok,
            format_object_std_detail(report.object_stats, args.min_xy_std),
        )
    )
    report.checks.append(
        CheckLine("Valid instance JSON count is readable", valid_json_count == args.expected_instances)
    )
    return report


def format_instance_detail(missing_ids, extra_files):
    parts = []
    if missing_ids:
        parts.append("missing " + summarize_ints(missing_ids))
    if extra_files:
        parts.append("extra " + format_list(extra_files, max_items=5))
    return "; ".join(parts)


def format_robot_pose_detail(missing_ids, invalid_pose_ids):
    parts = []
    if missing_ids:
        parts.append("missing instance files " + summarize_ints(missing_ids))
    invalid_ids = sorted(set(invalid_pose_ids) - set(missing_ids))
    if invalid_ids:
        parts.append("invalid robot pose " + summarize_ints(invalid_ids))
    return "; ".join(parts)


def summarize_ints(values, max_items=12):
    values = list(values)
    if len(values) <= max_items:
        return ", ".join(str(value) for value in values)
    head = ", ".join(str(value) for value in values[:max_items])
    return f"{head}, ... (+{len(values) - max_items} more)"


def format_list(values, max_items=8):
    values = [str(value) for value in values]
    if not values:
        return ""
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f", ... (+{len(values) - max_items} more)"


def format_object_std_detail(object_stats, min_xy_std):
    if not object_stats:
        return "no object poses found"
    values = [stats.std_xy for stats in object_stats.values() if stats.count >= 2]
    if not values:
        return "not enough repeated object poses"
    low = [name for name, stats in object_stats.items() if stats.count >= 2 and stats.std_xy < min_xy_std]
    return f"min {min(values):.3f} m, median {float(np.median(values)):.3f} m, max {max(values):.3f} m" + (
        f"; low: {format_list([short_object_label(name) for name in low], max_items=5)}" if low else ""
    )


def scan_all_json(dataset_dir):
    invalid = []
    count = 0
    for path in sorted(dataset_dir.rglob("*.json")):
        if ".git" in path.parts:
            continue
        count += 1
        try:
            read_json(path)
        except Exception as exc:
            invalid.append(f"{path.relative_to(dataset_dir)}: {exc}")
    return count, invalid


def validate_dataset_shape(dataset_dir, discovered_paths, expected_instances):
    allowed_files = {
        Path("README.md"),
        *(Path("metadata") / name for name in METADATA_FILENAMES),
    }
    allowed_dirs = {
        Path("."),
        Path("metadata"),
        Path("scenes"),
    }

    for scene_json_dir in sorted((dataset_dir / "scenes").glob("*/json")):
        scene_dir = scene_json_dir.parent
        scene = scene_dir.name
        allowed_dirs.add(scene_dir.relative_to(dataset_dir))
        allowed_dirs.add(scene_json_dir.relative_to(dataset_dir))
        stable = scene_json_dir / f"{scene}_stable.json"
        allowed_files.add(stable.relative_to(dataset_dir))

    for paths in discovered_paths.values():
        if not paths.instance_dir or not paths.prefix:
            continue
        allowed_dirs.add(paths.instance_dir.relative_to(dataset_dir))
        allowed_files.add(paths.template_path.relative_to(dataset_dir))
        allowed_files.add(paths.partial_rooms_path.relative_to(dataset_dir))
        for instance_id in range(1, expected_instances + 1):
            allowed_files.add(instance_state_path(paths, instance_id).relative_to(dataset_dir))

    extras = []
    for path in sorted(dataset_dir.rglob("*")):
        if ".git" in path.parts:
            continue
        rel = path.relative_to(dataset_dir)
        if path.is_dir():
            if rel not in allowed_dirs:
                extras.append(f"{rel}/")
        elif rel not in allowed_files:
            extras.append(str(rel))
    return extras


def available_scene_for_task(available_tasks, task_name):
    task_entry = available_tasks.get(task_name)
    if not isinstance(task_entry, dict):
        return None
    for instance_entry in task_entry.values():
        if isinstance(instance_entry, dict) and instance_entry.get("scene_model"):
            return instance_entry["scene_model"]
    return None


def custom_scenes_for_task(custom_lists, task_name):
    task_entry = custom_lists.get(task_name)
    if not isinstance(task_entry, dict):
        return []
    return sorted(key for key, value in task_entry.items() if key != "room_types" and isinstance(value, dict))


def add_metadata_scene_check(report, available_tasks, custom_lists, paths):
    available_scene = available_scene_for_task(available_tasks, report.task_name)
    custom_scenes = custom_scenes_for_task(custom_lists, report.task_name)
    expected = set(filter(None, [available_scene, *custom_scenes]))
    if not expected:
        report.checks.append(CheckLine("Metadata scene matches folder", False, "no scene in metadata"))
        return
    ok = paths.scene in expected if paths.scene is not None else False
    detail = "" if ok else f"folder scene {paths.scene}; metadata scenes {format_list(sorted(expected))}"
    report.checks.append(CheckLine("Metadata scene matches folder", ok, detail))


def build_reports(args):
    dataset_dir = args.dataset_dir.expanduser().resolve()
    args.dataset_dir = dataset_dir
    metadata_dir = dataset_dir / "metadata"
    available_path = metadata_dir / "available_tasks.yaml"
    custom_lists_path = metadata_dir / "task_custom_lists.json"
    b100_path = metadata_dir / "B100_task_misc.csv"

    metadata_load_errors = []
    available_tasks = load_or_report(load_yaml, available_path, {}, metadata_load_errors)
    available_tasks = require_mapping(available_tasks, "available_tasks.yaml", metadata_load_errors)
    available_duplicates = find_top_level_yaml_duplicates(available_path) if available_path.exists() else []
    custom_duplicate_keys = []
    custom_lists = load_or_report(
        lambda path: load_json_file(path, duplicate_keys=custom_duplicate_keys),
        custom_lists_path,
        {},
        metadata_load_errors,
    )
    custom_lists = require_mapping(custom_lists, "task_custom_lists.json", metadata_load_errors)
    b100_rows, b100_duplicates = load_or_report(load_b100_rows, b100_path, ([], []), metadata_load_errors)
    discovered_paths, extra_instance_dirs = discover_task_paths(dataset_dir)
    selected_tasks = choose_tasks(available_tasks, b100_rows)

    b100_by_task = {row["task_name"]: row for row in b100_rows}
    reports = []
    for task_name in selected_tasks:
        paths = discovered_paths.get(task_name, TaskPaths(task_name=task_name))
        b100_row = b100_by_task.get(task_name, {})
        reports.append(
            analyze_task(
                paths=paths,
                task_id=b100_row.get("task_id"),
                rooms=b100_row.get("rooms", []),
                args=args,
            )
        )
        add_metadata_scene_check(reports[-1], available_tasks, custom_lists, paths)

    global_json_count, global_invalid_json = scan_all_json(dataset_dir)
    readable_errors = sorted(set(global_invalid_json + metadata_load_errors))
    discovered_task_names = set(discovered_paths)
    selected_set = set(selected_tasks)
    available_set = set(available_tasks)
    custom_set = set(custom_lists)
    b100_set = set(b100_by_task)
    metadata_selected_ok = selected_set <= available_set and selected_set <= custom_set and selected_set <= b100_set
    discovered_selected_ok = selected_set <= discovered_task_names
    metadata_sets_match = available_set == custom_set
    extra_discovered_tasks = sorted(discovered_task_names - available_set)
    readme_text = (dataset_dir / "README.md").read_text(errors="ignore") if (dataset_dir / "README.md").exists() else ""
    readme_missing = sorted(task for task in selected_tasks if task not in readme_text)
    dataset_extras = validate_dataset_shape(dataset_dir, discovered_paths, args.expected_instances)
    duplicate_ok = not (available_duplicates or custom_duplicate_keys or b100_duplicates)

    global_checks = [
        CheckLine("Metadata and JSON files open cleanly", not readable_errors, format_list(readable_errors)),
        CheckLine(
            "Selected tasks are in every metadata file",
            metadata_selected_ok,
            metadata_detail(selected_set, available_set, custom_set, b100_set),
        ),
        CheckLine(
            "2026 available_tasks and task_custom_lists match",
            metadata_sets_match,
            metadata_set_diff_detail(available_set, custom_set),
        ),
        CheckLine(
            "No duplicates",
            duplicate_ok,
            "" if duplicate_ok else duplicate_detail(available_duplicates, custom_duplicate_keys, b100_duplicates),
        ),
        CheckLine("README mentions every selected task", not readme_missing, format_list(readme_missing)),
        CheckLine(
            "Selected tasks have folders",
            discovered_selected_ok,
            format_list(sorted(selected_set - discovered_task_names)),
        ),
        CheckLine(
            "No incomplete task folders",
            not (extra_discovered_tasks or extra_instance_dirs),
            format_list(extra_discovered_tasks + [str(path.relative_to(dataset_dir)) for path in extra_instance_dirs]),
        ),
        CheckLine("No extra file/folder than required", not dataset_extras, format_list(dataset_extras, max_items=12)),
    ]
    return {
        "dataset_dir": dataset_dir,
        "selected_tasks": selected_tasks,
        "reports": reports,
        "global_checks": global_checks,
        "global_json_count": global_json_count,
    }


def metadata_detail(selected, available, custom, b100):
    parts = []
    for label, source in [("available_tasks", available), ("task_custom_lists", custom), ("B100_task_misc", b100)]:
        missing = sorted(selected - source)
        if missing:
            parts.append(f"{label} missing {format_list(missing)}")
    return "; ".join(parts)


def metadata_set_diff_detail(available, custom):
    parts = []
    missing_custom = sorted(available - custom)
    missing_available = sorted(custom - available)
    if missing_custom:
        parts.append("custom missing " + format_list(missing_custom))
    if missing_available:
        parts.append("available missing " + format_list(missing_available))
    return "; ".join(parts)


def duplicate_detail(available_duplicates, custom_duplicate_keys, b100_duplicates):
    parts = []
    if available_duplicates:
        parts.append("available_tasks " + format_list(available_duplicates))
    if custom_duplicate_keys:
        parts.append("task_custom_lists " + format_list(sorted(set(custom_duplicate_keys))))
    if b100_duplicates:
        parts.append("B100 CSV " + format_list(b100_duplicates))
    return "; ".join(parts)


def append_visible_point(points, position, transform, instance_id):
    point = world_to_display_point(position, transform)
    point["instance"] = instance_id
    if point["visible"]:
        points.append(point)


def attach_gui_payloads(reports, args):
    cache = {}
    task_paths = discover_task_paths(args.dataset_dir)[0]
    for report in reports:
        if report.scene is None:
            continue
        cache_key = (report.scene, tuple(sorted(set(report.rooms))))
        if cache_key not in cache:
            try:
                cache[cache_key] = make_floor_plan(
                    scene=report.scene,
                    chosen_rooms=report.rooms,
                    floor=args.floor,
                    target_size=args.target_size,
                    crop_margin_px=args.crop_margin_px,
                )
            except Exception as exc:
                report.checks.append(CheckLine("Floor map opens", False, str(exc)))
                continue
        floor_plan = cache[cache_key]
        missing_rooms = sorted(set(report.rooms) - floor_plan.existing_room_names)
        report.checks.append(
            CheckLine("CSV rooms exist in the floor map", not missing_rooms, format_list(missing_rooms))
        )

        transform = floor_plan.transform
        report.map_payload = {
            "image": floor_plan.image_uri,
            "width": floor_plan.width,
            "height": floor_plan.height,
            "rooms": floor_plan.rooms,
        }

        paths = task_paths.get(report.task_name)
        if paths is None or paths.instance_dir is None:
            continue
        scene_to_bddl_name = {}
        if paths.template_path is not None and paths.template_path.exists():
            try:
                template_data = read_json(paths.template_path)
            except Exception:
                template_data = None
            scene_to_bddl_name = template_scene_to_bddl_names(template_data) if template_data is not None else {}
            if template_data is not None:
                template_robot_position = first_template_robot_position(template_data)
                if template_robot_position is not None:
                    append_visible_point(report.robot_points, template_robot_position, transform, instance_id=0)
                try:
                    template_positions = template_object_positions(template_data)
                except ValueError as exc:
                    report.checks.append(CheckLine("Template object registry is valid", False, str(exc)))
                else:
                    for object_name, positions in template_positions.items():
                        for object_position in positions:
                            append_visible_point(
                                report.object_points.setdefault(object_name, []),
                                object_position,
                                transform,
                                instance_id=0,
                            )
        for instance_id in range(1, args.expected_instances + 1):
            path = instance_state_path(paths, instance_id)
            if not path.exists():
                continue
            try:
                data = read_json(path)
            except Exception:
                continue
            robot_position = first_robot_position(data)
            if robot_position is not None:
                append_visible_point(report.robot_points, robot_position, transform, instance_id)
            for object_name, object_data in data.items():
                for object_position in object_world_positions(object_name, object_data, data, scene_to_bddl_name):
                    append_visible_point(
                        report.object_points.setdefault(object_name, []),
                        object_position,
                        transform,
                        instance_id,
                    )


def print_report(result, min_xy_std, expected_instances, max_details):
    reports = result["reports"]
    global_checks = result["global_checks"]
    print(f"Dataset: {result['dataset_dir']}")
    print(f"Tasks checked: {len(reports)}")
    print(f"JSON files seen: {result['global_json_count']}")
    print()

    for check in global_checks:
        print(f"{check.text}: {yes_no(check.ok)}")
    task_all_checks = [
        (
            "Every task has the required files",
            lambda report: (
                check_by_text(report, "Template and partial rooms are there")
                and check_by_text(report, "Scene stable file is there")
            ),
        ),
        (
            f"Every task has {expected_instances} instances",
            lambda report: check_by_text(report, f"Task has {expected_instances} instances"),
        ),
        ("Every instance has a robot pose", lambda report: check_by_text(report, "Every instance has a robot pose")),
        (f"Robot pose std is over {min_xy_std:g} m", lambda report: report.robot_stats.std_xy >= min_xy_std),
        (
            f"Object pose std is over {min_xy_std:g} m",
            lambda report: (
                bool(report.object_stats)
                and max((stats.std_xy for stats in report.object_stats.values()), default=0.0) >= min_xy_std
            ),
        ),
    ]
    for label, predicate in task_all_checks:
        print(f"{label}: {yes_no(bool(reports) and all(predicate(report) for report in reports))}")

    print("\nStd by task:")
    for report in reports:
        object_values = [stats.std_xy for stats in report.object_stats.values() if stats.count >= 2]
        object_summary = "none"
        if object_values:
            object_summary = (
                f"min {min(object_values):.3f}, median {float(np.median(object_values)):.3f}, "
                f"max {max(object_values):.3f}"
            )
        print(f"- {report.task_name}: robot {report.robot_stats.std_xy:.3f} m; objects {object_summary} m")

    details = []
    for check in global_checks:
        if not check.ok and check.detail:
            details.append(f"{check.text}: {check.detail}")
    for report in reports:
        for check in report.checks:
            if not check.ok and check.detail:
                details.append(f"{report.task_name} - {check.text}: {check.detail}")
    if details:
        print("\nDetails:")
        for detail in details[:max_details]:
            print(f"- {detail}")
        if len(details) > max_details:
            print(f"- ... {len(details) - max_details} more")


def check_by_text(report, text):
    return any(check.text == text and check.ok for check in report.checks)


def task_to_gui(report, index):
    object_names = sorted(report.object_points)
    object_colors = {name: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i, name in enumerate(object_names)}
    checks = [{"text": check.text, "ok": check.ok, "detail": check.detail} for check in report.checks]
    object_stats = {
        name: {
            "label": short_object_label(name),
            "std_xy": stats.std_xy,
            "count": stats.count,
            "color": object_colors.get(name, "#666666"),
        }
        for name, stats in report.object_stats.items()
    }
    return {
        "index": index,
        "task": report.task_name,
        "task_id": report.task_id,
        "scene": report.scene,
        "ok": report.ok,
        "rooms": report.rooms,
        "checks": checks,
        "robotStats": {
            "count": report.robot_stats.count,
            "stdX": report.robot_stats.std_x,
            "stdY": report.robot_stats.std_y,
            "stdXY": report.robot_stats.std_xy,
        },
        "objectStats": object_stats,
        "map": report.map_payload,
        "robotPoints": report.robot_points,
        "objectPoints": report.object_points,
        "objectColors": object_colors,
        "objectLabels": {name: short_object_label(name) for name in object_names},
    }


def image_data_uri(path, mime_type="image/png"):
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime_type};base64,{encoded}"


def browser_url(host, port):
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{browser_host}:{port}"


def run_gui(result, args):
    try:
        from flask import Flask, render_template
    except ImportError as exc:
        raise RuntimeError("Flask is needed to show the QC GUI. Install it with `pip install flask`.") from exc

    gui_data = {
        "datasetDir": str(result["dataset_dir"]),
        "expectedInstances": args.expected_instances,
        "globalChecks": [
            {"text": check.text, "ok": check.ok, "detail": check.detail} for check in result["global_checks"]
        ],
        "tasks": [task_to_gui(report, index) for index, report in enumerate(result["reports"])],
    }
    logo_uri = image_data_uri(REPO_ROOT / "docs" / "assets" / "behavior_logo3.png")
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

    @app.route("/")
    def index():
        return render_template("challenge_instance_qc_gui.html", gui_data=gui_data, logo_uri=logo_uri)

    url = browser_url(args.host, args.port)
    print(f"\nGUI: {url}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


def parse_args():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--expected-instances", type=int, default=300)
    parser.add_argument("--min-xy-std", type=float, default=0.05)
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=1200)
    parser.add_argument("--crop-margin-px", type=int, default=80)
    parser.add_argument("--max-details", type=int, default=40)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Building QC report for all tasks...", flush=True)
    result = build_reports(args)
    attach_gui_payloads(result["reports"], args)
    print_report(
        result,
        min_xy_std=args.min_xy_std,
        expected_instances=args.expected_instances,
        max_details=args.max_details,
    )
    run_gui(result, args)


if __name__ == "__main__":
    main()
