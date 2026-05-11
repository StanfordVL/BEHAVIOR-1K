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


SCRIPT_DESCRIPTION = "Validate 2026 challenge task instances and show sampled poses on floor plans."
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "2026-challenge-task-instances"
DEFAULT_ASSET_SCENES_DIR = REPO_ROOT / "datasets" / "behavior-1k-assets" / "scenes"
ROOM_CATEGORIES_PATH = REPO_ROOT / "datasets" / "behavior-1k-assets" / "metadata" / "room_categories.txt"

METADATA_FILENAMES = {"B100_task_misc.csv", "available_tasks.yaml", "task_custom_lists.json"}
INSTANCE_SUFFIX = "_template-tro_state.json"
MAP_RESOLUTION = 0.01
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


# -----------------------------------------------------------------------------
# Small data carriers


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


# File and metadata loading


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
        return default
    try:
        return load_fn(path)
    except Exception as exc:
        errors.append(f"{path.parent.name}/{path.name}: {exc}")
        return default


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
            raw_rooms = row.get("Rooms to inlcude") or row.get("Rooms to include") or ""
            rooms = [room.strip() for room in raw_rooms.splitlines() if room.strip()]
            try:
                task_id = int(row.get("Task ID", ""))
            except ValueError:
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
    return sorted(task_names, key=lambda name: (id_by_task.get(name) is None, id_by_task.get(name, 10_000), name))


def choose_tasks(available_tasks, b100_rows):
    return task_order_from_b100(available_tasks.keys(), b100_rows)


def parse_instance_id(path, prefix):
    match = re.match(rf"^{re.escape(prefix)}_0_(\d+)_template-tro_state\.json$", path.name)
    return int(match.group(1)) if match else None


# Pose extraction


def instance_state_path(paths, instance_id):
    return paths.instance_dir / f"{paths.prefix}_0_{instance_id}_template-tro_state.json"


def template_task_metadata(data):
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    return metadata.get("task") if isinstance(metadata, dict) else None


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
            if not (isinstance(position, list) and len(position) >= 2):
                return False
            if not (isinstance(orientation, list) and len(orientation) >= 4):
                return False
    return True


def nested_template_robot_poses(data):
    task_metadata = template_task_metadata(data)
    return task_metadata.get("robot_poses") if isinstance(task_metadata, dict) else None


def first_position_from_robot_poses(robot_poses):
    if not isinstance(robot_poses, dict):
        return None
    for poses in robot_poses.values():
        if not isinstance(poses, list) or not poses:
            continue
        position = poses[0].get("position") if isinstance(poses[0], dict) else None
        if isinstance(position, list) and len(position) >= 2:
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
    return position if isinstance(position, list) and len(position) >= 2 else None


def object_root_pose(value):
    if not isinstance(value, dict):
        return None
    root_link = value.get("root_link")
    if not isinstance(root_link, dict):
        return None
    position = root_link.get("pos")
    orientation = root_link.get("ori")
    if not (isinstance(position, list) and len(position) >= 3):
        return None
    if not (isinstance(orientation, list) and len(orientation) >= 4):
        return None
    return position, orientation


def quat_to_matrix(quat):
    x, y, z, w = quat[:4]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_local_position(position, parent_state):
    pose = object_root_pose(parent_state)
    if pose is None:
        return position
    parent_position, parent_orientation = pose
    local = np.asarray(position[:3], dtype=np.float64)
    world = np.asarray(parent_position[:3], dtype=np.float64) + quat_to_matrix(parent_orientation) @ local
    return world.tolist()


def template_scene_to_bddl_names(data):
    task_metadata = template_task_metadata(data)
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
    return [position for position in positions if isinstance(position, list) and len(position) >= 2]


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
    task_metadata = template_task_metadata(data)
    inst_to_name = task_metadata.get("inst_to_name") if isinstance(task_metadata, dict) else None
    state = data.get("state") if isinstance(data, dict) else None
    registry = state.get("registry") if isinstance(state, dict) else None
    object_registry = registry.get("object_registry") if isinstance(registry, dict) else {}
    system_registry = registry.get("system_registry") if isinstance(registry, dict) else {}
    if not isinstance(inst_to_name, dict):
        return {}

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


# Floor-plan rendering


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
    visible = 0 <= x <= transform["display_w"] and 0 <= y <= transform["display_h"]
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


# QC checks


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
        if not check_robot_pose_dict(nested_template_robot_poses(data)):
            issues.append(f"{label} robot pose is missing")
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
    robot_pose_ok = True
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
            continue
        try:
            data = read_json(path)
            valid_json_count += 1
        except Exception as exc:
            robot_pose_ok = False
            report.invalid_json_files.append(f"{path.relative_to(args.dataset_dir)}: {exc}")
            continue

        robot_poses = data.get("robot_poses") if isinstance(data, dict) else None
        if not check_robot_pose_dict(robot_poses):
            robot_pose_ok = False
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
    report.checks.append(CheckLine("Every instance has a robot pose", robot_pose_ok))
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
    available_duplicates = find_top_level_yaml_duplicates(available_path) if available_path.exists() else []
    custom_duplicate_keys = []
    custom_lists = load_or_report(
        lambda path: load_json_file(path, duplicate_keys=custom_duplicate_keys),
        custom_lists_path,
        {},
        metadata_load_errors,
    )
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
        CheckLine("JSON files open cleanly", not readable_errors, format_list(readable_errors)),
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
        CheckLine("No duplicate", duplicate_ok),
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
        cache_key = (report.scene, tuple(report.rooms))
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
                for object_name, positions in template_object_positions(template_data).items():
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
            lambda report: check_by_text(report, "Template and partial rooms are there")
            and check_by_text(report, "Scene stable file is there"),
        ),
        (
            f"Every task has {expected_instances} instances",
            lambda report: check_by_text(report, f"Task has {expected_instances} instances"),
        ),
        ("Every instance has a robot pose", lambda report: check_by_text(report, "Every instance has a robot pose")),
        (f"Robot pose std is over {min_xy_std:g} m", lambda report: report.robot_stats.std_xy >= min_xy_std),
        (
            f"Object pose std is over {min_xy_std:g} m",
            lambda report: bool(report.object_stats)
            and max((stats.std_xy for stats in report.object_stats.values()), default=0.0) >= min_xy_std,
        ),
    ]
    for label, predicate in task_all_checks:
        print(f"{label}: {yes_no(all(predicate(report) for report in reports))}")

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


def render_gui_html(gui_data):
    payload = json.dumps(gui_data)
    logo_uri = image_data_uri(REPO_ROOT / "docs" / "assets" / "behavior_logo3.png")
    logo_markup = (
        f'<img class="brand-logo" src="{logo_uri}" alt="">'
        if logo_uri
        else '<span class="brand-logo brand-logo-fallback"></span>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Challenge Instance QC</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #2c3e50;
      --muted: #5a6c7d;
      --line: #d9e2ec;
      --accent: #1577d8;
      --accent-strong: #0f6fd8;
      --accent-soft: rgba(21, 119, 216, 0.1);
      --warn: #b42318;
      --ok: #1f7a4d;
      --shadow: 0 8px 24px rgba(44, 62, 80, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(240px, 312px) 1fr;
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 14px;
      overflow: auto;
    }}
    .brand {{
      display: grid;
      grid-template-columns: 44px 1fr;
      align-items: center;
      gap: 10px;
      margin: 0 0 16px;
      padding: 2px 2px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .brand-logo {{
      width: 42px;
      height: 32px;
      object-fit: contain;
    }}
    .brand-logo-fallback {{
      border-radius: 6px;
      background: linear-gradient(65deg, #0f6fd8 0, #2793ea 100%);
    }}
    .brand-name {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
      line-height: 1;
      text-transform: uppercase;
    }}
    h1 {{
      color: var(--ink);
      font-size: 16px;
      font-weight: 700;
      line-height: 1.15;
      margin: 2px 0 0;
      letter-spacing: 0;
    }}
    .task-list {{
      display: grid;
      gap: 8px;
    }}
    .task-button {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 10px;
      display: grid;
      grid-template-columns: 10px 1fr;
      gap: 9px;
      align-items: center;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      font-size: 13px;
      transition: background 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
    }}
    .task-button:hover {{
      background: var(--panel-soft);
      border-color: rgba(21, 119, 216, 0.35);
    }}
    .task-button.active {{
      border-color: var(--accent);
      background: #eef7ff;
      box-shadow: 0 0 0 2px rgba(21, 119, 216, 0.12);
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: var(--ok);
    }}
    .dot.bad {{ background: var(--warn); }}
    main {{
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      padding: 14px 20px;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .title {{
      display: grid;
      gap: 2px;
    }}
    .title strong {{ color: var(--ink); font-size: 18px; font-weight: 700; }}
    .title span {{ color: var(--muted); font-size: 13px; }}
    .mode {{
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }}
    .mode button {{
      border: 0;
      padding: 9px 14px;
      background: transparent;
      cursor: pointer;
      font-weight: 600;
      color: var(--muted);
    }}
    .mode button.active {{
      background: linear-gradient(65deg, var(--accent-strong) 0, #2793ea 100%);
      color: white;
    }}
    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
      min-height: 0;
    }}
    .map-shell {{
      position: relative;
      padding: 18px 18px 78px;
      min-width: 0;
      overflow: auto;
      background: var(--bg);
    }}
    .map-frame {{
      position: relative;
      margin: 0 auto;
      width: min(100%, 1200px);
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .map-frame img, .map-frame svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .map-frame svg {{
      position: absolute;
      inset: 0;
    }}
    .room-label {{
      font-size: 11px;
      font-weight: 700;
      fill: var(--ink);
      paint-order: stroke;
      stroke: rgba(255, 255, 255, 0.9);
      stroke-width: 4px;
      stroke-linejoin: round;
    }}
    .legend {{
      position: absolute;
      right: 32px;
      bottom: 112px;
      max-width: 240px;
      max-height: 38%;
      overflow: auto;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      box-shadow: var(--shadow);
      font-size: 12px;
      z-index: 3;
    }}
    .legend-row {{
      display: grid;
      grid-template-columns: 11px 1fr;
      align-items: center;
      gap: 7px;
      width: 100%;
      margin: 2px 0;
      padding: 4px;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      text-align: left;
    }}
    .legend-row:hover {{
      background: var(--accent-soft);
    }}
    .legend-row.off {{
      opacity: 0.38;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }}
    .instance-control {{
      display: grid;
      justify-items: center;
      gap: 8px;
      margin: 10px auto 0;
      width: min(100%, 1200px);
    }}
    .instance-toggle {{
      width: 34px;
      height: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--accent);
      cursor: pointer;
      font-weight: 800;
    }}
    .instance-panel {{
      display: grid;
      grid-template-columns: auto minmax(180px, 520px) 42px;
      align-items: center;
      gap: 10px;
      width: min(100%, 620px);
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: var(--shadow);
      color: var(--muted);
      font-size: 13px;
    }}
    .instance-panel.hidden {{
      display: none;
    }}
    .instance-panel input {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .side-panel {{
      border-left: 1px solid var(--line);
      background: var(--panel-soft);
      padding: 16px;
      overflow: auto;
    }}
    .sheet-card {{
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
      padding: 12px;
      border: 1px solid rgba(21, 119, 216, 0.28);
      border-radius: 8px;
      background: white;
      box-shadow: 0 8px 20px rgba(44, 62, 80, 0.07);
    }}
    .sheet-question {{
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}
    .sheet-actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .sheet-button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      padding: 8px 10px;
    }}
    .sheet-button.active {{
      border-color: var(--accent);
      background: linear-gradient(65deg, var(--accent-strong) 0, #2793ea 100%);
      color: white;
    }}
    .checks {{
      display: grid;
      gap: 8px;
    }}
    .check-section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      box-shadow: 0 8px 20px rgba(44, 62, 80, 0.07);
      overflow: hidden;
    }}
    .check-section.ok {{
      border-color: rgba(31, 122, 77, 0.44);
    }}
    .check-section.bad {{
      border-color: rgba(180, 35, 24, 0.48);
      box-shadow: 0 10px 26px rgba(180, 35, 24, 0.12);
    }}
    .check-section-header {{
      width: 100%;
      border: 0;
      border-left: 5px solid var(--ok);
      background: #eef8f2;
      color: var(--ink);
      cursor: pointer;
      display: grid;
      grid-template-columns: 20px 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 11px 10px;
      text-align: left;
      font: inherit;
    }}
    .check-section.bad .check-section-header {{
      border-left-color: var(--warn);
      background: #fff1ee;
    }}
    .section-arrow {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 900;
      text-align: center;
    }}
    .section-name {{
      font-size: 12px;
      font-weight: 850;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .section-status {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--ok);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .check-section.bad .section-status {{
      color: var(--warn);
    }}
    .status-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--ok);
      box-shadow: 0 0 0 3px rgba(31, 122, 77, 0.12);
    }}
    .check-section.bad .status-dot {{
      background: var(--warn);
      box-shadow: 0 0 0 3px rgba(180, 35, 24, 0.12);
    }}
    .check-list {{
      padding: 10px;
    }}
    .check-list.hidden {{
      display: none;
    }}
    .check {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: white;
      display: grid;
      gap: 4px;
    }}
    .check-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
      font-weight: 600;
    }}
    .answer.ok {{ color: var(--ok); }}
    .answer.bad {{ color: var(--warn); }}
    .detail {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    @media (max-width: 920px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); max-height: 220px; }}
      .content {{ grid-template-columns: 1fr; }}
      .side-panel {{ border-left: 0; border-top: 1px solid var(--line); }}
      .legend {{ right: 24px; bottom: 112px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        {logo_markup}
        <div>
          <div class="brand-name">BEHAVIOR</div>
          <h1>Challenge Instance QC</h1>
        </div>
      </div>
      <div class="task-list" id="taskList"></div>
    </aside>
    <main>
      <header>
        <div class="title">
          <strong id="taskTitle"></strong>
          <span id="taskSubtitle"></span>
        </div>
        <div class="mode">
          <button id="robotMode" class="active" type="button">Robot poses</button>
          <button id="objectMode" type="button">Object poses</button>
        </div>
      </header>
      <div class="content">
        <section class="map-shell">
          <div class="map-frame" id="mapFrame">
            <img id="mapImage" alt="">
            <svg id="overlay"></svg>
          </div>
          <div class="instance-control">
            <button class="instance-toggle" id="instanceToggle" type="button" aria-expanded="false">v</button>
            <div class="instance-panel hidden" id="instancePanel">
              <label for="instanceSlider">Instance</label>
              <input id="instanceSlider" type="range" min="0" max="300" value="0">
              <strong id="instanceValue">0</strong>
            </div>
          </div>
          <div class="legend" id="legend"></div>
        </section>
        <section class="side-panel">
          <div class="sheet-card">
            <div class="sheet-question">Did you update the sheet?</div>
            <div class="sheet-actions">
              <button class="sheet-button" id="sheetYes" type="button">Yes</button>
              <button class="sheet-button active" id="sheetNo" type="button">No</button>
            </div>
          </div>
          <div class="checks">
            <section class="check-section" id="datasetSection">
              <button
                class="check-section-header"
                id="datasetHeader"
                type="button"
                aria-expanded="true"
                aria-controls="globalChecks"
              >
                <span class="section-arrow" id="datasetArrow">v</span>
                <span class="section-name">Dataset checks</span>
                <span class="section-status" id="datasetStatus"></span>
              </button>
              <div class="checks check-list" id="globalChecks"></div>
            </section>
            <section class="check-section" id="taskSection">
              <button
                class="check-section-header"
                id="taskHeader"
                type="button"
                aria-expanded="true"
                aria-controls="checks"
              >
                <span class="section-arrow" id="taskArrow">v</span>
                <span class="section-name">Task checks</span>
                <span class="section-status" id="taskStatus"></span>
              </button>
              <div class="checks check-list" id="checks"></div>
            </section>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const DATA = {payload};
    let currentTask = DATA.tasks[0]?.task;
    let mode = "robot";
    let instanceFilterOpen = false;
    let selectedInstance = 0;
    let sheetUpdated = false;
    const checkSectionsOpen = {{
      dataset: true,
      task: true,
    }};
    const visibleObjectsByTask = {{}};
    const taskList = document.getElementById("taskList");
    const taskTitle = document.getElementById("taskTitle");
    const taskSubtitle = document.getElementById("taskSubtitle");
    const mapImage = document.getElementById("mapImage");
    const overlay = document.getElementById("overlay");
    const legend = document.getElementById("legend");
    const instanceToggle = document.getElementById("instanceToggle");
    const instancePanel = document.getElementById("instancePanel");
    const instanceSlider = document.getElementById("instanceSlider");
    const instanceValue = document.getElementById("instanceValue");
    const globalChecks = document.getElementById("globalChecks");
    const checks = document.getElementById("checks");
    const datasetSection = document.getElementById("datasetSection");
    const taskSection = document.getElementById("taskSection");
    const datasetHeader = document.getElementById("datasetHeader");
    const taskHeader = document.getElementById("taskHeader");
    const datasetArrow = document.getElementById("datasetArrow");
    const taskArrow = document.getElementById("taskArrow");
    const datasetStatus = document.getElementById("datasetStatus");
    const taskStatus = document.getElementById("taskStatus");
    const sheetYes = document.getElementById("sheetYes");
    const sheetNo = document.getElementById("sheetNo");
    const robotMode = document.getElementById("robotMode");
    const objectMode = document.getElementById("objectMode");

    function taskByName(name) {{
      return DATA.tasks.find(task => task.task === name);
    }}

    function setMode(nextMode) {{
      mode = nextMode;
      robotMode.classList.toggle("active", mode === "robot");
      objectMode.classList.toggle("active", mode === "object");
      render();
    }}

    function getVisibleObjects(task) {{
      if (!visibleObjectsByTask[task.task]) {{
        visibleObjectsByTask[task.task] = new Set(Object.keys(task.objectPoints));
      }}
      return visibleObjectsByTask[task.task];
    }}

    function pointPassesInstance(point) {{
      return !instanceFilterOpen || Number(point.instance) === selectedInstance;
    }}

    function renderTaskList() {{
      taskList.innerHTML = "";
      DATA.tasks.forEach(task => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "task-button" + (task.task === currentTask ? " active" : "");
        button.innerHTML = `<span class="dot ${{task.ok ? "" : "bad"}}"></span><span>${{task.task}}</span>`;
        button.addEventListener("click", () => {{
          currentTask = task.task;
          render();
        }});
        taskList.appendChild(button);
      }});
    }}

    function circle(x, y, r, color, opacity, extra = "") {{
      return [
        `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{r}}"`,
        `fill="${{color}}" fill-opacity="${{opacity}}"`,
        `stroke="white" stroke-width="0.55" ${{extra}} />`,
      ].join(" ");
    }}

    function renderOverlay(task) {{
      const map = task.map;
      if (!map) {{
        overlay.innerHTML = "";
        return;
      }}
      overlay.setAttribute("viewBox", `0 0 ${{map.width}} ${{map.height}}`);
      let html = "";
      map.rooms.forEach(room => {{
        if (room.chosen) {{
          html += [
            `<text class="room-label"`,
            `x="${{room.x.toFixed(1)}}"`,
            `y="${{room.y.toFixed(1)}}"`,
            `text-anchor="middle">${{room.name}}</text>`,
          ].join(" ");
        }}
      }});

      if (mode === "robot") {{
        task.robotPoints.forEach(point => {{
          if (!pointPassesInstance(point)) return;
          html += circle(point.x, point.y, 4.2, "#111111", 0.58);
        }});
      }} else {{
        const visibleObjects = getVisibleObjects(task);
        Object.keys(task.objectPoints).forEach(name => {{
          if (!visibleObjects.has(name)) return;
          const color = task.objectColors[name] || "#666666";
          task.objectPoints[name].forEach(point => {{
            if (!pointPassesInstance(point)) return;
            const radius = name.includes("dust.") || name.includes("sand.") ? 2.2 : 3.4;
            const opacity = name.includes("dust.") || name.includes("sand.") ? 0.62 : 0.48;
            html += circle(point.x, point.y, radius, color, opacity);
          }});
        }});
      }}
      overlay.innerHTML = html;
    }}

    function renderLegend(task) {{
      if (mode === "robot") {{
        legend.innerHTML = [
          `<button class="legend-row" type="button">`,
          `<span class="swatch" style="background:#111111"></span>`,
          `<span>robot</span>`,
          `</button>`,
        ].join("");
        return;
      }}
      const names = Object.keys(task.objectPoints);
      const visibleObjects = getVisibleObjects(task);
      legend.innerHTML = names.map(name => {{
        const color = task.objectColors[name] || "#666666";
        const label = task.objectLabels[name] || name;
        const offClass = visibleObjects.has(name) ? "" : " off";
        return [
          `<button class="legend-row${{offClass}}" type="button" data-object-name="${{name}}">`,
          `<span class="swatch" style="background:${{color}}"></span>`,
          `<span>${{label}}</span>`,
          `</button>`,
        ].join("");
      }}).join("");
      legend.querySelectorAll("[data-object-name]").forEach(row => {{
        row.addEventListener("click", () => {{
          const name = row.getAttribute("data-object-name");
          if (visibleObjects.has(name)) {{
            visibleObjects.delete(name);
          }} else {{
            visibleObjects.add(name);
          }}
          render();
        }});
      }});
    }}

    function renderInstanceControl() {{
      instancePanel.classList.toggle("hidden", !instanceFilterOpen);
      instanceToggle.textContent = instanceFilterOpen ? "^" : "v";
      instanceToggle.setAttribute("aria-expanded", instanceFilterOpen ? "true" : "false");
      instanceSlider.max = DATA.expectedInstances || 300;
      instanceSlider.value = selectedInstance;
      instanceValue.textContent = instanceFilterOpen ? selectedInstance.toString() : "all";
    }}

    function renderCheckList(target, checkList) {{
      target.innerHTML = checkList.map(check => {{
        const statusClass = check.ok ? "ok" : "bad";
        const answer = check.ok ? "Yes" : "No";
        const detail = check.detail ? `<div class="detail">${{check.detail}}</div>` : "";
        return [
          `<div class="check">`,
          `<div class="check-top">`,
          `<span>${{check.text}}</span>`,
          `<span class="answer ${{statusClass}}">${{answer}}</span>`,
          `</div>`,
          detail,
          `</div>`,
        ].join("");
      }}).join("");
    }}

    function renderCheckSection(section, header, arrow, status, list, checkList, isOpen) {{
      const ok = checkList.every(check => check.ok);
      section.classList.toggle("ok", ok);
      section.classList.toggle("bad", !ok);
      arrow.textContent = isOpen ? "v" : ">";
      header.setAttribute("aria-expanded", isOpen ? "true" : "false");
      status.innerHTML = ok
        ? `<span class="status-dot"></span>`
        : `<span class="status-dot"></span><span>Needs check</span>`;
      list.classList.toggle("hidden", !isOpen);
    }}

    function renderSheetQuestion() {{
      sheetYes.classList.toggle("active", sheetUpdated);
      sheetNo.classList.toggle("active", !sheetUpdated);
    }}

    function renderChecks(task) {{
      renderCheckList(globalChecks, DATA.globalChecks);
      renderCheckList(checks, task.checks);
      renderCheckSection(
        datasetSection,
        datasetHeader,
        datasetArrow,
        datasetStatus,
        globalChecks,
        DATA.globalChecks,
        checkSectionsOpen.dataset
      );
      renderCheckSection(
        taskSection,
        taskHeader,
        taskArrow,
        taskStatus,
        checks,
        task.checks,
        checkSectionsOpen.task
      );
    }}

    function render() {{
      const task = taskByName(currentTask);
      if (!task) return;
      renderTaskList();
      taskTitle.textContent = task.task;
      const prefix = task.task_id === null || task.task_id === undefined ? "" : `${{task.task_id}} · `;
      taskSubtitle.textContent = `${{prefix}}${{task.scene || "missing scene"}}`;
      if (task.map) {{
        mapImage.src = task.map.image;
        mapImage.style.aspectRatio = `${{task.map.width}} / ${{task.map.height}}`;
      }} else {{
        mapImage.removeAttribute("src");
      }}
      renderOverlay(task);
      renderLegend(task);
      renderChecks(task);
      renderInstanceControl();
      renderSheetQuestion();
    }}

    robotMode.addEventListener("click", () => setMode("robot"));
    objectMode.addEventListener("click", () => setMode("object"));
    datasetHeader.addEventListener("click", () => {{
      checkSectionsOpen.dataset = !checkSectionsOpen.dataset;
      render();
    }});
    taskHeader.addEventListener("click", () => {{
      checkSectionsOpen.task = !checkSectionsOpen.task;
      render();
    }});
    sheetYes.addEventListener("click", () => {{
      sheetUpdated = true;
      renderSheetQuestion();
    }});
    sheetNo.addEventListener("click", () => {{
      sheetUpdated = false;
      renderSheetQuestion();
    }});
    instanceToggle.addEventListener("click", () => {{
      instanceFilterOpen = !instanceFilterOpen;
      if (!instanceFilterOpen) {{
        selectedInstance = 0;
      }}
      render();
    }});
    instanceSlider.addEventListener("input", () => {{
      selectedInstance = Number(instanceSlider.value);
      render();
    }});
    render();
  </script>
</body>
</html>"""


def run_gui(result, args):
    try:
        from flask import Flask
    except ImportError as exc:
        raise RuntimeError("Flask is needed to show the QC GUI.") from exc

    gui_data = {
        "datasetDir": str(result["dataset_dir"]),
        "expectedInstances": args.expected_instances,
        "globalChecks": [
            {"text": check.text, "ok": check.ok, "detail": check.detail} for check in result["global_checks"]
        ],
        "tasks": [task_to_gui(report, index) for index, report in enumerate(result["reports"])],
    }
    html = render_gui_html(gui_data)
    app = Flask(__name__)

    @app.route("/")
    def index():
        return html

    url = f"http://{args.host}:{args.port}"
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
