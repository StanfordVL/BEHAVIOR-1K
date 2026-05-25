import json
import argparse
import csv
import glob
import os
import textwrap
from collections import defaultdict
import numpy as np
from omnigibson.utils.bddl_utils import get_knowledge_base, GOOD_MODELS, BAD_CLOTH_MODELS
from omnigibson.utils.asset_utils import get_all_object_category_models, get_scene_path
from constants import DATASET_2025_PATH, DATASET_2026_PATH, TASK_CUSTOM_LIST_PATH
from floor_plan_visualization import get_floor_plan_data


SYNSET_BASE_URL = "https://behavior.stanford.edu/knowledgebase/synsets"
TASK_MISC_HEADER = ["Task ID", "Task", "Rooms to include"]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--activity", type=str, required=True)
parser.add_argument("--no-room-gui", action="store_true", help="Skip the floor-plan room picker GUI.")
parser.add_argument("--floor", type=int, default=0, help="Floor index to show in the room picker.")


def get_2025_models_for_task(activity_name):
    """Return {synset: {category: [model_id, ...]}} for a task found in the 2025 dataset."""
    pattern = os.path.join(DATASET_2025_PATH, "scenes", "*", "json", f"*_task_{activity_name}_0_0_template.json")
    results = {}
    for template_path in glob.glob(pattern):
        try:
            with open(template_path) as f:
                d = json.load(f)
            inst_to_name = d["metadata"]["task"]["inst_to_name"]
            objs_info = d["objects_info"]["init_info"]
            for bddl_inst, obj_name in inst_to_name.items():
                if "agent" in bddl_inst or obj_name not in objs_info:
                    continue
                args = objs_info[obj_name]["args"]
                synset = "_".join(bddl_inst.split("_")[:-1])
                category = args["category"]
                model = args["model"]
                results.setdefault(synset, {}).setdefault(category, set()).add(model)
        except Exception:
            pass
    return results


def prompt_choice(prompt, options, multi=False):
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Enter index or name" + (" (comma-separated for multiple)" if multi else "") + ": ").strip()
        chosen = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and int(part) < len(options):
                chosen.append(options[int(part)])
            elif part in options:
                chosen.append(part)
            else:
                print(f"  Invalid choice: {part!r}")
                chosen = []
                break
        if chosen:
            return chosen if multi else chosen[0]


def _iter_predicates(conds):
    if isinstance(conds, str):
        return
    assert isinstance(conds, list)
    if not conds:
        return
    if any(isinstance(ele, list) for ele in conds):
        for ele in conds:
            yield from _iter_predicates(ele)
    else:
        yield conds


def _synset_from_bddl_name(name, kb):
    for i in range(len(name.split("_")), 0, -1):
        candidate = "_".join(name.split("_")[:i])
        if kb.get_synset(candidate) is not None:
            return candidate
    return None


def _get_leaf_categories(synset_obj):
    leaf_synsets = [synset_obj] if synset_obj.is_leaf else [d for d in synset_obj.descendants if d.is_leaf]
    return sorted({cat.name for s in leaf_synsets for cat in s.categories})


def _load_scene_object_locations(scene, category_to_synsets):
    stable_path = os.path.join(DATASET_2026_PATH, "scenes", scene, "json", f"{scene}_stable.json")
    if not os.path.exists(stable_path):
        return {}

    with open(stable_path, "r") as f:
        scene_dict = json.load(f)

    locations = defaultdict(lambda: defaultdict(int))
    for obj_info in scene_dict.get("objects_info", {}).get("init_info", {}).values():
        args = obj_info.get("args", {})
        category = args.get("category")
        if category not in category_to_synsets:
            continue
        in_rooms = args.get("in_rooms") or []
        in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms
        for room in in_rooms:
            for synset in category_to_synsets[category]:
                locations[room][synset] += 1
    return {room: dict(counts) for room, counts in locations.items()}


def _write_task_misc_rooms(activity_name, room_instances):
    if not room_instances:
        raise ValueError("At least one room must be selected before writing B100_task_misc.csv.")

    task_misc_path = os.path.join(DATASET_2026_PATH, "metadata", "B100_task_misc.csv")
    rows = []
    if os.path.exists(task_misc_path):
        with open(task_misc_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter=",", quotechar='"'))

    if not rows:
        rows = [TASK_MISC_HEADER]

    room_cell = "\n".join(room_instances)
    max_task_id = -1
    updated = False
    for row in rows[1:]:
        if len(row) >= 1 and row[0].isdigit():
            max_task_id = max(max_task_id, int(row[0]))
        if len(row) >= 2 and row[1] == activity_name:
            row[:] = [row[0], activity_name, room_cell]
            updated = True

    if not updated:
        rows.append([str(max_task_id + 1), activity_name, room_cell])

    with open(task_misc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerows(rows)
    print(f"Wrote selected rooms for '{activity_name}' to {task_misc_path}")


def _room_type_from_instance(room_instance):
    return "_".join(room_instance.split("_")[:-1])


def _get_room_reasons(room, bddl_room_types, object_locations):
    reasons = []
    if _room_type_from_instance(room) in bddl_room_types:
        reasons.append("Task requires this room type")
    if room in object_locations:
        for synset, count in sorted(object_locations[room].items()):
            suffix = f" x{count}" if count > 1 else ""
            reasons.append(f"{synset}{suffix}")
    return reasons


def _format_room_label(room, bddl_room_types, object_locations):
    lines = [room]
    for reason in _get_room_reasons(room, bddl_room_types, object_locations):
        wrapped = textwrap.wrap(reason, width=28) or [reason]
        lines.append(f"  - {wrapped[0]}")
        lines.extend(f"    {line}" for line in wrapped[1:])
    return "\n".join(lines)


def _format_predicate(predicate):
    return f"{predicate[0]}({', '.join(str(arg) for arg in predicate[1:])})"


def _format_column_items(items, width=24, max_lines=7):
    if not items:
        return "none"

    lines = []
    consumed_items = 0
    for item in items:
        wrapped = textwrap.wrap(str(item), width=width) or [str(item)]
        next_lines = [wrapped[0], *[f"  {line}" for line in wrapped[1:]]]
        if len(lines) + len(next_lines) > max_lines:
            break
        lines.extend(next_lines)
        consumed_items += 1

    remaining = len(items) - consumed_items
    if remaining > 0:
        lines = lines[: max_lines - 1]
        lines.append(f"... (+{remaining} more)")

    return "\n".join(lines)


def _add_metadata_panel(fig, required_rooms, required_objects, goal_conditions):
    info_ax = fig.add_axes([0.03, 0.83, 0.66, 0.11])
    info_ax.set_axis_off()

    columns = [
        ("Required rooms", sorted(required_rooms), 16, 0.00),
        ("Required objects", sorted(required_objects), 22, 0.18),
        ("Goal conditions", goal_conditions, 52, 0.45),
    ]
    for title, items, width, x in columns:
        info_ax.text(x, 0.98, title, va="top", fontsize=10, fontweight="bold", transform=info_ax.transAxes)
        info_ax.text(
            x,
            0.74,
            _format_column_items(items, width=width),
            va="top",
            fontsize=8,
            transform=info_ax.transAxes,
        )


def _select_room_instances_with_gui(
    scene,
    bddl_room_types,
    object_locations,
    floor=0,
    activity_name=None,
    required_objects=None,
    goal_conditions=None,
):
    scene_dir = get_scene_path(scene)
    plan = get_floor_plan_data(scene_dir, floor=floor)
    room_names = sorted(plan["ins_id_to_name"].values())
    name_to_ins_id = {name: ins_id for ins_id, name in plan["ins_id_to_name"].items()}
    bddl_rooms = {room for room in room_names if _room_type_from_instance(room) in bddl_room_types}
    default_rooms = bddl_rooms

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons
    except Exception as e:
        print(f"\nCould not open room picker GUI ({e}); using default rooms: {sorted(default_rooms)}")
        return sorted(default_rooms)

    if plt.get_backend().lower() == "agg":
        print(f"\nMatplotlib is using a non-interactive backend; using default rooms: {sorted(default_rooms)}")
        return sorted(default_rooms)

    selected = set(default_rooms)
    ins = plan["ins"]

    def make_overlay():
        overlay_data = np.zeros((*ins.shape, 4), dtype=np.float32)
        for room in selected:
            ins_id = name_to_ins_id.get(room)
            if ins_id is None:
                continue
            overlay_data[ins == ins_id] = [1.0, 0.95, 0.0, 0.42]
        return overlay_data

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(activity_name or scene, fontsize=14, fontweight="bold", y=0.99)
    _add_metadata_panel(
        fig=fig,
        required_rooms=bddl_room_types,
        required_objects=required_objects or [],
        goal_conditions=goal_conditions or [],
    )

    ax = fig.add_axes([0.03, 0.21, 0.66, 0.57])
    ax.imshow(plan["canvas"], interpolation="nearest")
    overlay = ax.imshow(make_overlay(), interpolation="nearest")

    for ins_id, name in plan["ins_id_to_name"].items():
        ys, xs = (ins == ins_id).nonzero()
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        ax.text(
            cx,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black", alpha=0.82, linewidth=0.5),
        )
        if name in bddl_rooms:
            ax.scatter(
                [cx],
                [cy - 22],
                marker="*",
                s=260,
                c="#d62728",
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
            )
    ax.set_title(f"{scene}: select rooms to load for task sampling")
    ax.set_axis_off()

    selected_ax = fig.add_axes([0.03, 0.05, 0.66, 0.12])
    selected_ax.set_axis_off()

    def format_selected_rooms():
        selected_rooms = sorted(selected)
        if not selected_rooms:
            return "Selected rooms: none"
        wrapped = textwrap.wrap(", ".join(selected_rooms), width=95)
        return "Selected rooms:\n" + "\n".join(wrapped)

    selected_text = selected_ax.text(
        0,
        0.95,
        format_selected_rooms(),
        va="top",
        fontsize=11,
        wrap=True,
    )

    room_labels = []
    label_to_room = {}
    for room in room_names:
        label = _format_room_label(room, bddl_room_types, object_locations)
        room_labels.append(label)
        label_to_room[label] = room

    check_height = min(0.76, max(0.16, 0.085 * len(room_names)))
    check_ax = fig.add_axes([0.71, 0.88 - check_height, 0.28, check_height])
    checks = CheckButtons(check_ax, room_labels, [room in selected for room in room_names])
    check_ax.set_title("Rooms")
    for label in checks.labels:
        label.set_fontsize(10)

    note_ax = fig.add_axes([0.71, 0.89, 0.28, 0.09])
    note_ax.set_axis_off()
    note_ax.text(
        0,
        0.5,
        "Yellow rooms are selected. Defaults come from task-required room types. Red stars mark those task-required room types.",
        va="center",
        wrap=True,
        fontsize=10,
    )

    button_ax = fig.add_axes([0.80, 0.05, 0.12, 0.06])
    done_button = Button(button_ax, "Done")

    def refresh():
        overlay.set_data(make_overlay())
        selected_text.set_text(format_selected_rooms())
        fig.canvas.draw_idle()

    def toggle_room(room):
        if room in selected:
            selected.remove(room)
        else:
            selected.add(room)
        refresh()

    def on_check(label):
        toggle_room(label_to_room[label])

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        y = int(round(event.ydata))
        x = int(round(event.xdata))
        if not (0 <= y < ins.shape[0] and 0 <= x < ins.shape[1]):
            return
        room = plan["ins_id_to_name"].get(int(ins[y, x]))
        if room not in room_names:
            return
        checks.set_active(room_names.index(room))

    checks.on_clicked(on_check)
    fig.canvas.mpl_connect("button_press_event", on_click)
    done_button.on_clicked(lambda _: plt.close(fig))
    plt.show()

    if not selected:
        if default_rooms:
            print(f"\nNo rooms selected; using default rooms: {', '.join(sorted(default_rooms))}")
            return sorted(default_rooms)
        raise ValueError("At least one room must be selected.")

    return sorted(selected)


def autogenerate_task_custom_list(activity_name, use_room_gui=True, floor=0):
    assert os.path.exists(DATASET_2026_PATH), f"2026 dataset not found: {DATASET_2026_PATH}"
    assert os.path.exists(TASK_CUSTOM_LIST_PATH), f"task_custom_lists.json not found: {TASK_CUSTOM_LIST_PATH}"

    kb = get_knowledge_base()
    task = kb.get_task(f"{activity_name}-0")
    conditions = task.parse_base_scope()[0]
    init_conds = conditions.parsed_initial_conditions
    goal_conds = conditions.parsed_goal_conditions
    synsets = set()
    room_types = set()
    task_synsets = set()
    for cond in _iter_predicates(init_conds):
        if len(cond) == 3 and cond[0] == "inroom":
            room_types.add(cond[2])
        for arg in cond[1:]:
            synset = _synset_from_bddl_name(arg, kb)
            if synset is None or "agent" in synset:
                continue
            task_synsets.add(synset)
            synset_obj = kb.get_synset(synset)
            if synset_obj is not None and "sceneObject" in synset_obj.abilities:
                continue
            synsets.add(synset)

    for cond in _iter_predicates(goal_conds):
        for arg in cond[1:]:
            synset = _synset_from_bddl_name(arg, kb)
            if synset is not None and "agent" not in synset:
                task_synsets.add(synset)

    # Prompt for scene — only offer scenes that match the task per the knowledge base
    matched_scene_names = sorted(s.name for s in task.matched_scenes)
    print(f"\nSelect scene for activity '{activity_name}' ({len(matched_scene_names)} matching):")
    for i, s in enumerate(matched_scene_names):
        print(f"  [{i}] {s}")
    while True:
        raw = input("Enter index, name, or custom string: ").strip()
        if raw.isdigit() and int(raw) < len(matched_scene_names):
            scene = matched_scene_names[int(raw)]
            break
        elif raw:
            scene = raw
            break

    scene_object_category_to_synsets = defaultdict(set)
    for synset in sorted(task_synsets):
        synset_obj = kb.get_synset(synset)
        if synset_obj is None or "sceneObject" not in synset_obj.abilities:
            continue
        for category in _get_leaf_categories(synset_obj):
            scene_object_category_to_synsets[category].add(synset)
    object_locations = _load_scene_object_locations(scene, scene_object_category_to_synsets)
    room_instances = None
    if use_room_gui:
        try:
            room_instances = _select_room_instances_with_gui(
                scene=scene,
                bddl_room_types=room_types,
                object_locations=object_locations,
                floor=floor,
                activity_name=activity_name,
                required_objects=sorted(synsets),
                goal_conditions=[_format_predicate(cond) for cond in _iter_predicates(goal_conds)],
            )
        except Exception as e:
            print(f"\nCould not open room picker GUI ({e}); using BDDL room types: {', '.join(sorted(room_types))}")
        if room_instances is not None:
            room_types = {_room_type_from_instance(room) for room in room_instances}
            selected_rooms_msg = ", ".join(room_instances) if room_instances else "none"
            print(f"\nSelected room instances: {selected_rooms_msg}")
            print(f"Saved room types for compatibility: {', '.join(sorted(room_types))}")

    # Prompt for models per synset/category
    models_2025 = get_2025_models_for_task(activity_name)
    whitelist = {}
    for synset in sorted(synsets):
        synset_obj = get_knowledge_base().get_synset(synset)
        if synset_obj is None:
            continue
        whitelist[synset] = {}
        # Non-leaf synsets have no direct categories; walk to leaf descendants
        leaf_synsets = [synset_obj] if synset_obj.is_leaf else [d for d in synset_obj.descendants if d.is_leaf]
        all_cats = [cat for s in leaf_synsets for cat in s.categories]
        for cat in all_cats:
            cat_name = cat.name
            available_models = set(get_all_object_category_models(cat_name))
            available_models = (
                available_models
                if cat_name not in GOOD_MODELS
                else available_models.intersection(GOOD_MODELS[cat_name])
            )
            available_models = sorted(available_models - BAD_CLOTH_MODELS.get(cat_name, set()))
            if not available_models:
                print(f"\n  No models found for category '{cat_name}', skipping.")
                continue
            used_in_2025 = sorted(models_2025.get(synset, {}).get(cat_name, []))
            hint = f"  (used in 2025: {', '.join(used_in_2025)})" if used_in_2025 else "  (not found in 2025 dataset)"
            models = prompt_choice(
                f"Select model(s) for {synset} / {cat_name} ({SYNSET_BASE_URL}/{synset}.html):\n{hint}",
                available_models,
                multi=True,
            )
            whitelist[synset][cat_name] = {m: None for m in models}

    task_entry = {
        activity_name: {
            "room_types": sorted(room_types),
            scene: {
                "whitelist": whitelist,
                "blacklist": {},
            },
        }
    }
    # Load, update, and write back
    with open(TASK_CUSTOM_LIST_PATH, "r") as f:
        existing = json.load(f)

    existing.update(task_entry)

    with open(TASK_CUSTOM_LIST_PATH, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"\nWrote entry for '{activity_name}' to {TASK_CUSTOM_LIST_PATH}")
    if room_instances is not None:
        _write_task_misc_rooms(activity_name, room_instances)


if __name__ == "__main__":
    args = parser.parse_args()
    autogenerate_task_custom_list(args.activity, use_room_gui=not args.no_room_gui, floor=args.floor)
