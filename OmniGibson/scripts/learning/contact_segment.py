import argparse
import json
from collections import defaultdict
from pathlib import Path


ROBOT_GRIPPER_FINGER_SUBSTR = "gripper_finger_link"


def is_robot_link(link):
    return link is not None and ROBOT_GRIPPER_FINGER_SUBSTR in link


def _format_time(second):
    total_seconds = int(second)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _format_link_name(link):
    return link.split("/")[-2]


def format_robot_link_name(link):
    if "left_gripper" in link:
        return "left"
    if "right_gripper" in link:
        return "right"
    return "unknown"


def _canonical_pair(obj1, obj2):
    return tuple(sorted([obj1, obj2]))


def _close_segment(segments, key, start_second, end_second):
    if key[0] == "hand-object":
        _, arm, obj = key
        segments.append(
            {
                "type": "hand-object",
                "arm": arm,
                "object": _format_link_name(obj),
                "start_time": _format_time(start_second),
                "end_time": _format_time(end_second),
            }
        )
    else:
        _, obj1, obj2 = key
        segments.append(
            {
                "type": "object-object",
                "object1": _format_link_name(obj1),
                "object2": _format_link_name(obj2),
                "start_time": _format_time(start_second),
                "end_time": _format_time(end_second),
            }
        )


def _update_segment(active, segments, key, second):
    if key not in active:
        active[key] = (second, second)
        return

    start_second, last_second = active[key]
    if second == last_second + 1:
        active[key] = (start_second, second)
        return

    _close_segment(segments, key, start_second, last_second)
    active[key] = (second, second)


def extract_segments(events, fps):
    grouped = defaultdict(list)
    for ev in events:
        frame = ev.get("frame")
        if frame is None:
            continue
        second = int(frame // fps)
        grouped[second].append(ev)

    contacted_objects = set()
    active = {}
    segments = []

    for second in sorted(grouped.keys()):
        hand_contacts = []
        obj_contacts = []
        for ev in grouped[second]:
            obj1, obj2 = ev.get("contact_obj1"), ev.get("contact_obj2")
            is_obj1_robot = is_robot_link(obj1)
            is_obj2_robot = is_robot_link(obj2)

            if is_obj1_robot and is_obj2_robot:
                continue
            if is_obj1_robot:
                hand_contacts.append((format_robot_link_name(obj1), obj2))
            elif is_obj2_robot:
                hand_contacts.append((format_robot_link_name(obj2), obj1))
            else:
                obj_contacts.append((obj1, obj2))

        for robot, obj in set(hand_contacts):
            contacted_objects.add(obj)
            key = ("hand-object", robot, obj)
            _update_segment(active, segments, key, second)

        for obj1, obj2 in set(obj_contacts):
            if obj1 not in contacted_objects and obj2 not in contacted_objects:
                continue
            obj1, obj2 = _canonical_pair(obj1, obj2)
            key = ("object-object", obj1, obj2)
            _update_segment(active, segments, key, second)

    for key, (start_second, end_second) in list(active.items()):
        _close_segment(segments, key, start_second, end_second)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Extract hand-object and object-object contact segments.")
    parser.add_argument("input_json", type=Path, help="Path to input contact json file.")
    parser.add_argument("output_json", type=Path, help="Path to output segment json file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second.")

    args = parser.parse_args()
    events = json.loads(args.input_json.read_text())
    segments = extract_segments(events, args.fps)
    args.output_json.write_text(json.dumps(segments, indent=2))


if __name__ == "__main__":
    main()
