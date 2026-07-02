"""
Interactive keyboard-teleop tool to debug the ToggledOn "button" behavior (marker not turning
green / not toggling when a finger touches it), mimicking JoyLo control with the keyboard.

Run WITH a display (NOT headless), e.g.:
    # Mimic a task (loads only its rooms, not the whole house/yard -> much faster to render):
    python -m omnigibson.examples.robots.toggle_button_debug --robot r1pro --task thawing_frozen_food
    # Or a raw scene with specific rooms:
    python -m omnigibson.examples.robots.toggle_button_debug --task "" --scene house_single_floor --rooms kitchen

Drive the robot with the printed keyboard bindings and bring a finger onto a button. The console
prints, for every ToggledOn object each ~10 steps:
    value (is it toggled on), marker color (green=on / red=off), and the min finger->marker distance.

Diagnostic keys:
    Y : force set_value(True) on ALL toggle objects  -> if the marker turns GREEN, color/render works,
        so any failure to green-on-touch is a DETECTION problem (surface touch not reaching the marker).
        If it stays RED even here, it's a COLOR/RENDER problem.
    H : force set_value(False) on all toggle objects.
    G : print the current finger + marker world positions and marker radius for the nearest button.
"""

import argparse
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import ToggledOn
from omnigibson.utils.ui_utils import KeyboardRobotController

gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False


def min_finger_distance(robot, marker):
    """Min distance from any of the robot's finger links to the marker center."""
    mc = marker.get_position_orientation()[0]
    best = float("inf")
    for links in robot.finger_links.values():
        for link in links:
            d = th.norm(link.get_position_orientation()[0] - mc).item()
            best = min(best, d)
    return best


def resolve_task(task_name):
    """Look up (scene_model) for a challenge task from the installed 2026 task-instance metadata."""
    import os
    import yaml
    from omnigibson.macros import gm as _gm

    path = os.path.join(_gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "available_tasks.yaml")
    try:
        tasks = yaml.safe_load(open(path))
        return tasks[task_name][0]["scene_model"]
    except Exception as e:
        print(f"[toggle-debug] Could not resolve task '{task_name}' ({e}).")
        return None


def resolve_rooms(task_name, scene_model):
    """
    Task-relevant room TYPES to partially load (so we don't render the whole house + yard).
    Mimics JoyLo: parse the task's `inroom` conditions + room dependencies. Falls back to
    ['kitchen'] (where microwave/countertop buttons live) if the JoyLo helpers aren't importable.
    """
    try:
        from gello.utils.og_teleop_utils import get_task_relevant_room_types, augment_rooms

        rooms = get_task_relevant_room_types(activity_name=task_name)
        rooms = augment_rooms(rooms, scene_model, task_name)
        if rooms:
            return rooms
    except Exception as e:
        print(f"[toggle-debug] Room auto-detect unavailable ({e}); defaulting to ['kitchen'].")
    return ["kitchen"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="r1pro", help="Robot model (e.g. r1pro, fetch)")
    parser.add_argument(
        "--task",
        default="thawing_frozen_food",
        help="Challenge task to mimic (looks up its scene). Use '' to load a raw --scene instead.",
    )
    parser.add_argument("--scene", default="house_single_floor", help="Scene model (used only when --task is empty)")
    parser.add_argument(
        "--rooms",
        default="",
        help="Comma-separated room types to load (partial load = fast). "
        "Empty = auto-detect the task's rooms. Use 'all' to load the entire scene (slow, incl. yard).",
    )
    parser.add_argument(
        "--target",
        default="microwave",
        help="Substring of the button object name to park next to (e.g. microwave, oven, electric_switch)",
    )
    args = parser.parse_args()

    # Resolve scene from the task (mimic the real task) unless --task is empty.
    scene_model = resolve_task(args.task) if args.task else None
    if scene_model is None:
        scene_model = args.scene
    print(f"[toggle-debug] task='{args.task}' -> scene='{scene_model}'")

    # Partial room loading: only render the task-relevant rooms (NOT the whole house/yard) for speed.
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": scene_model}
    if args.rooms.strip().lower() == "all":
        print("[toggle-debug] Loading ALL rooms (slow).")
    else:
        rooms = (
            [r.strip() for r in args.rooms.split(",") if r.strip()]
            if args.rooms
            else (resolve_rooms(args.task, scene_model) if args.task else None)
        )
        if rooms:
            scene_cfg["load_room_types"] = rooms
            print(f"[toggle-debug] Partial load — rooms: {rooms}")

    cfg = {
        "scene": scene_cfg,
        "robots": [
            {
                "model": args.robot,
                "obs_modalities": ["rgb"],
                "action_normalize": True,
                "grasping_mode": "assisted",
            }
        ],
    }
    env = og.Environment(configs=cfg)
    robot = env.robots[0][0]

    # r1pro defaults to JointController on the arms (move one joint at a time -- painful for reaching
    # a button). Override ONLY the arms to InverseKinematicsController so the arrow keys move the HAND
    # in cartesian space; every other component keeps its default (the config is merged over defaults).
    try:
        ik_cfg = {
            f"arm_{a}": {"name": "InverseKinematicsController"}
            for a in robot.arm_names
            if f"arm_{a}" in robot.controller_order
        }
        if ik_cfg:
            robot.reload_controllers(controller_config=ik_cfg)
            env.scene.update_initial_file()
            print(f"[toggle-debug] Arms set to IK (arrow keys move the hand): {list(ik_cfg)}")
    except Exception as e:
        print(f"[toggle-debug] Could not switch arms to IK ({e}); using default joint controllers.")

    # Collect all ToggledOn objects (the "buttons")
    toggle_objs = [o for o in env.scene.objects if ToggledOn in o.states]
    print(f"\n[toggle-debug] Found {len(toggle_objs)} ToggledOn objects: {[o.name for o in toggle_objs]}\n")

    # Park the robot next to the target toggle object (default: the microwave) so you don't drive far.
    if toggle_objs:
        matches = [o for o in toggle_objs if args.target in o.name]
        target = matches[0] if matches else toggle_objs[0]
        print(f"[toggle-debug] Parking next to '{target.name}' (matched --target='{args.target}')")
        tpos = target.get_position_orientation()[0]
        robot.set_position_orientation(position=tpos + th.tensor([0.8, 0.0, 0.0], dtype=th.float32))
        og.sim.viewer_camera.set_position_orientation(
            position=tpos + th.tensor([1.8, 1.8, 1.4], dtype=th.float32),
        )
    robot.reset()
    og.sim.step()

    # Keyboard teleop
    action_generator = KeyboardRobotController(robot=robot)

    def set_all(val):
        for o in toggle_objs:
            ok = o.states[ToggledOn].set_value(val)
            c = o.states[ToggledOn].marker.color.tolist() if o.states[ToggledOn].marker else None
            print(
                f"[toggle-debug] set_value({val}) on {o.name} -> ok={ok} marker.color={c} "
                f"(green={ToggledOn.COLOR_ON.tolist()})"
            )

    def print_positions():
        for o in toggle_objs:
            st = o.states[ToggledOn]
            if st.marker is None:
                print(f"[toggle-debug] {o.name}: NO MARKER")
                continue
            mc = st.marker.get_position_orientation()[0]
            r = th.min(st.marker.extent * st.marker.scale).item()
            d = min_finger_distance(robot, st.marker)
            print(
                f"[toggle-debug] {o.name}: marker_center={[round(x,3) for x in mc.tolist()]} "
                f"radius={r:.4f} min_finger_dist={d:.4f} value={st.get_value()}"
            )

    # NOTE: keys chosen to avoid the teleop bindings (t=gripper, i/j/k/l, arrows, p/;/n/b/o/u/v/c, 1-6, m).
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Y,
        description="FORCE toggle ON all buttons (test color/render)",
        callback_fn=lambda: set_all(True),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.H,
        description="FORCE toggle OFF all buttons",
        callback_fn=lambda: set_all(False),
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.G,
        description="Print marker/finger positions + radius",
        callback_fn=print_positions,
    )

    action_generator.print_keyboard_teleop_info()
    print("\n[toggle-debug] Drive a finger onto a button. Watch value/color below.")
    print("[toggle-debug] Y=force ON (isolates render vs detection)  H=force OFF  G=print radius/dist  ESC=quit\n")

    step = 0
    while True:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1
        if step % 10 == 0 and toggle_objs:
            parts = []
            for o in toggle_objs:
                st = o.states[ToggledOn]
                if st.marker is None:
                    continue
                d = min_finger_distance(robot, st.marker)
                on = st.get_value()
                col = "GREEN" if st.marker.color.tolist() == ToggledOn.COLOR_ON.tolist() else "red"
                parts.append(f"{o.name}[on={on},{col},d={d:.3f}]")
            print("[toggle-debug] " + " ".join(parts))


if __name__ == "__main__":
    main()
