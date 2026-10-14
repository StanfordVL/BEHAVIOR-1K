"""Tests for export_tasks_mujoco's pure functions (no dataset needed)."""

import xml.etree.ElementTree as ET

import numpy as np

from b1k_pipeline import export_tasks_mujoco as tasks


def test_qpos_layout_follows_document_order_through_attachments():
    part = ET.fromstring(
        """<mujoco><worldbody>
             <body name="a_0_root"><freejoint name="a_0_freejoint"/>
               <frame><attach model="cup-x" body="base_link" prefix="a_0_"/></frame></body>
             <body name="w_0_root"><body name="w_0_base_link"><joint name="w_0_j" type="slide"/></body></body>
           </worldbody></mujoco>"""
    )
    scene = ET.fromstring(
        """<mujoco><worldbody>
             <attach model="part-shell" prefix="shell__"/>
             <body name="t_0_root"><attach model="cab-y" body="base_link" prefix="t_0_"/></body>
             <frame name="robot_template"/>
           </worldbody></mujoco>"""
    )

    def resolve(name):
        return {
            "part-shell": ("part", part),
            "cup-x": ("object", [("j_lid", "hinge")]),
            "cab-y": ("object", [("j_door", "hinge"), ("j_drawer", "slide")]),
        }[name]

    layout = tasks.qpos_layout(scene, resolve)
    assert [(n, t, a, s) for n, t, a, s in layout] == [
        ("shell__a_0_freejoint", "free", 0, 7),
        ("shell__a_0_j_lid", "hinge", 7, 1),
        ("shell__w_0_j", "slide", 8, 1),
        ("t_0_j_door", "hinge", 9, 1),
        ("t_0_j_drawer", "slide", 10, 1),
    ]


def test_eval_room_types():
    init = [["inroom", "table.n.02_1", "living_room"], ["ontop", "a", "b"], ["inroom", "floor.n.01_1", "kitchen"]]
    deps = {"house": {"kitchen": ["corridor"]}}
    extra = {"act": {"house": ["garage"]}}
    assert tasks.eval_room_types(init, "house", "act", deps, extra) == ["corridor", "garage", "kitchen", "living_room"]
    assert tasks.eval_room_types(init, "other", "act", deps, extra) == ["kitchen", "living_room"]


def test_room_type():
    assert tasks.room_type("living_room_0") == "living_room"
    assert tasks.room_type("childs_room_12") == "childs_room"


def test_robot_pose_lookup_and_yaw_frame():
    poses = {"R1Pro": [{"position": [1, 2, 0], "orientation": [0, 0, np.sin(0.5), np.cos(0.5)]}]}
    key, pose = tasks.robot_pose(poses, "r1pro")
    assert key == "R1Pro"
    pos, quat, tilt = tasks.yaw_frame(pose)
    np.testing.assert_allclose(pos, [1, 2, 0])
    np.testing.assert_allclose(quat, [np.cos(0.5), 0, 0, np.sin(0.5)], atol=1e-12)
    assert tilt < 1e-9
    key, _ = tasks.robot_pose({"robot": poses["R1Pro"], "R1Pro": []}, "R1Pro")
    assert key == "robot"


def test_shift_frame_moves_in_the_frames_own_coordinates():
    yaw = np.pi / 2  # Facing +y: forward is +y and left is -x.
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    pos, new_quat = tasks.shift_frame([1.0, 2.0, 0.1], quat, [-0.1, 0.04, 0.0])
    np.testing.assert_allclose(pos, [1.0 - 0.04, 2.0 - 0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(new_quat, quat, atol=1e-12)
    _, turned = tasks.shift_frame([0, 0, 0], quat, [0, 0, 10.0])
    new_yaw = yaw + np.radians(10.0)
    np.testing.assert_allclose(turned, [np.cos(new_yaw / 2), 0, 0, np.sin(new_yaw / 2)], atol=1e-12)


def test_load_robot_frame_offsets(tmp_path):
    assert tasks.load_robot_frame_offsets(tmp_path / "missing.json") == {}
    path = tmp_path / "offsets.json"
    path.write_text('{"activities": {"a": {"scene": "s", "frames": {"test_0301": [-0.1, 0.0, 0.0]}}}}')
    assert tasks.load_robot_frame_offsets(path) == {"a": {"scene": "s", "frames": {"test_0301": [-0.1, 0.0, 0.0]}}}
    path.write_text('{"activities": {"a": {"scene": "s", "frames": {"test_0301": [-0.1, 0.0]}}}}')
    try:
        tasks.load_robot_frame_offsets(path)
    except ValueError:
        pass
    else:
        raise AssertionError("a two-element shift should be rejected")


def test_checked_in_robot_frame_offsets_are_small():
    offsets = tasks.load_robot_frame_offsets()
    for activity, entry in offsets.items():
        for label, (dx, dy, dyaw) in entry["frames"].items():
            # check_robot_fit's last-resort limits (docs/mujoco_export.md).
            assert np.hypot(dx, dy) <= 0.8 + 1e-9, (activity, label)
            assert abs(dyaw) <= 90, (activity, label)
