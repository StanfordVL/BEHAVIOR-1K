"""Tests for the pure-Python parts of the MuJoCo export (no dataset needed)."""

import io
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from b1k_pipeline import export_objects_mujoco
from b1k_pipeline import mujoco_common
from b1k_pipeline import mujoco_scene_utils


def _object_root(joint_axis="1 0 0", child_quat=None, geom_type="mesh", geom_quat=None):
    root = ET.fromstring(
        """
        <mujoco model="toy">
          <compiler angle="radian" meshdir="." texturedir="."/>
          <default>
            <default class="visual"><geom group="2" contype="0" conaffinity="0" mass="0"/></default>
            <default class="collision"><geom group="0"/></default>
          </default>
          <asset>
            <mesh name="base_link_visual" file="visual/base_link.obj"/>
            <texture name="base_link_albedo" type="2d" file="textures/base_link_albedo.png"/>
            <material name="base_link_material"><layer texture="base_link_albedo" role="rgb"/></material>
          </asset>
          <worldbody>
            <body name="base_link">
              <geom name="base_link_visual" class="visual" type="mesh" mesh="base_link_visual"
                    material="base_link_material"/>
              <body name="link_1" pos="0.1 0 0">
                <joint name="j_link_1" type="slide"/>
                <geom name="link_1_collision_0" class="collision" type="mesh" mesh="base_link_visual"/>
              </body>
            </body>
          </worldbody>
          <custom><text name="b1k_metadata" data='{"format": "b1k_metadata/1", "model": "toy"}'/></custom>
        </mujoco>
        """
    )
    link = root.find(".//body[@name='link_1']")
    link.find("joint").attrib["axis"] = joint_axis
    if child_quat is not None:
        link.attrib["quat"] = child_quat
    geom = link.find("geom")
    geom.attrib["type"] = geom_type
    if geom_type != "mesh":
        geom.attrib.pop("mesh")
        geom.attrib["size"] = "0.1 0.1 0.1"
    if geom_quat is not None:
        geom.attrib["quat"] = geom_quat
    return root


class TestAnisotropicScalability:
    def test_aligned_slide_passes(self):
        export_objects_mujoco.assert_anisotropically_scalable(_object_root())

    def test_axis_rotated_by_child_body_still_aligned(self):
        # 90 degrees about z maps x onto y: still aligned.
        quat = f"{np.cos(np.pi / 4)} 0 0 {np.sin(np.pi / 4)}"
        export_objects_mujoco.assert_anisotropically_scalable(_object_root(child_quat=quat))

    def test_diagonal_joint_fails(self):
        with pytest.raises(ValueError, match="j_link_1"):
            export_objects_mujoco.assert_anisotropically_scalable(_object_root(joint_axis="1 1 0"))

    def test_joint_rotated_off_axis_by_body_fails(self):
        quat = f"{np.cos(np.pi / 8)} 0 0 {np.sin(np.pi / 8)}"  # 45 degrees about z
        with pytest.raises(ValueError, match="not aligned"):
            export_objects_mujoco.assert_anisotropically_scalable(_object_root(child_quat=quat))

    def test_small_misalignment_within_tolerance_passes(self):
        angle = np.deg2rad(1.0)
        quat = f"{np.cos(angle / 2)} 0 0 {np.sin(angle / 2)}"
        export_objects_mujoco.assert_anisotropically_scalable(_object_root(child_quat=quat))

    def test_aligned_box_passes_and_rotated_box_fails(self):
        export_objects_mujoco.assert_anisotropically_scalable(_object_root(geom_type="box"))
        quat = f"{np.cos(np.pi / 8)} 0 0 {np.sin(np.pi / 8)}"
        with pytest.raises(ValueError, match="box geom"):
            export_objects_mujoco.assert_anisotropically_scalable(_object_root(geom_type="box", geom_quat=quat))

    def test_sphere_and_cylinder_fail(self):
        for geom_type in ("sphere", "cylinder", "capsule"):
            with pytest.raises(ValueError, match="stretched"):
                export_objects_mujoco.assert_anisotropically_scalable(_object_root(geom_type=geom_type))


class TestOpenableMetadata:
    def test_annotated_joints_have_direction_one(self):
        # metadata.json stores [[id, name, direction]], but OmniGibson's USD
        # conversion keeps {id: name}, so every direction is 1.
        meta = {"openable_joint_ids": [[0, "j_a", -1], [2, "j_c"]], "openable_both_sides": True}
        result = export_objects_mujoco._openable_metadata(meta, ["j_a", "j_b", "j_c"])
        assert result["joints"] == [{"joint": "j_a", "direction": 1}, {"joint": "j_c", "direction": 1}]
        assert result["both_sides"] is True
        assert result["openable_joint_ids"] == meta["openable_joint_ids"]

    def test_fallback_to_every_joint(self):
        result = export_objects_mujoco._openable_metadata({}, ["j_a", "j_b"])
        assert [j["joint"] for j in result["joints"]] == ["j_a", "j_b"]
        assert all(j["direction"] == 1 for j in result["joints"])
        assert result["both_sides"] is False

    def test_unknown_joint_fails(self):
        with pytest.raises(AssertionError):
            export_objects_mujoco._openable_metadata({"openable_joint_ids": [[0, "nope"]]}, ["j_a"])


class TestParts:
    def test_shell_categories(self):
        for category in ("walls", "ceilings", "door", "sliding_door", "background", "roof", "fence"):
            assert mujoco_scene_utils.part_for_object(category, ["kitchen_0"]) == "shell"

    def test_ground_goes_with_its_room(self):
        assert mujoco_scene_utils.part_for_object("floors", ["kitchen_0"]) == "kitchen_0"

    def test_multi_room_part(self):
        part = mujoco_scene_utils.part_for_object("countertop", ["living_room_0", "kitchen_0"])
        assert part == "kitchen_0-living_room_0"
        assert mujoco_scene_utils.part_rooms(part) == ["kitchen_0", "living_room_0"]

    def test_non_shell_without_rooms_goes_to_no_room(self):
        assert mujoco_scene_utils.part_for_object("floors", []) == "no_room"
        assert mujoco_scene_utils.part_rooms("no_room") == []
        assert mujoco_scene_utils.parts_for_rooms(["shell", "no_room", "kitchen_0"], ["kitchen_0"]) == [
            "shell",
            "kitchen_0",
        ]

    def test_parts_for_rooms(self):
        parts = ["shell", "kitchen_0", "living_room_0", "kitchen_0-living_room_0", "bedroom_0"]
        assert mujoco_scene_utils.parts_for_rooms(parts, ["kitchen_0"]) == [
            "shell",
            "kitchen_0",
            "kitchen_0-living_room_0",
        ]
        assert mujoco_scene_utils.parts_for_rooms(parts, None) == parts


class TestAddObject:
    def test_loose_scaled(self):
        world = ET.Element("worldbody")
        mujoco_scene_utils.add_object(world, "cup_abc_0", "cup-abc", [1, 2, 3], [1, 0, 0, 0], False, [1, 2, 1])
        body = world.find("body")
        assert body.attrib["name"] == "cup_abc_0_root"
        assert body.find("freejoint").attrib["name"] == "cup_abc_0_freejoint"
        frame = body.find("frame")
        assert frame.attrib == {"scale": "1 2 1", "scaling": "geometry", "massscale": "1"}
        attach = frame.find("attach")
        assert attach.attrib == {"model": "cup-abc", "body": "base_link", "prefix": "cup_abc_0_"}

    def test_fixed_unit_scale(self):
        world = ET.Element("worldbody")
        mujoco_scene_utils.add_object(world, "t_0", "t-x", [0, 0, 0], [1, 0, 0, 0], True, [1, 1, 1])
        body = world.find("body")
        assert body.find("freejoint") is None
        assert body.find("frame") is None
        assert body.find("attach").attrib["prefix"] == "t_0_"


def _toy_archive():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.xml", ET.tostring(_object_root()))
        zf.writestr("visual/base_link.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
        from PIL import Image

        image = io.BytesIO()
        Image.new("RGB", (2048, 1024)).save(image, format="PNG")
        zf.writestr("textures/base_link_albedo.png", image.getvalue())
    buf.seek(0)
    return buf


class TestStructureInliner:
    def test_reference_integrity_and_names(self):
        part = ET.Element("mujoco")
        asset = ET.SubElement(part, "asset")
        world = ET.SubElement(part, "worldbody")
        custom = ET.SubElement(part, "custom")
        inliner = mujoco_scene_utils.StructureInliner("kitchen_0", asset, custom, texture_max_size=1024)
        for name in ("floors_abc_0", "floors_abc_1"):
            inliner.add(world, name, "floors", "abc", _toy_archive(), [0, 0, 0], [1, 0, 0, 0], [2, 1, 1])

        asset_names = {(el.tag, el.attrib["name"]) for el in asset}
        # Assets are copied once and renamed per part and model.
        assert asset_names == {
            ("mesh", "kitchen_0__floors-abc-base_link_visual"),
            ("texture", "kitchen_0__floors-abc-base_link_albedo"),
            ("material", "kitchen_0__floors-abc-base_link_material"),
        }
        layer = asset.find("material/layer")
        assert layer.attrib["texture"] == "kitchen_0__floors-abc-base_link_albedo"
        for el in world.iter():
            for attr in ("mesh", "material"):
                if attr in el.attrib:
                    assert (attr, el.attrib[attr]) in asset_names
            assert "class" not in el.attrib
        # Names match what attach with prefix N_ would produce.
        bodies = [b.attrib["name"] for b in world.iter("body")]
        assert bodies[:3] == ["floors_abc_0_root", "floors_abc_0_base_link", "floors_abc_0_link_1"]
        assert world.find(".//joint").attrib["name"] == "floors_abc_0_j_link_1"
        # Default classes are expanded.
        visual = world.find(".//geom[@name='floors_abc_0_base_link_visual']")
        assert visual.attrib["group"] == "2" and visual.attrib["contype"] == "0"
        collision = world.find(".//geom[@name='floors_abc_0_link_1_collision_0']")
        assert collision.attrib["group"] == "0"
        # The scale is a frame.
        assert world.find("body/frame").attrib["scale"] == "2 1 1"
        # Files are copied once, and textures are capped.
        assert set(inliner.files) == {
            "structure/floors_abc/visual/base_link.obj",
            "structure/floors_abc/textures/base_link_albedo.png",
        }
        from PIL import Image

        texture = Image.open(io.BytesIO(inliner.files["structure/floors_abc/textures/base_link_albedo.png"]))
        assert texture.size == (1024, 512)
        # Metadata texts are prefixed.
        texts = [t.attrib["name"] for t in custom]
        assert texts == ["floors_abc_0_b1k_metadata", "floors_abc_1_b1k_metadata"]
        assert json.loads(custom[0].attrib["data"])["model"] == "toy"


def test_common_helpers():
    assert mujoco_common.object_name("bottom_cabinet-qacthv-0") == "bottom_cabinet_qacthv_0"
    assert mujoco_common.is_unit_scale([1, 1, 1 + 1e-5])
    assert not mujoco_common.is_unit_scale([1, 1, 1.01])
    np.testing.assert_allclose(mujoco_common.xyzw_to_wxyz([1, 2, 3, 4]), [4, 1, 2, 3])


def test_check_scalable():
    bad = {"anisotropic_scaling": {"ok": False, "problems": ["joint j axis 1 1 0 is not aligned"]}}
    good = {"anisotropic_scaling": {"ok": True, "problems": []}}
    mujoco_common.check_scalable("x", bad, [1.2, 1.2, 1.2])  # Uniform scales always work.
    mujoco_common.check_scalable("x", bad, [1, 1.00001, 1])  # So do unit scales.
    mujoco_common.check_scalable("x", good, [1.2, 0.8, 1.0])
    with pytest.raises(ValueError, match="isn't anisotropically scalable"):
        mujoco_common.check_scalable("x", bad, [1.2, 0.8, 1.0])


def test_problems_listed_without_raising():
    problems = export_objects_mujoco.anisotropic_scaling_problems(_object_root(joint_axis="1 1 0"))
    assert len(problems) == 1 and "j_link_1" in problems[0]
    assert export_objects_mujoco.anisotropic_scaling_problems(_object_root()) == []


def test_inliner_prefixes_contact_excludes():
    root = _object_root()
    ET.SubElement(ET.SubElement(root, "contact"), "exclude", {"body1": "base_link", "body2": "link_1"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.xml", ET.tostring(root))
        zf.writestr("visual/base_link.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        from PIL import Image

        image = io.BytesIO()
        Image.new("RGB", (4, 4)).save(image, format="PNG")
        zf.writestr("textures/base_link_albedo.png", image.getvalue())
    buf.seek(0)
    part = ET.Element("mujoco")
    inliner = mujoco_scene_utils.StructureInliner(
        "p", ET.SubElement(part, "asset"), ET.SubElement(part, "custom")
    )
    inliner.add(ET.SubElement(part, "worldbody"), "w_0", "walls", "abc", buf, [0, 0, 0], [1, 0, 0, 0])
    assert inliner.excludes == [("w_0_base_link", "w_0_link_1")]


def test_decide_joint_fixes():
    from b1k_pipeline import find_mujoco_joint_fixes

    urdf = ET.fromstring(
        """
        <robot name="toy">
          <link name="base_link"/><link name="lid"/><link name="lever"/><link name="knob"/>
          <joint name="j_lid" type="revolute"><parent link="base_link"/><child link="lid"/>
            <origin xyz="0 0 0" rpy="0 0 0.05"/><axis xyz="0 1 0"/><limit lower="0" upper="1"/></joint>
          <joint name="j_lever" type="revolute"><parent link="base_link"/><child link="lever"/>
            <axis xyz="0.7071 -0.7071 0"/><limit lower="0" upper="0.5"/></joint>
          <joint name="j_knob" type="continuous"><parent link="base_link"/><child link="knob"/>
            <origin rpy="0 0 1.5707963"/><axis xyz="1 0 0"/></joint>
        </robot>
        """
    )
    fixes = find_mujoco_joint_fixes.decide_fixes(urdf, {"openable_joint_ids": [[0, "j_lid"]]}, 10.0)
    # The lid's axis is y rotated 0.05 rad (2.9 degrees) about z: snapped.
    assert fixes["j_lid"]["action"] == "snap" and fixes["j_lid"]["openable"]
    assert fixes["j_lid"]["nearest_base_axis"] == "+y"
    # The lever is 45 degrees off: fixed.
    assert fixes["j_lever"]["action"] == "fix"
    # The knob's x axis, rotated 90 degrees about z, is aligned with y.
    assert "j_knob" not in fixes
