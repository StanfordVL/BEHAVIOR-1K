"""Building blocks for MuJoCo scene MJCFs, shared by the scene and task exporters.

* `add_object` attaches a catalog object archive as `N_root` (with a free joint
  `N_freejoint` if the object is loose), scaled by a frame if needed.
* `StructureInliner` copies structure objects (walls, floors, ...) into a
  base-scene part: every element gets the same name that attaching the object
  with prefix `N_` would give it, so code that binds objects by name can't tell
  the two apart.
* `part_for_object` decides which base-scene part an object belongs to.
* `add_robot_frame` and `add_keyframe` write the task-specific additions.

See docs/mujoco_export.md for the conventions.
"""

import copy
import io
import json
import os
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from b1k_pipeline import mujoco_common

# Part that holds what OmniGibson loads whatever the rooms: shell structure and
# doors, plus the global cameras.
SHELL_PART = "shell"

# Part of the non-shell objects that have no rooms. OmniGibson doesn't load them
# when it loads only some rooms, so only full scenes attach this part.
NO_ROOM_PART = "no_room"

# Separator between room names in a multi-room part's name. Part names end up in
# asset ids, which allow [a-z0-9_.-].
ROOM_SEPARATOR = "-"

# Attributes that hold names of other elements, by element tag.
_REFERENCE_ATTRS = {
    "geom": ("mesh", "material", "hfield"),
    "site": ("material",),
    "material": ("texture",),
    "layer": ("texture",),
    "skin": ("material",),
}

# Elements whose names are renamed per instance (N_{name}).
_INSTANCE_NAMED_TAGS = ("body", "geom", "site", "joint", "light", "camera", "frame")


def add_object(parent_xml, name, model_asset, pos, quat_wxyz, fixed, scale=None):
    """Attach a catalog object as the body `N_root`.

    Args:
        parent_xml: Element to add the body to (usually <worldbody>).
        name: The object's name N.
        model_asset: Name of the <model> asset holding the object's archive.
        pos: World position of the object's base link.
        quat_wxyz: World orientation of the object's base link.
        fixed: Whether the object is static (no joint) or loose (free joint).
        scale: Optional per-axis scale, applied in the base link's frame.

    Returns:
        The root body element.
    """
    root_xml = ET.SubElement(
        parent_xml,
        "body",
        {
            "name": mujoco_common.root_body_name(name),
            "pos": mujoco_common.fmt(pos),
            "quat": mujoco_common.fmt(quat_wxyz),
        },
    )
    if not fixed:
        ET.SubElement(root_xml, "freejoint", {"name": mujoco_common.freejoint_name(name)})
    attach_parent = root_xml
    if scale is not None and not mujoco_common.is_unit_scale(scale):
        attach_parent = ET.SubElement(
            root_xml, "frame", {"scale": mujoco_common.fmt(scale), **mujoco_common.SCALE_ATTRS}
        )
    ET.SubElement(
        attach_parent,
        "attach",
        {"model": model_asset, "body": "base_link", "prefix": mujoco_common.attach_prefix(name)},
    )
    return root_xml


def part_for_object(category, rooms):
    """The base-scene part of an object, from OmniGibson's loading rule.

    Shell categories always load and go into the shell part. Anything else
    loads when any of its rooms does, so it goes into the part of its exact room
    set; a task scene with a subset of rooms attaches every part whose room set
    intersects it. Anything else without rooms only loads with every room, and
    goes into the `no_room` part.
    """
    if category in mujoco_common.SHELL_CATEGORIES:
        return SHELL_PART
    rooms = sorted(set(rooms or []))
    if not rooms:
        return NO_ROOM_PART
    return ROOM_SEPARATOR.join(rooms)


def part_rooms(part):
    """Rooms of a part (none for the shell and the no-room part)."""
    return [] if part in (SHELL_PART, NO_ROOM_PART) else part.split(ROOM_SEPARATOR)


def parts_for_rooms(parts, rooms):
    """The parts a scene restricted to `rooms` attaches (None means all rooms)."""
    if rooms is None:
        return list(parts)
    rooms = set(rooms)
    return [p for p in parts if p == SHELL_PART or rooms & set(part_rooms(p))]


def add_robot_frame(worldbody_xml, name, pos, quat_wxyz):
    """Add a robot frame. Robot frames must come after every body."""
    return ET.SubElement(
        worldbody_xml,
        "frame",
        {"name": name, "pos": mujoco_common.fmt(pos), "quat": mujoco_common.fmt(quat_wxyz)},
    )


def add_keyframe(root_xml, name, qpos):
    keyframe_xml = root_xml.find("keyframe")
    if keyframe_xml is None:
        keyframe_xml = ET.SubElement(root_xml, "keyframe")
    return ET.SubElement(keyframe_xml, "key", {"name": name, "qpos": mujoco_common.fmt(qpos)})


def _resolve_defaults(archive_root):
    """Default classes of an archive: class -> tag -> attributes.

    Object archives declare their classes one level deep under the top-level
    <default> (see export_objects_mujoco.py).
    """
    classes = {}
    top = archive_root.find("default")
    if top is None:
        return classes
    for class_xml in top.findall("default"):
        classes[class_xml.attrib["class"]] = {child.tag: dict(child.attrib) for child in class_xml}
    for child in top:
        if child.tag != "default":
            raise ValueError("Top-level default attributes aren't supported in inlined archives")
    return classes


def _expand_classes(element_xml, classes):
    """Replace `class` attributes with the attributes they stand for."""
    for el in element_xml.iter():
        class_name = el.attrib.pop("class", None)
        if class_name is None:
            continue
        defaults = classes[class_name].get(el.tag, {})
        merged = dict(defaults)
        merged.update(el.attrib)
        el.attrib.clear()
        el.attrib.update(merged)
        if "childclass" in el.attrib:
            raise ValueError("childclass isn't supported in inlined archives")


class StructureInliner:
    """Copies structure objects' archives into a base-scene part.

    Assets are copied once per (category, model) into `structure/{cat}_{model}/`
    under the part's directory and renamed `{part}__{cat}-{model}-{orig}`, so
    that parts composed with an empty attach prefix never collide. Each instance
    becomes a static `N_root` body whose contents carry the names that
    attaching the object with prefix `N_` would give them; the scale goes into
    a frame, which MuJoCo applies to the geometry at compile time. Default
    classes are expanded into explicit attributes for the same reason.

    Attributes:
        files: Archive name -> bytes, relative to the part's directory.
    """

    def __init__(self, part, asset_xml, custom_xml, texture_max_size=1024):
        self.part = part
        self.asset_xml = asset_xml
        self.custom_xml = custom_xml
        self.texture_max_size = texture_max_size
        self.files = {}
        # Contact exclusions of the inlined instances, as (body1, body2).
        self.excludes = []
        # (category, model) -> (base link element with renamed assets, metadata
        # text, contact exclusions)
        self._models = {}

    def _asset_name(self, category, model, orig):
        return f"{self.part}__{category}-{model}-{orig}"

    def _cap_texture(self, data):
        image = Image.open(io.BytesIO(data))
        if self.texture_max_size and max(image.size) > self.texture_max_size:
            ratio = self.texture_max_size / max(image.size)
            size = (max(1, round(image.size[0] * ratio)), max(1, round(image.size[1] * ratio)))
            fmt = image.format or "PNG"
            image = image.resize(size, Image.LANCZOS)
            out = io.BytesIO()
            image.save(out, format=fmt)
            return out.getvalue()
        return data

    def _load_model(self, category, model, archive):
        """Copy an archive's assets into the part, once per (category, model)."""
        key = (category, model)
        if key in self._models:
            return self._models[key]
        with zipfile.ZipFile(archive) as zf:
            archive_root = ET.fromstring(zf.read(mujoco_common.ARCHIVE_ROOT_FILE))
            compiler = archive_root.find("compiler")
            meshdir = (compiler.attrib.get("meshdir", "") if compiler is not None else "") or "."
            texturedir = (compiler.attrib.get("texturedir", "") if compiler is not None else "") or "."
            if compiler is not None and compiler.attrib.get("angle", "degree") != "radian":
                raise ValueError(f"{archive}: inlined archives must use radians")
            classes = _resolve_defaults(archive_root)
            renames = {}
            out_dir = f"structure/{category}_{model}"
            archive_assets = archive_root.find("asset")
            for asset_el in [] if archive_assets is None else list(archive_assets):
                new_el = copy.deepcopy(asset_el)
                if "name" in new_el.attrib:
                    renames[(new_el.tag, new_el.attrib["name"])] = self._asset_name(
                        category, model, new_el.attrib["name"]
                    )
                    new_el.attrib["name"] = renames[(new_el.tag, new_el.attrib["name"])]
                if "file" in new_el.attrib:
                    base = texturedir if new_el.tag == "texture" else meshdir
                    src = posixpath.normpath(posixpath.join(base, new_el.attrib["file"]))
                    data = zf.read(src)
                    if new_el.tag == "texture":
                        data = self._cap_texture(data)
                    dst = f"{out_dir}/{src}"
                    self.files[dst] = data
                    new_el.attrib["file"] = dst
                for child in new_el:
                    if child.tag == "layer":
                        child.attrib["texture"] = self._asset_name(category, model, child.attrib["texture"])
                if new_el.tag == "material" and "texture" in new_el.attrib:
                    new_el.attrib["texture"] = self._asset_name(category, model, new_el.attrib["texture"])
                _expand_classes(new_el, classes)
                self.asset_xml.append(new_el)

            bodies = archive_root.find("worldbody").findall("body")
            if len(bodies) != 1 or bodies[0].attrib.get("name") != "base_link":
                raise ValueError(f"{archive}: expected a single base_link body")
            base_link = copy.deepcopy(bodies[0])
            _expand_classes(base_link, classes)
            for el in base_link.iter():
                for attr in _REFERENCE_ATTRS.get(el.tag, ()):
                    if attr in el.attrib:
                        tag = "mesh" if attr == "mesh" else attr
                        el.attrib[attr] = renames[(tag, el.attrib[attr])]

            excludes = []
            contact = archive_root.find("contact")
            if contact is not None:
                for el in contact:
                    if el.tag != "exclude":
                        raise ValueError(f"{archive}: unsupported contact element <{el.tag}>")
                    excludes.append((el.attrib["body1"], el.attrib["body2"]))

            metadata_text = None
            custom = archive_root.find("custom")
            if custom is not None:
                for text in custom.findall("text"):
                    if text.attrib.get("name") == mujoco_common.METADATA_TEXT:
                        metadata_text = text.attrib["data"]
        self._models[key] = (base_link, metadata_text, excludes)
        return self._models[key]

    def add(self, parent_xml, name, category, model, archive, pos, quat_wxyz, scale=None):
        """Inline one structure instance as the static body `N_root`.

        Args:
            parent_xml: Element to add the body to (usually <worldbody>).
            name: The object's name N.
            category: Its category.
            model: Its model id.
            archive: Path or file object of the object's MJZ.
            pos: World position of the base link.
            quat_wxyz: World orientation of the base link.
            scale: Optional per-axis scale in the base link's frame.

        Returns:
            The root body element.
        """
        base_link, metadata_text, excludes = self._load_model(category, model, archive)
        mujoco_common.check_scalable(name, json.loads(metadata_text) if metadata_text else None, scale)
        root_xml = ET.SubElement(
            parent_xml,
            "body",
            {
                "name": mujoco_common.root_body_name(name),
                "pos": mujoco_common.fmt(pos),
                "quat": mujoco_common.fmt(quat_wxyz),
            },
        )
        parent = root_xml
        if scale is not None and not mujoco_common.is_unit_scale(scale):
            parent = ET.SubElement(
                root_xml, "frame", {"scale": mujoco_common.fmt(scale), **mujoco_common.SCALE_ATTRS}
            )
        instance = copy.deepcopy(base_link)
        prefix = mujoco_common.attach_prefix(name)
        for el in instance.iter():
            if el.tag in _INSTANCE_NAMED_TAGS and "name" in el.attrib:
                el.attrib["name"] = prefix + el.attrib["name"]
        parent.append(instance)
        self.excludes += [(prefix + body1, prefix + body2) for body1, body2 in excludes]
        if metadata_text is not None:
            ET.SubElement(
                self.custom_xml,
                "text",
                {"name": prefix + mujoco_common.METADATA_TEXT, "data": metadata_text},
            )
        return root_xml


def relative_path(from_dir, to_path):
    """POSIX relative path between two dataset-relative locations."""
    return posixpath.relpath(to_path, from_dir)


def read_metadata_text(root_xml, name):
    """Parse the b1k_metadata text of object N from a compiled-from MJCF tree."""
    for text in root_xml.iter("text"):
        if text.attrib.get("name") == mujoco_common.attach_prefix(name) + mujoco_common.METADATA_TEXT:
            return json.loads(text.attrib["data"])
    return None
