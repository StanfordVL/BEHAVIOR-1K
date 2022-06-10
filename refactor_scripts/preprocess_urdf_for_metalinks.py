from igibson import ig_dataset_path
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime


def pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    space = '\t' if use_tabs else ' ' * 4
    for i, node in enumerate(current):
        pretty_print_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + (space * depth)
        else:
            parent[index - 1].tail = '\n' + (space * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + (space * (depth - 1))


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.
    Examples:
        [0, 1, 2] => "0 1 2"
    Args:
        array (n-array): Array to convert to a string
    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def convert_to_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, np.str_} into an mujoco-xml compatible string.
        Note that an input string / np.str_ results in a no-op action.
    Args:
        inp: Input to convert to string
    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool, np.float32, np.float64, np.int32, np.int64}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def create_joint(name, parent, child, pos=(0, 0, 0), rpy=(0, 0, 0), joint_type="fixed",
                 axis=None, damping=None, friction=None, limits=None):
    """
    Generates XML joint
    Args:
        name (str): Name of this joint
        parent (str or ET.Element): Name of parent link or parent link element itself for this joint
        child (str or ET.Element): Name of child link or child link itself for this joint
        pos (list or tuple or array): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or array): (r,p,y) offset rot values when creating the joint
        joint_type (str): What type of joint to create. Must be one of {fixed, revolute, prismatic}
        axis (None or 3-tuple): If specified, should be (x,y,z) axis corresponding to DOF
        damping (None or float): If specified, should be damping value to apply to joint
        friction (None or float): If specified, should be friction value to apply to joint
        limits (None or 2-tuple): If specified, should be min / max limits to the applied joint
    Returns:
        ET.Element: Generated joint element
    """
    # Create the initial joint
    jnt = ET.Element("joint", name=name, type=joint_type)
    # Create origin subtag
    origin = ET.SubElement(jnt, "origin", attrib={"rpy": convert_to_string(rpy), "xyz": convert_to_string(pos)})
    # Make sure parent and child are both names (str) -- if they're not str already, we assume it's the element ref
    if not isinstance(parent, str):
        parent = parent.get("name")
    if not isinstance(child, str):
        child = child.get("name")
    # Create parent and child subtags
    parent = ET.SubElement(jnt, "parent", link=parent)
    child = ET.SubElement(jnt, "child", link=child)
    # Add additional parameters if specified
    if axis is not None:
        ax = ET.SubElement(jnt, "axis", xyz=convert_to_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = convert_to_string(damping)
    if friction is not None:
        dynamic_params["friction"] = convert_to_string(friction)
    if dynamic_params:
        dp = ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        lim = ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def create_link(name, subelements=None, mass=None, inertia=None):
    """
    Generates XML link element
    Args:
        name (str): Name of this link
        subelements (None or list): If specified, specifies all nested elements that should belong to this link
            (e.g.: visual, collision body elements)
        mass (None or float): If specified, will add an inertial tag with specified mass value
        inertia (None or 6-array): If specified, will add an inertial tag with specified inertia value
            Value should be (ixx, iyy, izz, ixy, ixz, iyz)
    Returns:
        ET.Element: Generated link
    """
    # Create the initial link
    link = ET.Element("link", name=name)
    # Add all subelements if specified
    if subelements is not None:
        for ele in subelements:
            link.append(ele)
    # Add mass subelement if requested
    if mass is not None or inertia is not None:
        inertial = ET.SubElement(link, "inertial")
    if mass is not None:
        ET.SubElement(inertial, "mass", value=convert_to_string(mass))
    if inertia is not None:
        axes = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        inertia_vals = {ax: str(i) for ax, i in zip(axes, inertia)}
        ET.SubElement(inertial, "inertia", **inertia_vals)

    # Return this element
    return link


def create_metalink(root_element, metalink_name, parent_link_name="base_link", pos=(0,0,0), rpy=(0,0,0)):
    # Create joint
    jnt = create_joint(
        name=f"{metalink_name}_joint",
        parent=parent_link_name,
        child=f"{metalink_name}_link",
        pos=pos,
        rpy=rpy,
        joint_type="fixed",
    )
    # Create child link
    link = create_link(
        name=f"{metalink_name}_link",
        mass=0.0001,
        inertia=[0.00001, 0.00001, 0.00001, 0, 0, 0],
    )

    # Add to root element
    root_element.append(jnt)
    root_element.append(link)


def generate_urdf_from_xmltree(root_element, name, dirpath, unique_urdf=False):
    """
    Generates a URDF file corresponding to @xmltree at @dirpath with name @name.urdf.
    Args:
        root_element (ET.Element): Element tree that compose the URDF
        name (str): Name of this file (name assigned to robot tag)
        dirpath (str): Absolute path to the location / filename for the generated URDF
        unique_urdf (bool): Whether to use a unique identifier when naming urdf (uses current datetime)
    Returns:
        str: Path to newly created urdf (fpath/<name>.urdf)
    """
    # Write to fpath, making sure the directory exists (if not, create it)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Get file
    date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
    fname = f"{name}_{date}.urdf" if unique_urdf else f"{name}.urdf"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, 'w') as f:
        # Write top level header line first
        f.write('<?xml version="1.0" ?>\n')
        # Convert xml to string form and write to file
        pretty_print_xml(current=root_element)
        xml_str = ET.tostring(root_element, encoding="unicode")
        f.write(xml_str)

    # Return path to file
    return fpath


def update_obj_urdf_with_metalinks(obj_category, obj_model):
    # Check if filepath exists
    model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
    urdf_path = f"{model_root_path}/{obj_model}.urdf"

    # Load urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Load metadata
    metadata_fpath = f"{model_root_path}/misc/metadata.json"
    with open(metadata_fpath, "r") as f:
        metadata = json.load(f)

    # Pop meta inks
    if "links" in metadata:
        meta_links = metadata.pop("links")
        print("meta_links:", meta_links)
        for link_name, attrs in meta_links.items():
            pos = attrs.get("xyz", None)
            rpy = attrs.get("rpy", None)
            pos = [0, 0, 0] if pos is None else pos
            rpy = [0, 0, 0] if rpy is None else rpy

            # TODO: Don't hardcode parent to be base_link!

            # Create metalink
            create_metalink(
                root_element=root,
                metalink_name=link_name,
                parent_link_name="base_link",
                pos=pos,
                rpy=rpy,
            )

    # Grab all elements

    # Export this URDF
    generate_urdf_from_xmltree(
        root_element=root,
        name=obj_model,
        dirpath=model_root_path,
        unique_urdf=False,
    )
