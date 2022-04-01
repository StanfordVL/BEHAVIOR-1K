import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy


def string_to_array(string, num_type):
    """
    Converts a array string in mujoco xml to np.array.
    Examples:
        "0 1 2" => [0, 1, 2]
    Args:
        string (str): String to convert to an array
    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([num_type(x) for x in string.split(" ")])


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


def split_obj_file(obj_fpath):
    """
    Splits obj file at @obj_fpath into individual obj files
    """
    with open(obj_fpath, "r") as f:
        lines = f.readlines()

    sub_obj_lines = []
    obj_idx = -1
    max_idx = 0
    offset_idx = 0
    for line in lines:
        # Every time we come across the geom header "o "..., we add to a new file
        if line[0] == "o":
            # Create a new list of lines we'll add to
            obj_idx += 1
            offset_idx = max_idx
            sub_obj_lines.append([])
        elif line[0] == "f":
            # Update biggest offset
            idxs = string_to_array(line[2:-2], num_type=int)
            max_idx = max(idxs.tolist() + [max_idx])
            # Need to potentially offset face indices
            idxs -= offset_idx
            # Update line
            line = f"f {array_to_string(idxs)} \n"
        sub_obj_lines[obj_idx].append(line)

    # Make sure idx is not -1
    assert obj_idx >= 0, "No valid geoms found in obj file!"

    # Procedurally create new files in the same folder as obj_fpath
    out_fpath = "/".join(obj_fpath.split("/")[:-1])
    out_fname_root = obj_fpath.split("/")[-1].split(".")[0]

    for i, new_lines in enumerate(sub_obj_lines):
        # Open a new file and write the lines to it
        with open(f"{out_fpath}/{out_fname_root}_{i}.obj", "w+") as f_out:
            f_out.writelines(new_lines)

    # We return the number of splits we had
    return obj_idx + 1


def split_objs_in_urdf(urdf_fpath, name_suffix="split", mesh_fpath_offset="."):
    """
    Splits the obj reference in its urdf
    """
    tree = ET.parse(urdf_fpath)
    root = tree.getroot()
    urdf_dir = "/".join(urdf_fpath.split("/")[:-1])
    out_fname_root = urdf_fpath.split("/")[-1].split(".")[0]

    def recursively_find_collision_meshes(ele):
        # Finds all collision meshes starting at @ele
        cols = []
        for child in ele:
            if child.tag == "collision":
                # If the nested geom type is a mesh, add this to our running list along with its parent node
                if child.find("./geometry/mesh") is not None:
                    cols.append((child, ele))
            elif child.tag == "visual":
                # There will be no collision mesh internally here so we simply pass
                continue
            else:
                # Recurisvely look through all children of the child
                cols += recursively_find_collision_meshes(ele=child)

        return cols

    # Iterate over the tree and find all collision entries
    col_elements = recursively_find_collision_meshes(ele=root)

    # For each collision element and its parent, we remove the original one and create a set of new ones with their
    # filename references changed
    for col, parent in col_elements:
        # Don't change the original
        col_copy = deepcopy(col)
        # Delete the original
        parent.remove(col)
        # Create new objs first so we know how many we need to create in the URDF
        obj_fpath = col_copy.find("./geometry/mesh").attrib["filename"]
        n_new_objs = split_obj_file(obj_fpath=f"{urdf_dir}/{mesh_fpath_offset}/{obj_fpath}")
        # Create the new objs in the URDF
        for i in range(n_new_objs):
            # Copy collision again
            col_copy_copy = deepcopy(col_copy)
            # Modify the filename
            fname = col_copy_copy.find("./geometry/mesh").attrib["filename"]
            fname = fname.split(".obj")[0] + f"_{i}.obj"
            col_copy_copy.find("./geometry/mesh").attrib["filename"] = fname
            # Add to parent
            parent.append(col_copy_copy)

    # Finally, write this to a new file
    urdf_out_path = f"{urdf_dir}/{out_fname_root}_{name_suffix}.urdf"
    tree.write(urdf_out_path)

    # Return the urdf it wrote to
    return urdf_out_path


if __name__ == "__main__":
    # Try it out
    URDF = "/cvgl2/u/jdwong/tmp/test_walls/walls.urdf"
    split_objs_in_urdf(urdf_fpath=URDF)
