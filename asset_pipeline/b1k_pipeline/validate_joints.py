import xml.etree.ElementTree as ET
import glob
import numpy as np
import tqdm

from b1k_pipeline.utils import ParallelZipFS


def check_urdf(objects_fs, fn):
    # Get the joint axes in the object
    with objects_fs.open(fn) as f:
        tree = ET.parse(f)
    joints = list(tree.findall('.//joint[@type="prismatic"]'))

    # If there are none, return.
    if len(joints) == 0:
        return True

    # Otherwise load the axes into a numpy array
    axes = np.array(
        [[float(y) for y in x.find("axis").attrib["xyz"].split()] for x in joints]
    )
    assert axes.shape[1] == 3, fn

    # Compare the axes with the canonical axes to see if any pair is close
    good_options = list(np.eye(3))
    is_close_to_parallel = lambda x, y: np.isclose(np.abs(np.dot(x, y)), 1, atol=1e-2)
    is_close_to_any_option = lambda x: any(
        is_close_to_parallel(x, option) for option in good_options
    )
    return all(is_close_to_any_option(x) for x in axes)


def main():
    with ParallelZipFS("objects.zip") as objects_fs:
        files = [x.path for x in objects_fs.glob("objects/*/*/urdf/*.urdf")]
        bad = sorted(fn for fn in tqdm.tqdm(files) if not check_urdf(objects_fs, fn))
        print("\n".join(bad))


if __name__ == "__main__":
    main()
