import json
import os
import re
import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from collections import Counter, defaultdict

from fs.zipfs import ZipFS
import numpy as np
import pymxs
from scipy.spatial.transform import Rotation as R

import b1k_pipeline.utils


rt = pymxs.runtime

SIDES = {
    # Facing the object
    # "front": R.from_euler("xy", [-np.pi / 2, np.pi / 2]),
    # Top down
    # "top": R.from_euler("Z", -np.pi / 2),
    # Diagonal
    "diagonal1": R.from_euler("xy", [-np.pi / 2, np.pi / 2])
    * R.from_euler("XYZ", [0, -np.pi / 6, np.pi / 6]),
    "diagonal2": (
        R.from_euler("xy", [-np.pi / 2, np.pi / 2])
        * R.from_euler("XYZ", [0, -np.pi / 6, np.pi / 6])
    ).inv(),
    # "top", "left", "right", "front", "back"
}
JSON_FILENAME = "generate_max_object_images.json"


# The camera somehow has different axes.
def main():
    output_dir = os.path.join(rt.maxFilePath, "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    with ZipFS(os.path.join(output_dir, "max_object_images.zip"), write=True) as zipfs:
        # Set the render preset
        preset_path = r"C:\Users\Cem\Documents\3ds Max 2022\renderpresets\imgrender.rps"
        preset_categories = rt.renderpresets.LoadCategories(preset_path)
        assert rt.renderpresets.Load(0, preset_path, preset_categories)

        # Set the exposure control
        ec = rt.VRay_Exposure_Control()
        ec.mode = 106
        ec.ev = -2
        rt.SceneExposureControl.exposurecontrol = ec

        # Group the objects
        failed = []
        groups = defaultdict(list)

        for obj in rt.objects:
            if rt.classOf(obj) != rt.Editable_Poly:
                continue

            match = b1k_pipeline.utils.parse_name(obj.name)
            if match is None:
                continue

            if (
                int(match.group("instance_id")) != 0
                or match.group("bad")
                or match.group("meta_type")
            ):
                continue

            groups[
                (
                    match.group("category"),
                    match.group("model_id"),
                    match.group("instance_id"),
                )
            ].append(obj)

        for group_key, group in groups.items():
            try:
                # Find the base link obj
                matches = [b1k_pipeline.utils.parse_name(obj.name) for obj in group]
                bases = [
                    obj
                    for obj, match in zip(group, matches)
                    if match is not None
                    and (
                        not match.group("link_name")
                        or match.group("link_name") == "base_link"
                    )
                ]
                (base,) = bases

                # Hide everything
                for x in rt.objects:
                    x.isHidden = True

                # Set the lights to be fixed for now
                for light in rt.lights:
                    light.isHidden = True
                #     light.normalizeColor = 1
                #     light.multiplier = 15000

                # Unhide the objects in the group
                for x in group:
                    x.isHidden = False

                # Get the object rotation
                obj_quat = base.rotation
                obj_rot = R.from_quat([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w])

                # Render
                for side_name, side_rot in SIDES.items():
                    # Switch the side
                    assert rt.viewport.setType(rt.Name("view_top"))

                    # Rotate the camera to match the object.
                    camera_rot = side_rot * obj_rot.inv()
                    rt.viewport.rotate(
                        rt.Quat(*[float(x) for x in camera_rot.as_quat()])
                    )

                    # Set the zoom extent
                    rt.select([])
                    rt.select(group)
                    assert pymxs.runtime.execute("max tool zoomextents")
                    rt.forceCompleteRedraw()
                    rt.windows.processPostedMessages()

                    # Render the image
                    obj_name = "-".join(group_key)
                    output_filename = f"{obj_name}--{side_name}.png"
                    image_path = zipfs.getsyspath(output_filename)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    print("Saving to", image_path)
                    assert pymxs.runtime.render(outputFile=image_path)
            except:
                failed.append(group_key)

    success = len(failed) == 0
    with open(os.path.join(output_dir, JSON_FILENAME), "w") as f:
        json.dump({"success": success, "failed_objs": sorted(failed)}, f)


if __name__ == "__main__":
    main()
