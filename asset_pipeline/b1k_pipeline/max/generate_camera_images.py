import json
import os
import re
from collections import Counter

import pymxs

rt = pymxs.runtime

PATTERN = re.compile(
    r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$"
)
OUTPUT_DIR = "camera_images"
OUTPUT_FILENAME = "{0}.png"
SUCCESS_FILENAME = "generate_camera_images.success"
RENDER_PRESET_FILENAME = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "render_presets", "scene_camera.rps"
    )
)


def main():
    success = False

    artifacts_dir = os.path.join(rt.maxFilePath, "artifacts")
    output_dir = os.path.join(artifacts_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Set the render preset
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Unhide everything
    # for x in rt.objects:
    #     x.isHidden = False

    # Hide the upper joints
    for x in rt.objects:
        match = PATTERN.fullmatch(x.name)
        if match is not None and (
            match.group("joint_side") == "upper" or "light" in match.group("category")
        ):
            x.isHidden = True

    # Prepare the viewport
    camera_id = "diag"  # TODO: make this into a loop later.
    (camera,) = [x for x in rt.objects if x.name == f"camera-{camera_id}"]

    assert rt.viewport.setLayout(rt.Name("layout_1"))
    assert rt.viewport.setCamera(camera)

    # Set the exposure control
    ec = rt.VRay_Exposure_Control()
    ec.mode = 106
    ec.ev = -2
    # rt.SceneExposureControl.exposurecontrol = rt.Automatic_Exposure_Control()
    # rt.lightLevel = 3.0
    # rt.lightTintColor = rt.Color(255, 227, 196)

    # Set the lights to be fixed for now
    for light in rt.lights:
        light.isHidden = True
        # light.normalizeColor = 1
        # light.multiplier = 15000

    # Render
    image_path = os.path.abspath(
        os.path.join(output_dir, OUTPUT_FILENAME.format(camera_id))
    )
    if os.path.exists(image_path):
        os.remove(image_path)
    print("Saving to", image_path)
    assert pymxs.runtime.render(outputFile=image_path)

    with open(os.path.join(artifacts_dir, SUCCESS_FILENAME), "w") as f:
        pass


if __name__ == "__main__":
    main()
