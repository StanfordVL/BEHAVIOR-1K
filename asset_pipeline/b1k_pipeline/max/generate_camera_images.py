import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import json
import os
import traceback

import pymxs

import b1k_pipeline.utils

rt = pymxs.runtime

OUTPUT_DIR = "camera_images"
OUTPUT_FILENAME = "{0}.png"
JSON_FILENAME = "generate_camera_images.json"
RENDER_PRESET_FILENAME = os.path.abspath(
    str(b1k_pipeline.utils.PIPELINE_ROOT / "render_presets" / "scene_camera.rps")
)


def prepare_scene():
    # Set the render preset
    preset_categories = rt.renderpresets.LoadCategories(RENDER_PRESET_FILENAME)
    assert rt.renderpresets.Load(0, RENDER_PRESET_FILENAME, preset_categories)

    # Unhide everything
    for x in rt.objects:
        x.isHidden = False

    # Hide the upper joints and meta links
    for x in rt.objects:
        match = b1k_pipeline.utils.parse_name(x.name)
        if match is not None and (
            match.group("joint_side") == "upper" or match.group("meta_type") is not None
        ):
            x.isHidden = True

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

    assert rt.viewport.setLayout(rt.Name("layout_1"))


def render_camera_images():
    artifacts_dir = os.path.join(rt.maxFilePath, "artifacts")
    output_dir = os.path.join(artifacts_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the scene once for all cameras.
    prepare_scene()

    # Find all cameras in the scene (identified by their camera superclass, since
    # their names, e.g. "camera-diag", don't follow the object naming convention).
    cameras = [x for x in rt.objects if rt.superClassOf(x) == rt.Camera]
    assert cameras, "No cameras found in the scene."

    rendered_cameras = []
    for camera in cameras:
        camera_id = camera.name
        if camera_id.startswith("camera-"):
            camera_id = camera_id[len("camera-") :]

        # Point the viewport at this camera and render.
        assert rt.viewport.setCamera(camera)

        image_path = os.path.abspath(
            os.path.join(output_dir, OUTPUT_FILENAME.format(camera_id))
        )
        if os.path.exists(image_path):
            os.remove(image_path)
        print("Saving to", image_path)
        assert pymxs.runtime.render(outputFile=image_path)
        rendered_cameras.append(camera_id)

    return rendered_cameras


def main():
    artifacts_dir = os.path.join(rt.maxFilePath, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    success = True
    error_msg = ""
    rendered_cameras = []
    try:
        rendered_cameras = render_camera_images()
    except:
        success = False
        error_msg = traceback.format_exc()

    json_file = os.path.join(artifacts_dir, JSON_FILENAME)
    with open(json_file, "w") as f:
        json.dump(
            {
                "success": success,
                "error_msg": error_msg,
                "cameras": rendered_cameras,
            },
            f,
        )

    if success:
        print("Camera image generation successful!")
    else:
        print("Camera image generation failed.")
        print(error_msg)


if __name__ == "__main__":
    main()
