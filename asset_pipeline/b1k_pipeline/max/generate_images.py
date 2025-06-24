import json
import os
import re
from collections import Counter

import pymxs

rt = pymxs.runtime

PATTERN = re.compile(
    r"^(?P<bad>B-)?(?P<randomization_disabled>F-)?(?P<loose>L-)?(?P<category>[a-z_]+)-(?P<model_id>[A-Za-z0-9_]+)-(?P<instance_id>[0-9]+)(?:-(?P<link_name>[a-z0-9_]+))?(?:-(?P<parent_link_name>[A-Za-z0-9_]+)-(?P<joint_type>[RP])-(?P<joint_side>lower|upper))?(?:-L(?P<light_id>[0-9]+))?(?:-M(?P<meta_type>[a-z]+)_(?P<meta_id>[0-9]+))?$"
)
OUTPUT_FILENAME = "top.png"
SUCCESS_FILENAME = "generate_images.success"

HIDE_CATEGORIES = ["ceilings", "square_light", "room_light", "downlight"]
DONT_ZOOM_CATEGORIES = ["ceilings", "floors", "background", "lawn"]


def main():
    success = False

    output_dir = os.path.join(rt.maxFilePath, "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Set the render preset
        preset_path = (
            r"C:\Users\Cem\Documents\3ds Max 2022\renderpresets\imgrender_cpu.rps"
        )
        preset_categories = rt.renderpresets.LoadCategories(preset_path)
        assert rt.renderpresets.Load(0, preset_path, preset_categories)

        # Unhide everything
        # for x in rt.objects:
        #     x.isHidden = False

        # Hide the ceilings
        for x in rt.objects:
            match = PATTERN.fullmatch(x.name)
            if match is not None and match.group("category").lower() in HIDE_CATEGORIES:
                x.isHidden = True

        # Go into top view
        assert rt.viewport.setType(rt.Name("view_top"))

        # Center the camera on non-floor, non-ceiling, non-background, non-lawn objects
        selection = []
        for x in rt.objects:
            if x.isHidden:
                continue

            match = PATTERN.fullmatch(x.name)
            if (
                match is not None
                and match.group("category").lower() not in DONT_ZOOM_CATEGORIES
            ):
                selection.append(x)
        rt.select(selection)
        # assert pymxs.runtime.execute('max tool zoomextents')
        rt.select([])

        # Set the exposure control
        ec = rt.VRay_Exposure_Control()
        ec.mode = 106
        ec.ev = -2
        rt.SceneExposureControl.exposurecontrol = ec

        # Set the lights to be fixed for now
        for light in rt.lights:
            light.isHidden = True
        #     light.normalizeColor = 1
        #     light.multiplier = 15000

        # Render
        image_path = os.path.abspath(os.path.join(output_dir, OUTPUT_FILENAME))
        if os.path.exists(image_path):
            os.remove(image_path)
        print("Saving to", image_path)
        assert pymxs.runtime.render(outputFile=image_path)

        success = True
    except:
        pass

    # if success:
    #     with open(os.path.join(output_dir, SUCCESS_FILENAME), "w") as f:
    #         pass


if __name__ == "__main__":
    main()
