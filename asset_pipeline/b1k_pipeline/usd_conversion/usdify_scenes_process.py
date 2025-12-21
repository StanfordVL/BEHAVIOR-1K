"""
Script to import scene and objects
"""

import pathlib
import sys
import time

from omnigibson.macros import gm

# Set some macros. Is this kosher?
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.USE_ENCRYPTED_ASSETS = True

import omnigibson as og
from omnigibson.utils.asset_conversion_utils import convert_scene_urdf_to_json
from b1k_pipeline.usd_conversion.make_maps import generate_maps_for_current_scene


if __name__ == "__main__":
    og.launch()

    dataset_root = sys.argv[1]
    gm.DATASET_PATH = str(dataset_root)

    urdf_path = pathlib.Path(dataset_root) / sys.argv[2]
    scene_basename = urdf_path.stem
    json_path = urdf_path.parent.parent / "json" / f"{scene_basename}.json"
    success_path = urdf_path.with_suffix(".success")

    # Convert URDF to USD
    convert_scene_urdf_to_json(urdf=str(urdf_path), json_path=str(json_path))

    # Generate the maps
    if urdf_path.name.endswith("_best.urdf"):
        # If the URDF is the best version, we generate maps
        print("Starting map generation")
        map_start = time.time()
        save_path = urdf_path.parent.parent / "layout"
        generate_maps_for_current_scene(str(save_path))
        map_end = time.time()
        print("Generated maps in ", map_end - map_start, "seconds")

    with open(success_path, "w") as f:
        pass

    # Clear the sim
    og.sim.clear()

    og.shutdown()
