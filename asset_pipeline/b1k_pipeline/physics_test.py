"""
This script is used to load a newly-exported scene and check it for physics stability.

To run this script, you need to have a valid version of iGibson 2.0 installed in your environment.
"""

import sys
import pybullet as p
from b1k_pipeline.utils import PIPELINE_ROOT

import igibson
igibson.ig_dataset_path = str(PIPELINE_ROOT / "artifacts/aggregate")

from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene


OUTPUT_FILENAME = "physics_test.json"
SUCCESS_FILENAME = "physics_test.success"



def main():
    target = sys.argv[1]
    scene_name = target.split("/")[1]
    output_dir = PIPELINE_ROOT / "cad" / target / "artifacts"

    s = Simulator(mode="headless", use_pb_gui=True)

    try:
        scene = InteractiveIndoorScene(
            scene_name,
            build_graph=True,
        )
        s.import_scene(scene)

        # Save the pose of everything
        initial_poses = {}
        for b in range(p.getNumBodies()):
            initial_poses{b} = p.getPositionAndOrientation(b)

        for _ in range(1000):
            p.stepSimulation()

        # Recheck the pose of everything
        errors = []
        for b in range(p.getNumBodies()):
            if b not in scene.objects_by_id:
                continue

            obj_name = scene.objects_by_id[b].name
            orig_pos, orig_orn = initial_poses[b]
            pos, orn = p.getPositionAndOrientation(b)
            diff_pos = np.linalg.norm(np.array(pos) - np.array(orig_pos))
            diff_orn = (R.from_quat(orn) * R.from_quat(old-1).inv()).magnitude()

            if diff_pos > POS_THRESHOLD:
                errors.append(f"{obj_name} moved by {diff_pos} meters.")
            if diff_rot > ORN_THRESHOLD:
                errors.append(f"{obj_name} rotated by {np.rad2deg(diff_orn)} degrees.")


        success = not errors
        filename = output_dir / OUTPUT_FILENAME
        with open(filename, "w") as f:
            json.dump({"success": success}, f, indent=4)

        if success:
            with open(output_dir / SUCCESS_FILENAME, "w") as f:
                pass

    finally:
        s.disconnect()


if __name__ == "__main__":
    main()