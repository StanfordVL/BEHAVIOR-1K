import hashlib
import pathlib
import traceback
import shutil
import os
from tqdm import tqdm
import sys
import json
import torch as th
from scipy.spatial.transform import Rotation as R
import glob

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)

gm.HEADLESS = True


def main():
    dataset_name = sys.argv[1]
    input_glob = sys.argv[2]
    DATASET_ROOT = pathlib.Path("/fsx-siro/cgokmen/behavior-data2") / dataset_name
    DATASET_ROOT.mkdir(exist_ok=True)
    ERRORS = DATASET_ROOT / "errors"
    ERRORS.mkdir(exist_ok=True)
    JOBS = DATASET_ROOT / "jobs"
    JOBS.mkdir(exist_ok=True)
    RESTART_EVERY = 16

    ROTATE_EVERYTHING_BY = th.as_tensor(R.from_euler("x", 90, degrees=True).as_quat())

    scenes = sorted([pathlib.Path(x) for x in glob.glob(input_glob, recursive=True)])

    object_mapping = json.loads((DATASET_ROOT / "object_name_mapping.json").read_text())

    # Re-sort jobs differently per run, so that if a previous array job failed it doesn't end up
    # with all the work again.
    scenes.sort(
        key=lambda x: hashlib.md5((str(x) + os.environ.get("SLURM_ARRAY_JOB_ID", default="")).encode()).hexdigest()
    )

    rank = int(sys.argv[3])
    world_size = int(sys.argv[4])

    completed_count = 0
    for scene_input_json in tqdm(scenes[rank::world_size]):
        if og.sim:
            og.clear()
        else:
            og.launch()

        scene_name = scene_input_json.stem.replace(".scene_instance", "")
        scene_root = DATASET_ROOT / "scenes" / scene_name
        scene_root.mkdir(exist_ok=True, parents=True)
        scene_output_json = scene_root / "json" / f"{scene_name}_best.json"
        success_file = scene_root / "import.success"
        if scene_root.exists():
            # Check if we're fully done.
            if success_file.exists():
                continue

            # Otherwise nuke the directory
            shutil.rmtree(scene_root)

        try:
            # Load the scene JSON
            scene_contents = json.loads(scene_input_json.read_text())
            assert scene_contents["translation_origin"] == "asset_local"

            # Load all the objects manually into a scene
            scene = Scene(use_floor_plane=False)
            og.sim.import_scene(scene)

            # Load the scene template. This is not an actual path but is namespaced as stages/iThor/etc and we just want the etc.
            stage_instance = os.path.basename(scene_contents["stage_instance"]["template_name"])
            stage_category, stage_model = object_mapping[stage_instance]
            tmpl = DatasetObject(
                name="stage", category=stage_category, model=stage_model, fixed_base=True, dataset_name=dataset_name
            )
            scene.add_object(tmpl)
            tmpl.set_position_orientation(position=th.zeros(3), orientation=ROTATE_EVERYTHING_BY)

            for i, obj_instance in enumerate(scene_contents["object_instances"]):
                try:
                    # This can similarly be prefixed, so we undo that.
                    template_name = os.path.basename(obj_instance["template_name"])
                    category, model = object_mapping[template_name]
                    pos = th.as_tensor(obj_instance["translation"])
                    orn = th.as_tensor(obj_instance["rotation"])[[1, 2, 3, 0]]
                    scale = th.as_tensor(obj_instance["non_uniform_scale"])

                    if th.any(scale < 0):
                        log.error(f"{category} {model} negative scale detected: {scale}")
                        scale = th.abs(scale)

                    obj = DatasetObject(
                        name=f"{category}_{i}",
                        category=category,
                        model=model,
                        scale=scale,
                        fixed_base=True,
                        dataset_name=dataset_name,
                    )
                except:
                    print("Skipping object", obj_instance)
                    with open(ERRORS / scene_input_json.stem, "a") as f:
                        f.write(str(obj_instance))
                        f.write("\n")
                        f.write(traceback.format_exc())
                        f.write("\n\n")
                    continue
                scene.add_object(obj)
                rotated_pos, rotated_orn = T.pose_transform(th.zeros(3), ROTATE_EVERYTHING_BY, pos, orn)
                obj.set_position_orientation(rotated_pos, rotated_orn)

            # Play the simulator, then save
            og.sim.play()

            # Take a sim step
            og.sim.step()

            og.sim.save(json_paths=[str(scene_output_json)])

            # Load the json, remove the init_info because we don't need it, then save it again
            with open(scene_output_json, "r") as f:
                scene_info = json.load(f)

            scene_info.pop("init_info")

            with open(scene_output_json, "w+") as f:
                json.dump(scene_info, f, indent=4)

            success_file.touch()

            completed_count += 1
        except Exception as e:
            print(f"Error processing {scene_input_json}: {e}")
            # Log the error
            with open(ERRORS / scene_input_json.stem, "w") as f:
                f.write(traceback.format_exc())

        if completed_count >= RESTART_EVERY:
            return

    # If we reach here, we're done. Record the rank success.
    (JOBS / f"{rank}.success").touch()

    og.shutdown()


if __name__ == "__main__":
    main()
