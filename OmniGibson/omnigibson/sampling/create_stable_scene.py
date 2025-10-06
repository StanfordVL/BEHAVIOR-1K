import argparse
import os
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_dataset_path
from utils import create_stable_scene_json

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None, help="Scene model to sample tasks in")

gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    if args.scene_model is None:
        # This MUST be specified
        assert os.environ.get(
            "SAMPLING_SCENE_MODEL"
        ), "scene model MUST be specified, either as a command-line arg or as an environment variable!"
        args.scene_model = os.environ["SAMPLING_SCENE_MODEL"]

    # If we want to create a stable scene config, do that now
    default_scene_fpath = os.path.join(
        get_dataset_path("behavior-1k-assets"), f"scenes/{args.scene_model}/json/{args.scene_model}_stable.json"
    )
    if not os.path.exists(default_scene_fpath):
        create_stable_scene_json(scene_model=args.scene_model, record_feedback=True)


if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
