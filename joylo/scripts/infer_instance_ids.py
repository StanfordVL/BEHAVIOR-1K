from omnigibson.envs import DataPlaybackWrapper
import os
from omnigibson.macros import gm
import argparse
import sys
import json
from gello.utils.qa_utils import *
import inspect
from datetime import datetime
from omnigibson.utils.python_utils import h5py_group_to_torch, recursively_convert_to_torch
from omnigibson.utils.asset_utils import get_dataset_path

RUN_QA = True

gm.ENABLE_TRANSITION_RULES = False  # This flag is needed to run data playback wrapper
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


def infer_instance_ids_from_hdf5_file(hdf_input_path):
    dir_path = os.path.dirname(hdf_input_path)
    fname = os.path.basename(hdf_input_path)
    instance_ids_fpath = os.path.join(dir_path, fname.replace(".hdf5", "_instance_ids_mapping.json"))

    # Define output -- random temp file (won't be used for anything)
    random_str = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    hdf_output_path = os.path.join(og.tempdir, f"{random_str}.hdf5")

    # Create the environment
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb"],
        robot_sensor_config=None,
        external_sensors_config=None,
        exclude_sensor_names=["zed"],
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=None,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    # Load the instances data
    instance_init_states = dict()
    instances_dir = os.path.join(get_dataset_path("behavior-1k-assets"), "scenes", env.task.scene_name, "json", f"{env.task.scene_name}_task_{env.task.activity_name}_instances")
    for fname in os.listdir(instances_dir):
        # name should be <SCENE>_task_<ACTIVITY>_0_<INSTANCE_ID>_template-tro_state.json
        instance_id = int(fname.split("_0_")[-1].split("_")[0])
        with open(os.path.join(instances_dir, fname), "r") as f:
            instance_init_state = recursively_convert_to_torch(json.load(f))
            instance_init_states[instance_id] = instance_init_state

    instance_ids_mapping = dict()
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        data_grp = env.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        # Skip early if found malformed data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            continue

        env.scene.restore(env.scene_file, update_initial_file=True)
        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # Try to infer the ID, matching the kinematic poses for the given objects
        matched_instance_id = None
        for instance_id, instance_init_state in instance_init_states.items():
            matched = True
            for name, bddl_inst in env.task.object_scope.items():
                if bddl_inst.is_system or not bddl_inst.exists or bddl_inst.fixed_base or "agent" in name:
                    continue
                pos = instance_init_state[name]["root_link"]["pos"]
                if not th.allclose(pos, bddl_inst.get_position_orientation()[0], atol=1e-2):
                    matched = False
                    break
            if matched:
                matched_instance_id = instance_id
                break

        assert matched_instance_id is not None, f"Could not find a matched instance_id for episode_id={episode_id}"
        instance_ids_mapping[episode_id] = matched_instance_id

    # Save metrics
    with open(instance_ids_fpath, "w+") as f:
        json.dump(instance_ids_mapping, f, indent=4)

    # Always clear the environment to free resources
    og.clear()

    print(f"Successfully matched instance_ids from {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 files to infer instance_ids")
    parser.add_argument("--dir", help="Directory containing HDF5 files to infer instance_ids")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to infer instance_ids")

    args = parser.parse_args()

    if args.dir and os.path.isdir(args.dir):
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                     if f.lower().endswith('.hdf5') and os.path.isfile(os.path.join(args.dir, f))]

        if not hdf_files:
            print(f"No HDF5 files found in directory: {args.dir}")
        else:
            print(f"Found {len(hdf_files)} HDF5 files to infer instance_ids")
    elif args.files:
        # Process individual files specified
        hdf_files = args.files
    else:
        parser.print_help()
        print("\nError: Either --dir or --files must be specified", file=sys.stderr)
        return

    # Process each file
    for hdf_file in hdf_files:
        if not os.path.exists(hdf_file):
            print(f"Error: File {hdf_file} does not exist", file=sys.stderr)
            continue

        infer_instance_ids_from_hdf5_file(hdf_file)

    og.shutdown()


if __name__ == "__main__":
    main()
