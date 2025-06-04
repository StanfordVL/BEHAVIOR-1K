import h5py
from omnigibson.learning.utils.pcd_utils import process_fused_point_cloud
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Convert RGBD data to PCD format.")
    parser.add_argument("-i", "--input-dir", type=str, help="Path to the input RGBD data directory.")
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), f"Input directory {args.input_dir} does not exist."
    assert args.input_dir.endswith("rgbd"), "Input directory must end with 'rgbd' to process RGBD data."
    output_dir = os.path.join(os.path.dirname(args.input_dir), "pcd")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(args.input_dir)):
        if file_name.endswith(".hdf5"):
            with h5py.File(os.path.join(args.input_dir, file_name), "r") as in_f:
                # create a new hdf5 file to store the point cloud data
                with h5py.File(os.path.join(output_dir, file_name), "w") as out_f:
                    for demo_name in in_f["data"]:
                        demo_data = in_f["data"][demo_name]
                        for key in demo_data:
                            if key == "obs":
                                # copy over all non rgbd keys
                                for obs_key in list(demo_data[key].keys()):
                                    if not (obs_key.endswith("rgb") or obs_key.endswith("depth_linear")):
                                        out_f.create_dataset(
                                            f"data/{demo_name}/{key}/{obs_key}", data=demo_data[key][obs_key][:]
                                        )
                                # process the fused point cloud
                                pcd = process_fused_point_cloud(demo_data[key])
                                out_f.create_dataset(f"data/{demo_name}/{key}/robot_r1::fused_pcd", data=pcd)

                            else:
                                # copy other keys as they are
                                out_f.create_dataset(f"data/{demo_name}/{key}", data=demo_data[key][:])
