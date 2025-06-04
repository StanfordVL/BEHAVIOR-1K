import h5py
from omnigibson.learning.utils.pcd_utils import process_fused_point_cloud


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Convert RGBD data to PCD format.")
    parser.add_argument("-f", "--files", type=str, help="Path to the input RGBD data file.")
    args = parser.parse_args()

    assert os.path.isfile(args.files), f"Input file {args.files} does not exist."
    assert args.files.endswith(".hdf5"), "Input file must be an HDF5 file to process RGBD data."
    assert os.path.basename(os.path.dirname(args.files)) == "rgbd", "Input file must be in a directory named 'rgbd'."
    output_dir = os.path.join(os.path.dirname(os.path.dirname(args.files)), "pcd")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(args.files, "r") as in_f:
        # extract the file name without the directory and extension
        file_name = os.path.basename(args.files)
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
