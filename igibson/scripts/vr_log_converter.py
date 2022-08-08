import argparse
import pathlib

import h5py
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR state saving and replay demo")
    parser.add_argument("--input", required=True, type=str, help="Input file to convert to time ordered hdf5")
    parser.add_argument("--output", required=True, type=str, help="Output file to write converted data")
    args = parser.parse_args()

    assert h5py.is_hdf5(args.input), "Error, input is not an HDF5 file"
    in_hf5 = h5py.File(args.input, "r")

    assert pathlib.Path(args.output).parent.exists(), "Error, output directory does not exist"
    out_hf5 = h5py.File(args.output, "w")

    keys_to_write = list()

    def map_to_time_ordered(name, item):
        if type(item) == h5py.Dataset:
            keys_to_write.append(name)

    in_hf5.visititems(map_to_time_ordered)

    frame_count = in_hf5["frame_data"].shape[0]

    progress_bar = tqdm(total=int(len(keys_to_write) * frame_count))
    for frame in range(frame_count):
        for key in keys_to_write:
            out_hf5[str(frame) + "/" + key] = in_hf5[key][frame]
            progress_bar.update()
