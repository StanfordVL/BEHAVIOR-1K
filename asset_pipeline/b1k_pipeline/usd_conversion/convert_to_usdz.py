import glob
import os
import pathlib
import shutil
import subprocess
import traceback
import tqdm

from omnigibson.utils.asset_utils import encrypt_file, decrypt_file
from pxr import Usd, UsdUtils
from dask.distributed import LocalCluster, as_completed


def convert_to_usdz(encrypted_usd_path_str: str):
    try:
        # print("Decrypting", encrypted_usd_path_str)
        usd_path_str = encrypted_usd_path_str.replace(".encrypted", "")
        decrypt_file(encrypted_usd_path_str, decrypted_filename=usd_path_str)

        usd_path = pathlib.Path(usd_path_str)
        model_dir = usd_path.parent.parent
        moved_usd_path = model_dir / usd_path.name
        usdz_path = moved_usd_path.with_suffix(".usdz")
        encrypted_usdz_path = usd_path.with_suffix(".usdz.encrypted")

        # Here we perform a trick where we first copy the USD file to a new location and then
        # update its asset paths to be relative to the new location. This allows the usdz archiver
        # to use relative paths instead of absolute ones.
        # print("Updating asset paths in USD")
        shutil.copy(usd_path, moved_usd_path)
        stage = Usd.Stage.Open(str(moved_usd_path))

        # Now update the asset paths
        materials_dir = (model_dir / "material").resolve()
        def _update_path(asset_path):
            # Compute the path that this path refers to
            asset_full_path = (usd_path.parent / asset_path).resolve()

            # Check if that's a subpath of the materials dir. If not, return unchanged
            try:
                asset_full_path.relative_to(materials_dir)
            except ValueError:
                return asset_path
            
            # Return it relative to the new USD location
            new_path = "./" + str(asset_full_path.relative_to(model_dir.resolve()))
            # print("Renaming", asset_path, "to", new_path)
            return new_path

        UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), _update_path)
        stage.Save()
        del stage

        # Convert USD to USDZ
        # print(f"Converting {moved_usd_path} to {usdz_path}")
        UsdUtils.CreateNewUsdzPackage(str(moved_usd_path), str(usdz_path))

        # Remove the MDL files from inside the USDZ. We want to load these from the
        # OmniGibson installation instead of from the USDZ directly, since they have
        # dependencies.
        # print("Removing MDL files from", usdz_path)
        path_7za = pathlib.Path(__file__).parents[2] / "7zzs"
        subprocess.run([str(path_7za), "d", str(usdz_path), "**/*.mdl"], check=True, capture_output=True)

        # print("Encrypting", usdz_path)
        encrypt_file(usdz_path, encrypted_filename=encrypted_usdz_path)

        os.remove(usdz_path)
        os.remove(moved_usd_path)
        os.remove(usd_path)

        return None
    except:
        return traceback.format_exc()

def main():
    encrypted_files = list(glob.glob("/scr/og-docker-data/datasets/og_dataset_3_7_0rc23/**/*.encrypted.usd", recursive=True))
    # convert_to_usdz(encrypted_files[0])

    print("Launching local Dask cluster...")
    cluster = LocalCluster()
    dask_client = cluster.get_client()

    print("Submitting jobs...")
    futures = dask_client.map(convert_to_usdz, encrypted_files)

    print("Waiting for jobs to finish...")
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        err = future.result()
        if err:
            print(f"Error processing {future}: {err}")

if __name__ == "__main__":
    main()