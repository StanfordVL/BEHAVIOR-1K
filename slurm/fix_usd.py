from concurrent.futures import ProcessPoolExecutor, as_completed
import os, glob
import traceback
from pxr import Usd, UsdUtils
from tqdm import tqdm
import shutil

def fix_usd_path(path):
    try:
        # make a backup
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)

        stage = Usd.Stage.Open(path)
        def _update_path(asset_path):
            if "fsx-siro" not in asset_path or "material" not in asset_path:
                return asset_path
            fixed_path = os.path.join("..", "material", os.path.basename(asset_path))
            # print(asset_path, "->", fixed_path)
            return fixed_path

        UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), _update_path)
        stage.GetRootLayer().Save()
    except:
        raise ValueError(traceback.format_exc())

def main():
    with ProcessPoolExecutor() as executor:
        # Map the fix_usd_path function to all USD files
        usds = list(glob.glob("/fsx-siro/cgokmen/behavior-data2/*/objects/*/*/usd/*.usd"))
        assert usds

        futures = {executor.submit(fix_usd_path, usd): usd for usd in usds}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(futures[future], "had error:", e)

if __name__ == "__main__":
    main()