import glob
import sys
import tqdm

import_list = [
    "gqtsam",
    "heoxnw",
    "hlqrxm",
    "jocrsz",
    "kfsxxl",
    "nsxhvs",
    "tehmzy",
    "wegawe",
    "ykujyn",
]

for fn in tqdm.tqdm(glob.glob(r"D:\BEHAVIOR-1K\asset_pipeline\cad\*\*\processed.max")):
    # Get the object list from the file
    f_objects = rt.getMAXFileObjectNames(fn, quiet=True)

    # Check if anything from that file matches
    matching_objects = []
    for obj in f_objects:
        m = b1k_pipeline.utils.parse_name(obj)
        if not m:
            continue
        if m.group("model_id") in import_list:
            matching_objects.append(obj)

    # If there's anything to import, get it!
    if matching_objects:
        success = rt.mergeMaxFile(
            fn,
            matching_objects,
            rt.Name("select"),
            rt.Name("autoRenameDups"),
            rt.Name("renameMtlDups"),
            quiet=True,
        )
