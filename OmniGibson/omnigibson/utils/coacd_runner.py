"""
Subprocess to run coacd to prevent the generate_collision_meshes() in asset_conversion_utils.py from crashing
"""

import pickle
import coacd
import sys

try:
    with open(sys.argv[1], "rb") as f:
        vertices, faces, hull_count = pickle.load(f)
    mesh = coacd.Mesh(vertices, faces)
    result = coacd.run_coacd(
        mesh,
        max_convex_hull=hull_count,
        decimate=True,
        max_ch_vertex=60,
    )
    with open(sys.argv[2], "wb") as f:
        pickle.dump(result, f)
    sys.exit(0)
except Exception as e:
    print("Error in CoACD:", e)
    sys.exit(1)
