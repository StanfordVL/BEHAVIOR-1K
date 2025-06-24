import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.max.generate_fillable_volume import generate_fillable_mesh_from_seed

if __name__ == "__main__":
    generate_fillable_mesh_from_seed(is_open=True)
