import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.max.convex_decomposition import generate_convex_decompositions
from pymxs import runtime as rt


def main():
    for obj in rt.selection:
        generate_convex_decompositions(obj, preferred_method="chull")


if __name__ == "__main__":
    main()
