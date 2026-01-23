import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name


def main():
    meta_links = [
        x
        for x in rt.objects
        if parse_name(x.name) and parse_name(x.name).group("meta_type")
    ]
    hidden = all(x.isHidden for x in meta_links)
    for x in meta_links:
        x.isHidden = not hidden


if __name__ == "__main__":
    main()
