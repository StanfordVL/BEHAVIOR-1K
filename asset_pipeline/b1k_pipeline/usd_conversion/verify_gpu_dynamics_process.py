import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject

gm.HEADLESS = True
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False


def process_object(cat, mdl):
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "obj",
                "category": cat,
                "model": mdl,
                "kinematic_only": False,
                "fixed_base": True,
            },
        ],
    }

    env = og.Environment(configs=cfg)

    for _ in range(100):
        og.sim.step()

    print(f"{mdl} finished successfully.", flush=True)


def main():
    import sys, pathlib

    dataset_root = str(pathlib.Path(sys.argv[1]))
    gm.DATASET_PATH = str(dataset_root)

    path = sys.argv[2]
    obj_category, obj_model = pathlib.Path(path).parts[-2:]
    obj_dir = pathlib.Path(dataset_root) / "objects" / obj_category / obj_model
    assert obj_dir.exists()
    process_object(obj_category, obj_model)


if __name__ == "__main__":
    main()
