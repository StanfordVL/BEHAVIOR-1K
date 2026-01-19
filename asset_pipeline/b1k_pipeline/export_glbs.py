import os
import traceback
from b1k_pipeline.urdfpy import URDF
import b1k_pipeline.utils
import trimesh
import fs.copy
from fs.tempfs import TempFS
from dask.distributed import LocalCluster, as_completed
import tqdm
import glob


def urdf_to_glb(in_obj_dir, out_obj_dir):
    # Get the URDF file
    urdf_files = glob.glob(in_obj_dir + "/urdf/*.urdf")
    assert (
        len(urdf_files) == 1
    ), f"Expected exactly one URDF file in {in_obj_dir}, found {len(urdf_files)}"

    urdf_file = urdf_files[0]

    robot = URDF.load(urdf_file)
    links = [l for l in robot.links if "meta__" not in l.name]
    visual_fk = robot.visual_trimesh_fk(links=links)

    scene = trimesh.Scene()
    for mesh, transform in visual_fk.items():
        m = mesh.copy()
        m.apply_transform(transform)
        scene.add_geometry(m)

    model_id = os.path.splitext(os.path.basename(urdf_file))[0]
    out_file = os.path.join(out_obj_dir, f"{model_id}.glb")
    scene.export(out_file)


def main():
    with (
        b1k_pipeline.utils.ParallelZipFS("objects.zip") as source_fs,
        TempFS(temp_dir=str(b1k_pipeline.utils.TMP_DIR)) as temp_fs,
        b1k_pipeline.utils.ParallelZipFS("objects_glb.zip", write=True) as out_fs,
    ):
        # Copy everything over to the temp FS
        print("Copying input to temp fs...")
        objdir_glob = [item.path for item in source_fs.glob("objects/*/*/")]
        for item in tqdm.tqdm(objdir_glob):
            if (
                source_fs.opendir(item).opendir("urdf").glob("*.urdf").count().files
                == 0
            ):
                continue
            fs.copy.copy_fs(
                source_fs.opendir(item), temp_fs.makedirs(item, recreate=True)
            )

        cluster = LocalCluster()
        dask_client = cluster.get_client()

        obj_futures = {}

        for objdir in tqdm.tqdm(
            objdir_glob, desc="Processing targets to queue objects"
        ):
            obj_futures[
                dask_client.submit(
                    urdf_to_glb,
                    temp_fs.opendir(objdir).getsyspath("/"),
                    out_fs.makedirs(objdir).getsyspath("/"),
                    pure=False,
                )
            ] = objdir

        for future in tqdm.tqdm(
            as_completed(obj_futures.keys()),
            total=len(obj_futures),
            desc="Processing objects",
        ):
            try:
                future.result()
            except:
                traceback.print_exc()

        print("Finished processing")


if __name__ == "__main__":
    main()
