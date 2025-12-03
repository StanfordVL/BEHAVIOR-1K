from fs.zipfs import ZipFS
import fs.path
import fs.copy
import tqdm

from b1k_pipeline.utils import PipelineFS, get_targets, parse_name


def main():
    with PipelineFS() as pipeline_fs:
        with (
            pipeline_fs.pipeline_output().open("max_object_images.zip", "wb") as zip_f,
            ZipFS(zip_f, write=True) as zip_fs,
        ):
            for target in tqdm.tqdm(
                get_targets("combined"), desc=" targets", position=0
            ):
                target_output_fs = pipeline_fs.target_output(target)

                with ZipFS(
                    target_output_fs.open("max_object_images.zip", "rb")
                ) as target_images:
                    # Walk through all images
                    paths = list(target_images.listdir("/"))

                    for path in tqdm.tqdm(
                        paths, desc=" images", position=1, leave=False
                    ):
                        name, ext = fs.path.splitext(path)
                        obj_name, photo_name = name.split("--")
                        parsed_name = parse_name(obj_name)
                        assert (
                            parsed_name is not None
                        ), f"Found bad name {path} in {target}"
                        model_id = parsed_name.group("model_id")
                        output_path = f"{model_id}-{photo_name}{ext}"

                        fs.copy.copy_file(target_images, path, zip_fs, output_path)


if __name__ == "__main__":
    main()
