import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict

import yaml

import igibson

if os.name == "nt":
    import win32api
    import win32con


def folder_is_hidden(p):
    """
    Removes hidden folders from a list. Works on Linux, Mac and Windows

    :return: true if a folder is hidden in the OS
    """
    if os.name == "nt":
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith(".")  # linux-osx


def get_ig_avg_category_specs():
    """
    Load average object specs (dimension and mass) for objects
    """
    avg_obj_dim_file = os.path.join(igibson.ig_dataset_path, "metadata", "avg_category_specs.json")
    if os.path.exists(avg_obj_dim_file):
        with open(avg_obj_dim_file) as f:
            return json.load(f)
    else:
        logging.warning(
            "Requested average specs of the object categories in the iGibson Dataset of objects, but the "
            "file cannot be found. Did you download the dataset? Returning an empty dictionary"
        )
        return dict()


def get_ig_category_ids():
    """
    Get iGibson object categories

    :return: file path to the scene name
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_categories_files = os.path.join(ig_dataset_path, "metadata", "categories.txt")
    name_to_id = {}
    with open(ig_categories_files, "r") as fp:
        for i, l in enumerate(fp.readlines()):
            name_to_id[l.rstrip()] = i
    return defaultdict(lambda: 255, name_to_id)


def get_available_ig_scenes():
    """
    iGibson interactive scenes

    :return: list of available iGibson interactive scenes
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_scenes_path = os.path.join(ig_dataset_path, "scenes")
    available_ig_scenes = sorted(
        [f for f in os.listdir(ig_scenes_path) if (not folder_is_hidden(f) and f != "background")]
    )
    return available_ig_scenes


def get_ig_scene_path(scene_name):
    """
    Get iGibson scene path

    :param scene_name: scene name
    :return: file path to the scene name
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_scenes_path = os.path.join(ig_dataset_path, "scenes")
    logging.info("Scene name: {}".format(scene_name))
    assert scene_name in os.listdir(ig_scenes_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(ig_scenes_path, scene_name)


def get_3dfront_scene_path(scene_name):
    """
    Get 3D-FRONT scene path

    :param scene_name: scene name
    :return: file path to the scene name
    """
    threedfront_dataset_path = igibson.threedfront_dataset_path
    threedfront_dataset_path = os.path.join(threedfront_dataset_path, "scenes")
    assert scene_name in os.listdir(threedfront_dataset_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(threedfront_dataset_path, scene_name)


def get_cubicasa_scene_path(scene_name):
    """
    Get cubicasa scene path

    :param scene_name: scene name
    :return: file path to the scene name
    """
    cubicasa_dataset_path = igibson.cubicasa_dataset_path
    cubicasa_dataset_path = os.path.join(cubicasa_dataset_path, "scenes")
    assert scene_name in os.listdir(cubicasa_dataset_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(cubicasa_dataset_path, scene_name)


def get_ig_category_path(category_name):
    """
    Get iGibson object category path

    :param category_name: object category
    :return: file path to the object category
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_categories_path = os.path.join(ig_dataset_path, "objects")
    assert category_name in os.listdir(ig_categories_path), "Category {} does not exist".format(category_name)
    return os.path.join(ig_categories_path, category_name)


def get_ig_model_path(category_name, model_name):
    """
    Get iGibson object model path

    :param category_name: object category
    :param model_name: object model
    :return: file path to the object model
    """
    ig_category_path = get_ig_category_path(category_name)
    assert model_name in os.listdir(ig_category_path), "Model {} from category {} does not exist".format(
        model_name, category_name
    )
    return os.path.join(ig_category_path, model_name)


def get_object_models_of_category(category_name, filter_method=None):
    """
    Get iGibson all object models of a given category

    :return: a list of all object models of a given
    """
    models = []
    ig_category_path = get_ig_category_path(category_name)
    for model_name in os.listdir(ig_category_path):
        if filter_method is None:
            models.append(model_name)
        elif filter_method in ["sliceable_part", "sliceable_whole"]:
            model_path = get_ig_model_path(category_name, model_name)
            metadata_json = os.path.join(model_path, "misc", "metadata.json")
            with open(metadata_json) as f:
                metadata = json.load(f)
            if (filter_method == "sliceable_part" and "object_parts" not in metadata) or (
                filter_method == "sliceable_whole" and "object_parts" in metadata
            ):
                models.append(model_name)
        else:
            raise Exception("Unknown filter method: {}".format(filter_method))
    return models


def get_all_object_categories():
    """
    Get iGibson all object categories

    :return: a list of all object categories
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_categories_path = os.path.join(ig_dataset_path, "objects")

    categories = sorted([f for f in os.listdir(ig_categories_path) if not folder_is_hidden(f)])
    return categories


def get_all_object_models():
    """
    Get iGibson all object models

    :return: a list of all object model paths
    """
    ig_dataset_path = igibson.ig_dataset_path
    ig_categories_path = os.path.join(ig_dataset_path, "objects")

    categories = os.listdir(ig_categories_path)
    categories = [item for item in categories if os.path.isdir(os.path.join(ig_categories_path, item))]
    models = []
    for category in categories:
        category_models = os.listdir(os.path.join(ig_categories_path, category))
        category_models = [
            item for item in category_models if os.path.isdir(os.path.join(ig_categories_path, category, item))
        ]
        models.extend([os.path.join(ig_categories_path, category, item) for item in category_models])
    return models


def get_ig_assets_version():
    """
    Get iGibson asset version

    :return: iGibson asset version
    """
    process = subprocess.Popen(
        ["git", "-C", igibson.ig_dataset_path, "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = str(process.communicate()[0].strip())
    return "{}".format(git_head_hash)


def get_available_g_scenes():
    """
    Gibson scenes

    :return: list of available Gibson scenes
    """
    data_path = igibson.g_dataset_path
    available_g_scenes = sorted([f for f in os.listdir(data_path) if not folder_is_hidden(f)])
    return available_g_scenes


def get_scene_path(scene_id):
    """
    Gibson scene path

    :param scene_id: scene id
    :return: scene path for this scene_id
    """
    data_path = igibson.g_dataset_path
    assert scene_id in os.listdir(data_path) or scene_id == "stadium", "Scene {} does not exist".format(scene_id)
    return os.path.join(data_path, scene_id)


def get_texture_file(mesh_file):
    """
    Get texture file

    :param mesh_file: mesh obj file
    :return: texture file path
    """
    model_dir = os.path.dirname(mesh_file)
    with open(mesh_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if "mtllib" in line]
        if len(lines) == 0:
            return
        mtl_file = lines[0].split()[1]
        mtl_file = os.path.join(model_dir, mtl_file)

    with open(mtl_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if "map_Kd" in line]
        if len(lines) == 0:
            return
        texture_file = lines[0].split()[1]
        texture_file = os.path.join(model_dir, texture_file)

    return texture_file


def download_assets():
    """
    Download iGibson assets
    """

    tmp_file = os.path.join(tempfile.gettempdir(), "assets_igibson.tar.gz")

    os.makedirs(os.path.dirname(igibson.assets_path), exist_ok=True)

    if not os.path.exists(igibson.assets_path):
        logging.info(
            "Downloading and decompressing assets from {}".format(
                "https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz"
            )
        )
        os.system(
            "wget -c --retry-connrefused --tries=5 --timeout=5 "
            "https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz -O {}".format(tmp_file)
        )
        os.system("tar -zxf {} --directory {}".format(tmp_file, os.path.dirname(igibson.assets_path)))


def download_demo_data():
    """
    Download iGibson demo dataset
    """

    tmp_file = os.path.join(tempfile.gettempdir(), "Rs.tar.gz")

    os.makedirs(igibson.g_dataset_path, exist_ok=True)

    if not os.path.exists(os.path.join(igibson.g_dataset_path, "Rs")):
        logging.info(
            "Downloading and decompressing Rs Gibson meshfile from {}".format(
                "https://storage.googleapis.com/gibson_scenes/Rs.tar.gz"
            )
        )
        os.system(
            "wget -c --retry-connrefused --tries=5 --timeout=5  "
            "https://storage.googleapis.com/gibson_scenes/Rs.tar.gz -O {}".format(tmp_file)
        )
        os.system("tar -zxf {} --directory {}".format(tmp_file, igibson.g_dataset_path))


def download_dataset(url):
    """
    Download Gibson dataset
    """

    if not os.path.exists(igibson.g_dataset_path):
        logging.info("Creating Gibson dataset folder at {}".format(igibson.g_dataset_path))
        os.makedirs(igibson.g_dataset_path)

    file_name = url.split("/")[-1]

    tmp_file = os.path.join(tempfile.gettempdir(), file_name)

    logging.info("Downloading and decompressing the full Gibson dataset from {}".format(url))
    os.system("wget -c --retry-connrefused --tries=5 --timeout=5 {} -O {}".format(url, tmp_file))
    os.system("tar -zxf {} --strip-components=1 --directory {}".format(tmp_file, igibson.g_dataset_path))
    # These datasets come as folders; in these folder there are scenes, so --strip-components are needed.


def download_ext_scene_assets():
    logging.info("Downloading and decompressing 3DFront and Cubicasa")
    os.makedirs(igibson.threedfront_dataset_path, exist_ok=True)
    os.makedirs(igibson.cubicasa_dataset_path, exist_ok=True)
    url = "https://storage.googleapis.com/gibson_scenes/default_materials.tar.gz"

    file_name = url.split("/")[-1]
    tmp_file = os.path.join(tempfile.gettempdir(), file_name)

    os.system("wget -c --retry-connrefused --tries=5 --timeout=5 {} -O /tmp/{}".format(url, file_name))
    os.system("tar -zxf {} --directory {}".format(tmp_file, igibson.cubicasa_dataset_path))
    os.system("tar -zxf {} --directory {}".format(tmp_file, igibson.threedfront_dataset_path))

    url = "https://storage.googleapis.com/gibson_scenes/threedfront_urdfs.tar.gz"
    file_name = url.split("/")[-1]
    tmp_file = os.path.join(tempfile.gettempdir(), file_name)

    os.system("wget -c --retry-connrefused --tries=5 --timeout=5 {} -O /tmp/{}".format(url, file_name))
    os.system("tar -zxf {} --directory {}".format(tmp_file, igibson.threedfront_dataset_path))


def download_ig_dataset():
    """
    Download iGibson dataset
    """
    while (
        input(
            "Do you agree to the terms for using iGibson dataset (http://svl.stanford.edu/igibson/assets/GDS_agreement.pdf)? [y/n]"
        )
        != "y"
    ):
        print("You need to agree to the terms for using iGibson dataset.")

    if not os.path.exists(igibson.ig_dataset_path):
        logging.info("Creating iGibson dataset folder at {}".format(igibson.g_dataset_path))
        os.makedirs(igibson.ig_dataset_path)

    url = "https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz"
    file_name = url.split("/")[-1]
    tmp_file = os.path.join(tempfile.gettempdir(), file_name)

    logging.info("Downloading and decompressing the full iGibson dataset of scenes from {}".format(url))
    os.system("wget -c --retry-connrefused --tries=5 --timeout=5 {} -O {}".format(url, tmp_file))
    os.system("tar -zxf {} --strip-components=1 --directory {}".format(tmp_file, igibson.ig_dataset_path))
    # These datasets come as folders; in these folder there are scenes, so --strip-components are needed.


def change_data_path():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "global_config.yaml")) as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    print("Current dataset path:")
    for k, v in global_config.items():
        print("{}: {}".format(k, v))
    for k, v in global_config.items():
        new_path = input("Change {} from {} to: ".format(k, v))
        global_config[k] = new_path

    print("New dataset path:")
    for k, v in global_config.items():
        print("{}: {}".format(k, v))
    response = input("Save? [y/n]")
    if response == "y":
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "global_config.yaml"), "w") as f:
            yaml.dump(global_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_assets", action="store_true", help="download assets file")
    parser.add_argument("--download_demo_data", action="store_true", help="download demo data Rs")
    parser.add_argument("--download_dataset", type=str, help="download dataset file given an URL")
    parser.add_argument("--download_ig_dataset", action="store_true", help="download iG Dataset")
    parser.add_argument(
        "--download_ext_scene_assets", action="store_true", help="download external scene dataset assets"
    )
    parser.add_argument("--change_data_path", action="store_true", help="change the path to store assets and datasets")

    args = parser.parse_args()

    if args.download_assets:
        download_assets()
    elif args.download_demo_data:
        download_demo_data()
    elif args.download_dataset is not None:
        download_dataset(args.download_dataset)
    elif args.download_ig_dataset:
        download_ig_dataset()
    elif args.change_data_path:
        change_data_path()
    elif args.download_ext_scene_assets:
        download_ext_scene_assets()
