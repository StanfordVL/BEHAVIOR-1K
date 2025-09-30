import argparse
import contextlib
from importlib.metadata import version
import inspect
import json
import os
from packaging.version import Version
import pathlib
import shutil
import subprocess
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import bddl
from huggingface_hub import hf_hub_download
from cryptography.fernet import Fernet

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

if os.getenv("OMNIGIBSON_NO_OMNIVERSE", default=0) != "1":
    import omnigibson.lazy as lazy

# Create module logger
log = create_module_logger(module_name=__name__)


def is_dot_file(p):
    """
    Check if a filename starts with a dot.
    Note that while this does not actually correspond to checking for hidden files on Windows, the
    files we want to ignore will still start with a dot and thus this works.

    Returns:
        bool: true if a folder is hidden in the OS
    """
    return p.startswith(".")


def get_dataset_path(dataset_name):
    return os.path.join(gm.DATA_PATH, dataset_name)


def get_key_path():
    return os.path.join(gm.DATA_PATH, "omnigibson.key")


def get_avg_category_specs():
    """
    Load average object specs (dimension and mass) for objects

    Returns:
        dict: Average category specifications for all object categories
    """
    avg_obj_dim_file = os.path.join(og.example_config_path, "avg_category_specs.json")
    if os.path.exists(avg_obj_dim_file):
        with open(avg_obj_dim_file) as f:
            return json.load(f)
    else:
        log.warning(
            "Requested average specs of the object categories in the BEHAVIOR-1K Dataset of objects, but the "
            "file cannot be found. Did you download the dataset? Returning an empty dictionary"
        )
        return dict()


def get_behavior_1k_category_ids():
    """
    Get OmniGibson object categories

    Returns:
        str: file path to the scene name
    """
    behavior_1k_assets_path = get_dataset_path("behavior-1k-assets")
    categories_files = os.path.join(behavior_1k_assets_path, "metadata", "categories.txt")
    name_to_id = {}
    with open(categories_files, "r") as fp:
        for i, l in enumerate(fp.readlines()):
            name_to_id[l.rstrip()] = i
    return defaultdict(lambda: 255, name_to_id)


def get_available_behavior_1k_scenes():
    """
    OmniGibson interactive scenes

    Returns:
        list: Available OmniGibson interactive scenes
    """
    behavior_1k_assets_path = get_dataset_path("behavior-1k-assets")
    behavior_1k_scenes_path = os.path.join(behavior_1k_assets_path, "scenes")
    available_behavior_1k_scenes = sorted(
        [f for f in os.listdir(behavior_1k_scenes_path) if (not is_dot_file(f) and f != "background")]
    )
    return available_behavior_1k_scenes


def get_scene_path(scene_name, dataset_name="behavior-1k-assets"):
    """
    Get OmniGibson scene path

    Args:
        scene_name (str): scene name, e.g., "Rs_int"

    Returns:
        str: file path to the scene name
    """
    dataset_path = get_dataset_path(dataset_name)
    scenes_path = os.path.join(dataset_path, "scenes")
    log.info("Scene name: {}".format(scene_name))
    assert scene_name in os.listdir(scenes_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(scenes_path, scene_name)


def get_task_instance_path(scene_name):
    """
    Get task instance path

    Args:
        scene_name (str): scene name, e.g., "Rs_int"

    Returns:
        str: file path to the scene name
    """
    task_instances_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances")
    scenes_path = os.path.join(task_instances_path, "scenes")
    log.info("Scene name: {}".format(scene_name))
    if scene_name in os.listdir(scenes_path):
        return os.path.join(scenes_path, scene_name)
    else:
        return None


def get_category_path(category_name, dataset_name="behavior-1k-assets"):
    """
    Get OmniGibson object category path

    Args:
        category_name (str): object category

    Returns:
        str: file path to the object category
    """
    dataset_path = get_dataset_path(dataset_name)
    categories_path = os.path.join(dataset_path, "objects")
    assert category_name in os.listdir(categories_path), "Category {} does not exist".format(category_name)
    return os.path.join(categories_path, category_name)


def get_model_path(category_name, model_name, dataset_name="behavior-1k-assets"):
    """
    Get OmniGibson object model path

    Args:
        category_name (str): object category
        model_name (str): object model

    Returns:
        str: file path to the object model
    """
    category_path = get_category_path(category_name, dataset_name=dataset_name)
    assert model_name in os.listdir(category_path), "Model {} from category {} in dataset {} does not exist".format(
        model_name, category_name, dataset_name
    )
    return os.path.join(category_path, model_name)


def get_all_system_categories(include_cloth=False):
    """
    Get OmniGibson all system categories

    Args:
        include_cloth (bool): whether to include cloth category; default to only include non-cloth particle systems

    Returns:
        list: all system categories
    """
    behavior_1k_assets_path = get_dataset_path("behavior-1k-assets")
    categories_path = os.path.join(behavior_1k_assets_path, "systems")

    categories = [f for f in os.listdir(categories_path) if not is_dot_file(f)]
    if include_cloth:
        categories.append("cloth")
    return sorted(categories)


def get_all_object_categories():
    """
    Get OmniGibson all object categories

    Returns:
        list: all object categories
    """
    categories = {x.name for x in Path(gm.DATA_PATH).glob("*/objects/*") if x.is_dir() and not is_dot_file(x.name)}
    return sorted(categories)


def get_all_object_models():
    """
    Get OmniGibson all object models

    Returns:
        list: all object model paths
    """
    return sorted({str(x) for x in Path(gm.DATA_PATH).glob("*/objects/*/*") if x.is_dir() and not is_dot_file(x.name)})


def get_all_object_category_models(category):
    """
    Get all object models from @category

    Args:
        category (str): Object category name

    Returns:
        list of str: all object models belonging to @category
    """
    behavior_1k_assets_path = get_dataset_path("behavior-1k-assets")
    categories_path = os.path.join(behavior_1k_assets_path, "objects", category)
    return sorted(os.listdir(categories_path)) if os.path.exists(categories_path) else []


def get_all_object_category_models_with_abilities(category, abilities):
    """
    Get all object models from @category whose assets are properly annotated with necessary requirements to support
    abilities @abilities

    Args:
        category (str): Object category name
        abilities (dict): Dictionary mapping requested abilities to keyword arguments to pass to the corresponding
            object state constructors. The abilities' required annotations will be guaranteed for the returned
            models

    Returns:
        list of str: all object models belonging to @category which are properly annotated with necessary requirements
            to support the requested list of @abilities
    """
    # Avoid circular imports
    from omnigibson.object_states.factory import get_requirements_for_ability, get_states_for_ability
    from omnigibson.objects.dataset_object import DatasetObject

    # Get all valid models
    all_models = get_all_object_category_models(category=category)

    # Generate all object states required per object given the requested set of abilities
    abilities_info = {
        ability: [(state_type, params) for state_type in get_states_for_ability(ability)]
        for ability, params in abilities.items()
    }

    # Get mapping for class init kwargs
    state_init_default_kwargs = dict()

    for ability, state_types_and_params in abilities_info.items():
        for state_type, _ in state_types_and_params:
            # Add each state's dependencies, too. Note that only required dependencies are added.
            for dependency in state_type.get_dependencies():
                if all(other_state != dependency for other_state, _ in state_types_and_params):
                    state_types_and_params.append((dependency, dict()))

        for state_type, _ in state_types_and_params:
            default_kwargs = inspect.signature(state_type.__init__).parameters
            state_init_default_kwargs[state_type] = {
                kwarg: val.default
                for kwarg, val in default_kwargs.items()
                if kwarg != "self" and val.default != inspect._empty
            }

    # Iterate over all models and sanity check each one, making sure they satisfy all the requested @abilities
    valid_models = []

    def supports_abilities(info, obj_prim):
        for ability, states_and_params in info.items():
            # Check ability requirements
            for requirement in get_requirements_for_ability(ability):
                if not requirement.is_compatible_asset(prim=obj_prim)[0]:
                    return False

            # Check all link states
            for state_type, params in states_and_params:
                kwargs = deepcopy(state_init_default_kwargs[state_type])
                kwargs.update(params)
                if not state_type.is_compatible_asset(prim=obj_prim, **kwargs)[0]:
                    return False
        return True

    for model in all_models:
        usd_path = DatasetObject.get_usd_path(category=category, model=model)
        usd_path = usd_path.replace(".usd", ".encrypted.usd")
        with decrypted(usd_path) as fpath:
            stage = lazy.pxr.Usd.Stage.Open(fpath)
            prim = stage.GetDefaultPrim()
            if supports_abilities(abilities_info, prim):
                valid_models.append(model)

    return valid_models


def get_attachment_meta_links(category, model):
    """
    Get attachment meta links for an object model

    Args:
        category (str): Object category name
        model (str): Object model name

    Returns:
        list of str: all attachment meta links for the object model
    """
    # Avoid circular imports
    from omnigibson.object_states import AttachedTo
    from omnigibson.objects.dataset_object import DatasetObject

    usd_path = DatasetObject.get_usd_path(category=category, model=model)
    usd_path = usd_path.replace(".usd", ".encrypted.usd")
    with decrypted(usd_path) as fpath:
        stage = lazy.pxr.Usd.Stage.Open(fpath)
        prim = stage.GetDefaultPrim()
        attachment_meta_links = []
        for child in prim.GetChildren():
            if child.GetTypeName() == "Xform":
                if any(meta_link_type in child.GetName() for meta_link_type in AttachedTo.meta_link_types):
                    attachment_meta_links.append(child.GetName())
        return attachment_meta_links


def get_omnigibson_robot_asset_git_hash():
    """
    Returns:
        str: OmniGibson asset commit hash
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", get_dataset_path("omnigibson-robot-assets"), "rev-parse", "HEAD"],
            shell=False,
            stderr=subprocess.DEVNULL,
        )
        return git_hash.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None


def get_omnigibson_robot_asset_version():
    """
    Returns:
        str: OmniGibson assets version
    """
    try:
        return (Path(get_dataset_path("omnigibson-robot-assets")) / "VERSION").read_text().strip()
    except FileNotFoundError:
        return None


def get_omnigibson_git_hash():
    """
    Returns:
        str: OmniGibson commit hash
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", Path(og.__file__).parent, "rev-parse", "HEAD"], shell=False, stderr=subprocess.DEVNULL
        )
        return git_hash.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None


def get_omnigibson_version():
    return og.__version__


def get_bddl_git_hash():
    """
    Returns:
        str: bddl asset commit hash
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", Path(bddl.__file__).parent, "rev-parse", "HEAD"], shell=False, stderr=subprocess.DEVNULL
        )
        return git_hash.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None


def get_bddl_version():
    return version("bddl")


def get_behavior_1k_assets_version():
    """
    Returns:
        str: BEHAVIOR-1K dataset version
    """
    try:
        return (Path(get_dataset_path("behavior-1k-assets")) / "VERSION").read_text().strip()
    except FileNotFoundError:
        return None


def check_minimum_behavior_1k_assets_version(min_version_str):
    current_version = get_behavior_1k_assets_version()
    if current_version is None:
        return False

    return Version(current_version) >= Version(min_version_str)


def get_texture_file(mesh_file):
    """
    Get texture file

    Args:
        mesh_file (str): path to mesh obj file

    Returns:
        str: texture file path
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


def download_and_unpack_zipped_dataset(dataset_name):
    # The current version of the dataset that should be downloaded
    BEHAVIOR_1K_DATASET_VERSION = "3.7.2rc1"

    tempdir = tempfile.mkdtemp()
    real_target = get_dataset_path(dataset_name)
    online_filename = (
        f"behavior-1k-assets-{BEHAVIOR_1K_DATASET_VERSION}.zip"
        if dataset_name == "behavior-1k-assets"
        else f"{dataset_name}.zip"
    )
    local_path = hf_hub_download(
        repo_id="behavior-1k/zipped-datasets",
        filename=online_filename,
        repo_type="dataset",
        local_dir=tempdir,
    )
    with zipfile.ZipFile(local_path, "r") as zip_ref:
        zip_ref.extractall(real_target)
    shutil.rmtree(tempdir)


def download_omnigibson_robot_assets():
    """
    Download OmniGibson assets
    """
    if os.path.exists(get_dataset_path("omnigibson-robot-assets")):
        print("Assets already downloaded.")
    else:
        download_and_unpack_zipped_dataset("omnigibson-robot-assets")


def print_user_agreement():
    print(
        "\n\nBEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT\n"
        "Last revision: December 8, 2022\n"
        "This License Agreement is for the BEHAVIOR Data Bundle (“Data”). It works with OmniGibson (“Software”) which is a software stack licensed under the MIT License, provided in this repository: https://github.com/StanfordVL/OmniGibson. The license agreements for OmniGibson and the Data are independent. This BEHAVIOR Data Bundle contains artwork and images (“Third Party Content”) from third parties with restrictions on redistribution. It requires measures to protect the Third Party Content which we have taken such as encryption and the inclusion of restrictions on any reverse engineering and use. Recipient is granted the right to use the Data under the following terms and conditions of this License Agreement (“Agreement”):\n\n"
        '1. Use of the Data is permitted after responding "Yes" to this agreement. A decryption key will be installed automatically.\n'
        "2. Data may only be used for non-commercial academic research. You may not use a Data for any other purpose.\n"
        "3. The Data has been encrypted. You are strictly prohibited from extracting any Data from OmniGibson or reverse engineering.\n"
        "4. You may only use the Data within OmniGibson.\n"
        "5. You may not redistribute the key or any other Data or elements in whole or part.\n"
        '6. THE DATA AND SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR SOFTWARE OR THE USE OR OTHER DEALINGS IN THE DATA OR SOFTWARE.\n\n'
    )


def download_key():
    if not os.path.exists(get_key_path()):
        _ = (() == ()) + (() == ())
        __ = ((_ << _) << _) * _
        ___ = (
            ("c%"[:: (([] != []) - (() == ()))])
            * (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + (_ + (() == ())))))
            % (
                (__ + (((_ << _) << _) + (_ << _))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
                (__ + (((_ << _) << _) + ((_ << _) * _))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + _))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ * _)))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + _))),
                (__ + (((_ << _) << _) + (() == ()))),
                (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
                (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + (_ * _)))),
                (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
                (__ + (((_ << _) << _) + (() == ()))),
                (__ + (((_ << _) << _) + ((_ << _) * _))),
                (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
                (__ + (((_ << _) << _) + (_ + (() == ())))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (() == ()))))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
                (__ + (((_ << _) << _) + _)),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
                (__ + (((_ << _) * _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + (_ + (() == ())))),
                (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
                (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
                (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
                (__ + (((_ << _) << _) + ((_ * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + (() == ())))),
                (__ + (((_ << _) << _) + _)),
                (__ + (((_ << _) << _) + (((_ << _) * _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + (_ + (() == ())))))),
                (__ + (((_ << _) << _) + ((_ << _) + ((_ * _) + _)))),
                (((_ << _) << _) + ((_ << _) + ((_ * _) + _))),
                (__ + (((_ << _) << _) + ((_ << _) + (_ + (() == ()))))),
                (__ + (((_ << _) << _) + ((_ * _) + (() == ())))),
                (__ + (((_ << _) << _) + (((_ << _) * _) + ((_ << _) + (() == ()))))),
            )
        )
        path = ___
        assert urlretrieve(path, get_key_path()), "Key download failed."


def download_behavior_1k_assets(accept_license=False):
    """
    Download BEHAVIOR-1K dataset
    """
    # Print user agreement
    if os.path.exists(get_key_path()):
        print("BEHAVIOR-1K dataset encryption key already installed.")
    else:
        if not accept_license:
            print("\n")
            print_user_agreement()
            while input("Do you agree to the above terms for using BEHAVIOR-1K dataset? [y/n]") != "y":
                print("You need to agree to the terms for using BEHAVIOR-1K dataset.")

        download_key()

    if os.path.exists(get_dataset_path("behavior-1k-assets")):
        print("BEHAVIOR-1K dataset already installed.")
    else:
        download_and_unpack_zipped_dataset("behavior-1k-assets")


def download_2025_challenge_task_instances():
    if os.path.exists(get_dataset_path("2025-challenge-task-instances")):
        # cd and git pull
        subprocess.run(
            ["git", "-C", get_dataset_path("2025-challenge-task-instances"), "pull"],
            shell=False,
            check=True,
        )
        print("2025 BEHAVIOR Challenge Tasks Instances updated.")
    else:
        download_and_unpack_zipped_dataset("2025-challenge-task-instances")


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(get_key_path(), "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def encrypt_file(original_filename, encrypted_filename=None, encrypted_file=None):
    with open(get_key_path(), "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(original_filename, "rb") as org_f:
        original = org_f.read()

    encrypted = fernet.encrypt(original)

    if encrypted_file is not None:
        encrypted_file.write(encrypted)
    else:
        with open(encrypted_filename, "wb") as encrypted_file:
            encrypted_file.write(encrypted)


@contextlib.contextmanager
def decrypted(encrypted_filename):
    decrypted_filename_template = pathlib.Path(encrypted_filename.replace(".encrypted", ""))
    decrypted_fd, decrypted_filename = tempfile.mkstemp(
        suffix=decrypted_filename_template.suffix, prefix=decrypted_filename_template.stem, dir=og.tempdir
    )
    os.close(decrypted_fd)
    decrypt_file(encrypted_filename=encrypted_filename, decrypted_filename=decrypted_filename)
    yield decrypted_filename
    os.remove(decrypted_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_omnigibson_robot_assets", action="store_true", help="download assets file")
    parser.add_argument("--download_behavior_1k_assets", action="store_true", help="download BEHAVIOR-1K Dataset")
    parser.add_argument(
        "--download_2025_challenge_task_instances",
        action="store_true",
        help="download 2025 BEHAVIOR Challenge Tasks dataset",
    )
    parser.add_argument("--accept_license", action="store_true", help="pre-accept the BEHAVIOR-1K dataset license")
    args = parser.parse_args()

    if args.download_omnigibson_robot_assets:
        download_omnigibson_robot_assets()
    if args.download_behavior_1k_assets:
        download_behavior_1k_assets(accept_license=args.accept_license)
    if args.download_2025_challenge_task_instances:
        download_2025_challenge_task_instances()

    og.shutdown()
