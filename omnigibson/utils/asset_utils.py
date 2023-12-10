import argparse
import json
import os
import subprocess
import tempfile
import contextlib
import inspect
from copy import deepcopy
from pathlib import Path
from cryptography.fernet import Fernet
from collections import defaultdict
from urllib.request import urlretrieve
import yaml
import progressbar
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
if os.getenv("OMNIGIBSON_NO_OMNIVERSE", default=0) != "1":
    from pxr import Usd

# Create module logger
log = create_module_logger(module_name=__name__)

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def is_dot_file(p):
    """
    Check if a filename starts with a dot.
    Note that while this does not actually correspond to checking for hidden files on Windows, the
    files we want to ignore will still start with a dot and thus this works.

    Returns:
        bool: true if a folder is hidden in the OS
    """
    return p.startswith(".")


def get_og_avg_category_specs():
    """
    Load average object specs (dimension and mass) for objects

    Returns:
        dict: Average category specifications for all object categories
    """
    avg_obj_dim_file = os.path.join(gm.DATASET_PATH, "metadata", "avg_category_specs.json")
    if os.path.exists(avg_obj_dim_file):
        with open(avg_obj_dim_file) as f:
            return json.load(f)
    else:
        log.warning(
            "Requested average specs of the object categories in the OmniGibson Dataset of objects, but the "
            "file cannot be found. Did you download the dataset? Returning an empty dictionary"
        )
        return dict()


def get_og_category_ids():
    """
    Get OmniGibson object categories

    Returns:
        str: file path to the scene name
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_files = os.path.join(og_dataset_path, "metadata", "categories.txt")
    name_to_id = {}
    with open(og_categories_files, "r") as fp:
        for i, l in enumerate(fp.readlines()):
            name_to_id[l.rstrip()] = i
    return defaultdict(lambda: 255, name_to_id)


def get_available_og_scenes():
    """
    OmniGibson interactive scenes

    Returns:
        list: Available OmniGibson interactive scenes
    """
    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes")
    available_og_scenes = sorted(
        [f for f in os.listdir(og_scenes_path) if (not is_dot_file(f) and f != "background")]
    )
    return available_og_scenes


def get_og_scene_path(scene_name):
    """
    Get OmniGibson scene path

    Args:
        scene_name (str): scene name, e.g., "Rs_int"

    Returns:
        str: file path to the scene name
    """
    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes")
    log.info("Scene name: {}".format(scene_name))
    assert scene_name in os.listdir(og_scenes_path), "Scene {} does not exist".format(scene_name)
    return os.path.join(og_scenes_path, scene_name)


def get_og_category_path(category_name):
    """
    Get OmniGibson object category path

    Args:
        category_name (str): object category

    Returns:
        str: file path to the object category
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_path = os.path.join(og_dataset_path, "objects")
    assert category_name in os.listdir(og_categories_path), "Category {} does not exist".format(category_name)
    return os.path.join(og_categories_path, category_name)


def get_og_model_path(category_name, model_name):
    """
    Get OmniGibson object model path

    Args:
        category_name (str): object category
        model_name (str): object model

    Returns:
        str: file path to the object model
    """
    og_category_path = get_og_category_path(category_name)
    assert model_name in os.listdir(og_category_path), "Model {} from category {} does not exist".format(
        model_name, category_name
    )
    return os.path.join(og_category_path, model_name)


def get_all_system_categories():
    """
    Get OmniGibson all system categories

    Returns:
        list: all system categories
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_path = os.path.join(og_dataset_path, "systems")

    categories =[f for f in os.listdir(og_categories_path) if not is_dot_file(f)]
    return sorted(categories)


def get_all_object_categories():
    """
    Get OmniGibson all object categories

    Returns:
        list: all object categories
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_path = os.path.join(og_dataset_path, "objects")

    categories =[f for f in os.listdir(og_categories_path) if not is_dot_file(f)]
    return sorted(categories)


def get_all_object_models():
    """
    Get OmniGibson all object models

    Returns:
        list: all object model paths
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_path = os.path.join(og_dataset_path, "objects")

    categories = os.listdir(og_categories_path)
    categories = [item for item in categories if os.path.isdir(os.path.join(og_categories_path, item))]
    models = []
    for category in categories:
        category_models = os.listdir(os.path.join(og_categories_path, category))
        category_models = [
            item for item in category_models if os.path.isdir(os.path.join(og_categories_path, category, item))
        ]
        models.extend([os.path.join(og_categories_path, category, item) for item in category_models])
    return sorted(models)


def get_all_object_category_models(category):
    """
    Get all object models from @category

    Args:
        category (str): Object category name

    Returns:
        list of str: all object models belonging to @category
    """
    og_dataset_path = gm.DATASET_PATH
    og_categories_path = os.path.join(og_dataset_path, "objects", category)
    return sorted(os.listdir(og_categories_path)) if os.path.exists(og_categories_path) else []


def get_all_object_category_models_with_abilities(category, abilities):
    """
    Get all object models from @category whose assets are properly annotated with necessary metalinks to support
    abilities @abilities

    Args:
        category (str): Object category name
        abilities (dict): Dictionary mapping requested abilities to keyword arguments to pass to the corresponding
            object state constructors. The abilities' required annotations will be guaranteed for the returned
            models

    Returns:
        list of str: all object models belonging to @category which are properly annotated with necessary metalinks
            to support the requested list of @abilities
    """
    # Avoid circular imports
    from omnigibson.objects.dataset_object import DatasetObject
    from omnigibson.object_states.factory import get_states_for_ability
    from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin

    # Get all valid models
    all_models = get_all_object_category_models(category=category)

    # Generate all object states required per object given the requested set of abilities
    state_types_and_params = [(state_type, params) for ability, params in abilities.items()
                              for state_type in get_states_for_ability(ability)]
    for state_type, _ in state_types_and_params:
        # Add each state's dependencies, too. Note that only required dependencies are added.
        for dependency in state_type.get_dependencies():
            if all(other_state != dependency for other_state, _ in state_types_and_params):
                state_types_and_params.append((dependency, dict()))

    # Get mapping for class init kwargs
    state_init_default_kwargs = dict()
    for state_type, _ in state_types_and_params:
        default_kwargs = inspect.signature(state_type.__init__).parameters
        state_init_default_kwargs[state_type] = \
            {kwarg: val.default for kwarg, val in default_kwargs.items()
             if kwarg != "self" and val.default != inspect._empty}

    # Iterate over all models and sanity check each one, making sure they satisfy all the requested @abilities
    valid_models = []

    def supports_state_types(states_and_params, obj_prim):
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
            stage = Usd.Stage.Open(fpath)
            prim = stage.GetDefaultPrim()
            if supports_state_types(state_types_and_params, prim):
                valid_models.append(model)

    return valid_models


def get_og_assets_version():
    """
    Returns:
        str: OmniGibson asset version
    """
    process = subprocess.Popen(
        ["git", "-C", gm.DATASET_PATH, "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = str(process.communicate()[0].strip())
    return "{}".format(git_head_hash)


def get_available_g_scenes():
    """
    Returns:
        list: available Gibson scenes
    """
    data_path = og.g_dataset_path
    available_g_scenes = sorted([f for f in os.listdir(data_path) if not is_dot_file(f)])
    return available_g_scenes


def get_scene_path(scene_id):
    """
    Args:
        scene_id (str): scene id, e.g., "Rs_int"

    Returns:
        str: scene path for this scene_id
    """
    data_path = og.g_dataset_path
    assert scene_id in os.listdir(data_path), "Scene {} does not exist".format(scene_id)
    return os.path.join(data_path, scene_id)


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


def download_assets():
    """
    Download OmniGibson assets
    """
    if os.path.exists(gm.ASSET_PATH):
        print("Assets already downloaded.")
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp_file = os.path.join(td, "og_assets.tar.gz")
            os.makedirs(gm.ASSET_PATH, exist_ok=True)
            path = "https://storage.googleapis.com/gibson_scenes/og_assets.tar.gz"
            log.info(f"Downloading and decompressing demo OmniGibson assets from {path}")
            assert urlretrieve(path, tmp_file, show_progress), "Assets download failed."
            assert subprocess.call(["tar", "-zxf", tmp_file, "--strip-components=1", "--directory", gm.ASSET_PATH]) == 0, "Assets extraction failed."
            # These datasets come as folders; in these folder there are scenes, so --strip-components are needed.


def download_demo_data():
    """
    Download OmniGibson demo dataset
    """
    # TODO: Update. Right now, OG just downloads beta release
    download_og_dataset()


def print_user_agreement():
    print('\n\nBEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT\n'
        'Last revision: December 8, 2022\n'
        'This License Agreement is for the BEHAVIOR Data Bundle (“Data”). It works with OmniGibson (“Software”) which is a software stack licensed under the MIT License, provided in this repository: https://github.com/StanfordVL/OmniGibson. The license agreements for OmniGibson and the Data are independent. This BEHAVIOR Data Bundle contains artwork and images (“Third Party Content”) from third parties with restrictions on redistribution. It requires measures to protect the Third Party Content which we have taken such as encryption and the inclusion of restrictions on any reverse engineering and use. Recipient is granted the right to use the Data under the following terms and conditions of this License Agreement (“Agreement”):\n\n'
          '1. Use of the Data is permitted after responding "Yes" to this agreement. A decryption key will be installed automatically.\n'
          '2. Data may only be used for non-commercial academic research. You may not use a Data for any other purpose.\n'
          '3. The Data has been encrypted. You are strictly prohibited from extracting any Data from OmniGibson or reverse engineering.\n'
          '4. You may only use the Data within OmniGibson.\n'
          '5. You may not redistribute the key or any other Data or elements in whole or part.\n'
          '6. THE DATA AND SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR SOFTWARE OR THE USE OR OTHER DEALINGS IN THE DATA OR SOFTWARE.\n\n')


def download_key():
    os.makedirs(os.path.dirname(gm.KEY_PATH), exist_ok=True)
    if not os.path.exists(gm.ASSET_PATH):
        _=((()==())+(()==()));__=(((_<<_)<<_)*_);___=('c%'[::(([]!=[])-(()==()))])*(((_<<_)<<_)+(((_<<_)*_)+((_<<_)+(_+(()==())))))%((__+(((_<<_)<<_)+(_<<_))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_*_)))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_*_)))),(__+(((_<<_)<<_)+((_<<_)*_))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(((_<<_)<<_)+(((_<<_)*_)+((_<<_)+_))),(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==()))))),(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_*_)))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+(((_<<_)*_)+_))),(__+(((_<<_)<<_)+(()==()))),(__+(((_<<_)<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_*_)+(()==())))),(((_<<_)<<_)+((_<<_)+((_*_)+_))),(__+(((_<<_)<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+(_*_)))),(__+(((_<<_)<<_)+((_*_)+(()==())))),(__+(((_<<_)<<_)+(()==()))),(__+(((_<<_)<<_)+((_<<_)*_))),(__+(((_<<_)<<_)+((_<<_)+(()==())))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(((_<<_)<<_)+((_<<_)+((_*_)+_))),(__+(((_<<_)<<_)+(_+(()==())))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(()==()))))),(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+(()==())))),(__+(((_<<_)<<_)+_)),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+_)))),(__+(((_<<_)*_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(__+(((_<<_)<<_)+(_+(()==())))),(__+(((_<<_)<<_)+((_*_)+(()==())))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+_)))),(__+(((_<<_)<<_)+((_*_)+(()==())))),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+_)))),(__+(((_<<_)<<_)+((_<<_)+(()==())))),(__+(((_<<_)<<_)+((_*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+(()==())))),(__+(((_<<_)<<_)+_)),(__+(((_<<_)<<_)+(((_<<_)*_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+(_+(()==())))))),(__+(((_<<_)<<_)+((_<<_)+((_*_)+_)))),(((_<<_)<<_)+((_<<_)+((_*_)+_))),(__+(((_<<_)<<_)+((_<<_)+(_+(()==()))))),(__+(((_<<_)<<_)+((_*_)+(()==())))),(__+(((_<<_)<<_)+(((_<<_)*_)+((_<<_)+(()==()))))))
        path = ___
        assert urlretrieve(path, gm.KEY_PATH, show_progress), "Key download failed."


def download_og_dataset():
    """
    Download OmniGibson dataset
    """
    # Print user agreement
    if os.path.exists(gm.KEY_PATH):
        print("OmniGibson dataset encryption key already installed.")
    else:
        print("\n")
        print_user_agreement()
        while (
            input(
                "Do you agree to the above terms for using OmniGibson dataset? [y/n]"
            )
            != "y"
        ):
            print("You need to agree to the terms for using OmniGibson dataset.")

        download_key()

    if os.path.exists(gm.DATASET_PATH):
        print("OmniGibson dataset already installed.")
    else:
        tmp_file = os.path.join(tempfile.gettempdir(), "og_dataset.tar.gz")
        os.makedirs(gm.DATASET_PATH, exist_ok=True)
        path = "https://storage.googleapis.com/gibson_scenes/og_dataset.tar.gz"
        log.info(f"Downloading and decompressing demo OmniGibson dataset from {path}")
        assert urlretrieve(path, tmp_file, show_progress), "Dataset download failed."
        assert subprocess.call(["tar", "-zxf", tmp_file, "--strip-components=1", "--directory", gm.DATASET_PATH]) == 0, "Dataset extraction failed."
        # These datasets come as folders; in these folder there are scenes, so --strip-components are needed.


def change_data_path():
    """
    Changes the data paths for this repo
    """
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


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(gm.KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def encrypt_file(original_filename, encrypted_filename=None, encrypted_file=None):
    with open(gm.KEY_PATH, "rb") as filekey:
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
    fpath = Path(encrypted_filename)
    decrypted_filename = os.path.join(og.tempdir, f"{fpath.stem}.tmp{fpath.suffix}")
    decrypt_file(encrypted_filename=encrypted_filename, decrypted_filename=decrypted_filename)
    yield decrypted_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_assets", action="store_true", help="download assets file")
    parser.add_argument("--download_demo_data", action="store_true", help="download demo data Rs")
    parser.add_argument("--download_og_dataset", action="store_true", help="download OmniGibson Dataset")
    parser.add_argument("--change_data_path", action="store_true", help="change the path to store assets and datasets")

    args = parser.parse_args()

    if args.download_assets:
        download_assets()
    elif args.download_demo_data:
        download_demo_data()
    elif args.download_og_dataset:
        download_og_dataset()
    elif args.change_data_path:
        change_data_path()

    og.shutdown()
