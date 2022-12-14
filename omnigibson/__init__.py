import logging
import os

import yaml
import builtins

# TODO: Need to fix somehow -- omnigibson gets imported first BEFORE we can actually modify the macros
from omnigibson.macros import gm

builtins.ISAAC_LAUNCHED_FROM_JUPYTER = (
    os.getenv("ISAAC_JUPYTER_KERNEL") is not None
)  # We set this in the kernel.json file

# Always enable nest_asyncio because MaterialPrim calls asyncio.run()
import nest_asyncio
nest_asyncio.apply()

__version__ = "0.0.1"

logging.getLogger().setLevel(logging.INFO)

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "global_config.yaml")) as f:
    global_config = yaml.load(f, Loader=yaml.FullLoader)

# can override assets_path and dataset_path from environment variable
if "GIBSON_ASSETS_PATH" in os.environ:
    assets_path = os.environ["GIBSON_ASSETS_PATH"]
else:
    assets_path = global_config["assets_path"]
assets_path = os.path.expanduser(assets_path)

if "GIBSON_DATASET_PATH" in os.environ:
    g_dataset_path = os.environ["GIBSON_DATASET_PATH"]
else:
    g_dataset_path = global_config["g_dataset_path"]
g_dataset_path = os.path.expanduser(g_dataset_path)

if "OMNIGIBSON_DATASET_PATH" in os.environ:
    og_dataset_path = os.environ["OMNIGIBSON_DATASET_PATH"]
else:
    og_dataset_path = global_config["og_dataset_path"]
og_dataset_path = os.path.expanduser(og_dataset_path)

if "KEY_PATH" in os.environ:
    key_path = os.environ["KEY_PATH"]
else:
    key_path = global_config["key_path"]
key_path = os.path.expanduser(key_path)

root_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.isabs(assets_path):
    assets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), assets_path)
if not os.path.isabs(g_dataset_path):
    g_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), g_dataset_path)
if not os.path.isabs(og_dataset_path):
    og_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), og_dataset_path)
if not os.path.isabs(key_path):
    key_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), key_path)

logging.info("Importing OmniGibson (omnigibson module)")
logging.info("Assets path: {}".format(assets_path))
logging.info("Gibson Dataset path: {}".format(g_dataset_path))
logging.info("OmniGibson Dataset path: {}".format(og_dataset_path))
logging.info("OmniGibson Key path: {}".format(key_path))

example_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples")
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

logging.info("Example path: {}".format(example_path))
logging.info("Example config path: {}".format(example_config_path))

# whether to enable debugging mode for object sampling
debug_sampling = False

# Finally, we must create the omnigibson application
from omnigibson.app_omni import OmniApp

# Create app as a global reference so any submodule can access it
app = OmniApp(
    {
        "headless": gm.HEADLESS,
    },
    debug=gm.DEBUG,
)

# Next import must be simulator
sim = None
from omnigibson.simulator import Simulator

# Create simulator (this is a singleton so it's okay that it's global)
sim = Simulator()

import omni
def print_save_usd_warning(_):
    logging.warning("Exporting individual USDs has been disabled in OG due to copyrights.")

omni.kit.widget.stage.context_menu.ContextMenu.save_prim = print_save_usd_warning

# Import any remaining items we want to access directly from the main omnigibson import
from omnigibson.envs import Environment
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.sensors import ALL_SENSOR_MODALITIES
from omnigibson.utils.asset_utils import download_demo_data, download_og_dataset, download_assets
from omnigibson.utils.ui_utils import choose_from_options


def setup():
    """
    Helper function to setup this OmniGibson repository. Configures environment and downloads assets
    """
    # Ask user which dataset to install
    print("Welcome to OmniGibson!")
    print()
    print("Downloading dataset...")
    dataset_options = {
        "Demo": "Download the demo OmniGibson dataset",
        "Full": "Download the full OmniGibson dataset",
    }
    dataset = choose_from_options(options=dataset_options, name="dataset")
    if dataset == "Demo":
        download_demo_data()
    else:
        download_og_dataset()

    print("Downloading assets...")
    download_assets()

    print("\nOmniGibson setup completed!\n")


# Define convenience function for shutting down OmniGibson cleanly
def shutdown():
    app.close()
    exit(0)
