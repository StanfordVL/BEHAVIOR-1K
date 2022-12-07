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

__version__ = "3.0.0"

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

if "IGIBSON_DATASET_PATH" in os.environ:
    og_dataset_path = os.environ["IGIBSON_DATASET_PATH"]
else:
    og_dataset_path = global_config["og_dataset_path"]
og_dataset_path = os.path.expanduser(og_dataset_path)

if "3DFRONT_DATASET_PATH" in os.environ:
    threedfront_dataset_path = os.environ["3DFRONT_DATASET_PATH"]
else:
    threedfront_dataset_path = global_config["threedfront_dataset_path"]
threedfront_dataset_path = os.path.expanduser(threedfront_dataset_path)

if "CUBICASA_DATASET_PATH" in os.environ:
    cubicasa_dataset_path = os.environ["CUBICASA_DATASET_PATH"]
else:
    cubicasa_dataset_path = global_config["cubicasa_dataset_path"]
cubicasa_dataset_path = os.path.expanduser(cubicasa_dataset_path)

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
if not os.path.isabs(threedfront_dataset_path):
    threedfront_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), threedfront_dataset_path)
if not os.path.isabs(cubicasa_dataset_path):
    cubicasa_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cubicasa_dataset_path)
if not os.path.isabs(key_path):
    key_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), key_path)

logging.info("Importing OmniGibson (omnigibson module)")
logging.info("Assets path: {}".format(assets_path))
logging.info("Gibson Dataset path: {}".format(g_dataset_path))
logging.info("iG Dataset path: {}".format(og_dataset_path))
logging.info("3D-FRONT Dataset path: {}".format(threedfront_dataset_path))
logging.info("CubiCasa5K Dataset path: {}".format(cubicasa_dataset_path))
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

# Define convenience function for shutting down OmniGibson cleanly
def shutdown():
    app.close()
    exit(0)
