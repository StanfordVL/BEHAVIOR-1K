import logging
import os
import shutil
import tempfile
import atexit
import signal
import builtins

# TODO: Need to fix somehow -- omnigibson gets imported first BEFORE we can actually modify the macros
from omnigibson.macros import gm

from omnigibson.envs import Environment
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.sensors import ALL_SENSOR_MODALITIES

# Create logger
logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s')
log = logging.getLogger(__name__)

builtins.ISAAC_LAUNCHED_FROM_JUPYTER = (
    os.getenv("ISAAC_JUPYTER_KERNEL") is not None
)  # We set this in the kernel.json file

# Always enable nest_asyncio because MaterialPrim calls asyncio.run()
import nest_asyncio
nest_asyncio.apply()

__version__ = "0.2.1"

log.setLevel(logging.DEBUG if gm.DEBUG else logging.INFO)

root_path = os.path.dirname(os.path.realpath(__file__))

# Store paths to example configs
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# Initialize global variables
app = None  # (this is a singleton so it's okay that it's global)
sim = None  # (this is a singleton so it's okay that it's global)


# Create and expose a temporary directory for any use cases. It will get destroyed upon omni
# shutdown by the shutdown function.
tempdir = tempfile.mkdtemp()

def shutdown():
    global app
    global sim
    sim.clear()
    # TODO: Currently tempfile removal will fail due to CopyPrim command (for example, GranularSystem in dicing_apple example.)
    try:
        shutil.rmtree(tempdir)
    except PermissionError:
        log.info("Permission error when removing temp files. Ignoring")
    from omnigibson.utils.ui_utils import suppress_omni_log
    log.info(f"{'-' * 10} Shutting Down OmniGibson {'-' * 10}")

    # Suppress carb warning here that we have no control over -- it's expected
    with suppress_omni_log(channels=["carb"]):
        app.close()

    exit(0)

# register signal handler for CTRL + C
def signal_handler(signal, frame):
    shutdown()
signal.signal(signal.SIGINT, signal_handler)

# register handler so that we always shut omiverse down correctly upon termination
atexit.register(shutdown)
