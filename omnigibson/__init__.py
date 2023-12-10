import logging
import os
import socket
import shutil
import tempfile
import atexit
import signal
import yaml
import builtins
from termcolor import colored


# TODO: Need to fix somehow -- omnigibson gets imported first BEFORE we can actually modify the macros
from omnigibson.macros import gm

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

# can override assets_path and dataset_path from environment variable
if "OMNIGIBSON_ASSET_PATH" in os.environ:
    gm.ASSET_PATH = os.environ["OMNIGIBSON_ASSET_PATH"]
gm.ASSET_PATH = os.path.expanduser(gm.ASSET_PATH)
if not os.path.isabs(gm.ASSET_PATH):
    gm.ASSET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), gm.ASSET_PATH)

if "OMNIGIBSON_DATASET_PATH" in os.environ:
    gm.DATASET_PATH = os.environ["OMNIGIBSON_DATASET_PATH"]
gm.DATASET_PATH = os.path.expanduser(gm.DATASET_PATH)
if not os.path.isabs(gm.DATASET_PATH):
    gm.DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), gm.DATASET_PATH)

if "OMNIGIBSON_KEY_PATH" in os.environ:
    gm.KEY_PATH = os.environ["OMNIGIBSON_KEY_PATH"]
gm.KEY_PATH = os.path.expanduser(gm.KEY_PATH)
if not os.path.isabs(gm.KEY_PATH):
    gm.KEY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), gm.KEY_PATH)

root_path = os.path.dirname(os.path.realpath(__file__))

# Store paths to example configs
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# Initialize global variables
app = None  # (this is a singleton so it's okay that it's global)
sim = None  # (this is a singleton so it's okay that it's global)
Environment = None
REGISTERED_SCENES = None
REGISTERED_OBJECTS = None
REGISTERED_ROBOTS = None
REGISTERED_CONTROLLERS = None
REGISTERED_TASKS = None
ALL_SENSOR_MODALITIES = None


# Helper functions for starting omnigibson
def print_save_usd_warning(_):
    log.warning("Exporting individual USDs has been disabled in OG due to copyrights.")


def create_app():
    global app
    from omni.isaac.kit import SimulationApp
    app = SimulationApp({"headless": gm.HEADLESS or bool(gm.REMOTE_STREAMING)})

    # Default Livestream settings
    if gm.REMOTE_STREAMING:
        app.set_setting("/app/window/drawMouse", True)
        app.set_setting("/app/livestream/proto", "ws")
        app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        app.set_setting("/ngx/enabled", False)

        from omni.isaac.core.utils.extensions import enable_extension

        hostname = socket.gethostname()

        # Note: Only one livestream extension can be enabled at a time
        if gm.REMOTE_STREAMING == "native":
            # Enable Native Livestream extension
            # Default App: Streaming Client from the Omniverse Launcher
            enable_extension("omni.kit.livestream.native")
            print(f"Now streaming on {hostname} via Omniverse Streaming Client\n")
        elif gm.REMOTE_STREAMING == "websocket":
            # Enable WebSocket Livestream extension
            # Default URL: http://localhost:8211/streaming/client/
            enable_extension("omni.services.streamclient.websocket")
            print(f"Now streaming on: http://{hostname}:8211/streaming/client\n")
        elif gm.REMOTE_STREAMING == "webrtc":
            # Enable WebRTC Livestream extension
            # Default URL: http://localhost:8211/streaming/webrtc-client/
            enable_extension("omni.services.streamclient.webrtc")
            print(f"Now streaming on: http://{hostname}:8211/streaming/webrtc-client?server={hostname}\n")
        else:
            raise ValueError(f"Invalid REMOTE_STREAMING option {gm.REMOTE_STREAMING}. Must be one of None, native, websocket, webrtc.")

    # If multi_gpu is used, og.sim.render() will cause a segfault when called during on_contact callbacks,
    # e.g. when an attachment joint is being created due to contacts (create_joint calls og.sim.render() internally).
    gpu_id = None if gm.GPU_ID is None else int(gm.GPU_ID)
    config_kwargs = {"headless":  gm.HEADLESS, "multi_gpu": False}
    if gpu_id is not None:
        config_kwargs["active_gpu"] = gpu_id
        config_kwargs["physics_gpu"] = gpu_id
    app = SimulationApp(config_kwargs)

    # Omni overrides the global logger to be DEBUG, which is very annoying, so we re-override it to the default WARN
    # TODO: Remove this once omniverse fixes it
    logging.getLogger().setLevel(logging.WARNING)

    import omni

    # Enable additional extensions we need
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.flowusd")
    enable_extension("omni.particle.system.bundle")

    # Additional import for windows
    if os.name == "nt":
        enable_extension("omni.kit.window.viewport")

    # If we're headless, suppress all warnings about GLFW
    if gm.HEADLESS:
        import omni.log
        log = omni.log.get_log()
        log.set_channel_enabled("carb.windowing-glfw.plugin", False, omni.log.SettingBehavior.OVERRIDE)
        
    # Globally suppress certain logging modules (unless we're in debug mode) since they produce spurious warnings
    if not gm.DEBUG:
        import omni.log
        log = omni.log.get_log()
        for channel in ["omni.hydra.scene_delegate.plugin", "omni.kit.manipulator.prim.model"]:
            log.set_channel_enabled(channel, False, omni.log.SettingBehavior.OVERRIDE)

    # Possibly hide windows if in debug mode
    if gm.GUI_VIEWPORT_ONLY:
        hide_window_names = ["Console", "Main ToolBar", "Stage", "Layer", "Property", "Render Settings", "Content",
                             "Flow", "Semantics Schema Editor"]
        for name in hide_window_names:
            window = omni.ui.Workspace.get_window(name)
            if window is not None:
                window.visible = False
                app.update()

    omni.kit.widget.stage.context_menu.ContextMenu.save_prim = print_save_usd_warning

    return app


def create_sim():
    global sim
    from omnigibson.simulator import Simulator
    sim = Simulator()
    return sim


def print_icon():
    raw_texts = [
        # Lgrey, grey, lgrey, grey, red, lgrey, red
        ("                   ___________", "", "", "", "", "", "_"),
        ("                  /          ", "", "", "", "", "", "/ \\"),
        ("                 /          ", "", "", "", "/ /", "__", ""),
        ("                /          ", "", "", "", "", "", "/ /  /\\"),
        ("               /", "__________", "", "", "/ /", "__", "/  \\"),
        ("               ", "\\   _____  ", "", "", "\\ \\", "__", "\\  /"),
        ("                ", "\\  \\  ", "/ ", "\\  ", "", "", "\\ \\_/ /"),
        ("                 ", "\\  \\", "/", "___\\  ", "", "", "\\   /"),
        ("                  ", "\\__________", "", "", "", "", "\\_/  "),
    ]
    for (lgrey_text0, grey_text0, lgrey_text1, grey_text1, red_text0, lgrey_text2, red_text1) in raw_texts:
        lgrey_text0 = colored(lgrey_text0, "light_grey", attrs=["bold"])
        grey_text0 = colored(grey_text0, "light_grey", attrs=["bold", "dark"])
        lgrey_text1 = colored(lgrey_text1, "light_grey", attrs=["bold"])
        grey_text1 = colored(grey_text1, "light_grey", attrs=["bold", "dark"])
        red_text0 = colored(red_text0, "light_red", attrs=["bold"])
        lgrey_text2 = colored(lgrey_text2, "light_grey", attrs=["bold"])
        red_text1 = colored(red_text1, "light_red", attrs=["bold"])
        print(lgrey_text0 + grey_text0 + lgrey_text1 + grey_text1 + red_text0 + lgrey_text2 + red_text1)


def print_logo():
    raw_texts = [
        ("       ___                  _", "  ____ _ _                     "),
        ("      / _ \ _ __ ___  _ __ (_)", "/ ___(_) |__  ___  ___  _ __  "),
        ("     | | | | '_ ` _ \| '_ \| |", " |  _| | '_ \/ __|/ _ \| '_ \ "),
        ("     | |_| | | | | | | | | | |", " |_| | | |_) \__ \ (_) | | | |"),
        ("      \___/|_| |_| |_|_| |_|_|", "\____|_|_.__/|___/\___/|_| |_|"),
    ]
    for (grey_text, red_text) in raw_texts:
        grey_text = colored(grey_text, "light_grey", attrs=["bold", "dark"])
        red_text = colored(red_text, "light_red", attrs=["bold"])
        print(grey_text + red_text)


def logo_small():
    grey_text = colored("Omni", "light_grey", attrs=["bold", "dark"])
    red_text = colored("Gibson", "light_red", attrs=["bold"])
    return grey_text + red_text


def start():
    global app, sim, Environment, REGISTERED_SCENES, REGISTERED_OBJECTS, REGISTERED_ROBOTS, REGISTERED_CONTROLLERS, \
        REGISTERED_TASKS, ALL_SENSOR_MODALITIES

    log.info(f"{'-' * 10} Starting {logo_small()} {'-' * 10}")

    # First create the app, then create the sim
    app = create_app()
    sim = create_sim()

    print()
    print_icon()
    print_logo()
    print()
    log.info(f"{'-' * 10} Welcome to {logo_small()}! {'-' * 10}")

    # Import any remaining items we want to access directly from the main omnigibson import
    from omnigibson.envs import Environment
    from omnigibson.scenes import REGISTERED_SCENES
    from omnigibson.objects import REGISTERED_OBJECTS
    from omnigibson.robots import REGISTERED_ROBOTS
    from omnigibson.controllers import REGISTERED_CONTROLLERS
    from omnigibson.tasks import REGISTERED_TASKS
    from omnigibson.sensors import ALL_SENSOR_MODALITIES
    return app, sim, Environment, REGISTERED_SCENES, REGISTERED_OBJECTS, REGISTERED_ROBOTS, REGISTERED_CONTROLLERS, \
        REGISTERED_TASKS, ALL_SENSOR_MODALITIES


# Automatically start omnigibson's omniverse backend unless explicitly told not to
OMNIGIBSON_NO_OMNIVERSE = (os.getenv("OMNIGIBSON_NO_OMNIVERSE", 'False').lower() in {'true', '1', 't'})
if not OMNIGIBSON_NO_OMNIVERSE:
    app, sim, Environment, REGISTERED_SCENES, REGISTERED_OBJECTS, REGISTERED_ROBOTS, REGISTERED_CONTROLLERS, \
        REGISTERED_TASKS, ALL_SENSOR_MODALITIES = start()

    # Create and expose a temporary directory for any use cases. It will get destroyed upon omni
    # shutdown by the shutdown function.
    tempdir = tempfile.mkdtemp()

def shutdown():
    if not OMNIGIBSON_NO_OMNIVERSE:
        global app
        global sim
        sim.clear()
        # TODO: Currently tempfile removal will fail due to CopyPrim command (for example, GranularSystem in dicing_apple example.)
        try:
            shutil.rmtree(tempdir)
        except PermissionError:
            log.info("Permission error when removing temp files. Ignoring")
        from omnigibson.utils.ui_utils import suppress_omni_log
        log.info(f"{'-' * 10} Shutting Down {logo_small()} {'-' * 10}")

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
