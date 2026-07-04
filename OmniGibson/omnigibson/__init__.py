import builtins
import logging
import os
import shutil
import signal
import tempfile

from omnigibson.envs import Environment, VectorEnvironment
from omnigibson.scenes import REGISTERED_SCENES


ALL_SENSOR_MODALITIES = set()
REGISTERED_CONTROLLERS = {}
REGISTERED_OBJECTS = {}
REGISTERED_ROBOTS = []
REGISTERED_TASKS = {}
gm = None
UNSUPPORTED_LEGACY_SUBSYSTEMS = {
    "action_primitives",
    "data_wrappers",
    "object_states",
    "particles",
    "policy_training",
    "sensors",
    "transition_rules",
}

from omnigibson.simulator import _launch_simulator as launch  # noqa: E402


def require_supported_subsystem(name):
    if name in UNSUPPORTED_LEGACY_SUBSYSTEMS:
        raise NotImplementedError(f"{name} is not implemented in the Newton-first OmniGibson runtime yet.")


# Create logger
RESET = "\033[0m"


class LogFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[32m",  # green
        logging.INFO: RESET,
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[31m" + "\033[1m",  # bold red
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, RESET)
        total_seconds = record.relativeCreated / 1000.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        # Format as HH:MM:SS.sss, with leading zeros
        record.relativeCreated_hms = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return f"{color}{super().format(record)}{RESET}"


formatter = LogFormatter("[%(relativeCreated_hms)s] [%(levelname)s] [%(name)s] %(message)s")
_og_handler = logging.StreamHandler()
_og_handler.setFormatter(formatter)
_og_logger = logging.getLogger("omnigibson")
_og_logger.addHandler(_og_handler)
_og_logger.propagate = False  # prevent Isaac Sim's Carbonite handler from double-printing
log = logging.getLogger(__name__)

builtins.OMNIGIBSON_NEWTON_NATIVE = True

__version__ = "3.9.0"

root_path = os.path.dirname(os.path.realpath(__file__))

# Store paths to example configs
# TODO: Move this elsewhere.
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# Initialize global variables
app = None  # (this is a singleton so it's okay that it's global)
sim = None  # (this is a singleton so it's okay that it's global)


# Create and expose a temporary directory for any use cases. It will get destroyed upon omni
# shutdown by the shutdown function.
tempdir = tempfile.mkdtemp()


def clear(
    gravity=None,
    physics_dt=None,
    rendering_dt=None,
    sim_step_dt=None,
    viewer_width=None,
    viewer_height=None,
    device=None,
):
    """Clear the current simulator and launch a fresh one."""
    global sim

    if sim is not None:
        sim.close()
        sim = None
    return launch()


def _close_simulator():
    global sim
    if sim is not None:
        sim.close()
        sim = None


def cleanup(*args, **kwargs):
    # TODO: Currently tempfile removal will fail due to CopyPrim command (for example, GranularSystem in dicing_apple example.)
    try:
        shutil.rmtree(tempdir)
    except PermissionError:
        log.info("Permission error when removing temp files. Ignoring")

    from omnigibson.simulator import logo_small

    log.info(f"{'-' * 10} Shutting Down {logo_small()} {'-' * 10}")


def shutdown(due_to_signal=False):
    global sim
    if app is not None:
        # If Isaac is running, we do the cleanup in its shutdown callback to avoid open handles.
        # TODO: Automated cleanup in callback doesn't work for some reason. Need to investigate.
        # Manually call cleanup for now.
        cleanup()
        _close_simulator()
        app.close()
    else:
        # Otherwise, we do the cleanup here.
        cleanup()
        _close_simulator()

        # If we're not shutting down due to a signal, we need to manually exit
        if not due_to_signal:
            exit(0)


def shutdown_handler(*args, **kwargs):
    shutdown(due_to_signal=True)
    return signal.default_int_handler(*args, **kwargs)


# Something somewhere disables the default SIGINT handler, so we need to re-enable it
signal.signal(signal.SIGINT, shutdown_handler)

__all__ = [
    "ALL_SENSOR_MODALITIES",
    "app",
    "cleanup",
    "clear",
    "Environment",
    "example_config_path",
    "gm",
    "launch",
    "log",
    "REGISTERED_CONTROLLERS",
    "REGISTERED_OBJECTS",
    "REGISTERED_ROBOTS",
    "REGISTERED_SCENES",
    "REGISTERED_TASKS",
    "require_supported_subsystem",
    "shutdown",
    "sim",
    "tempdir",
    "UNSUPPORTED_LEGACY_SUBSYSTEMS",
    "VectorEnvironment",
]
