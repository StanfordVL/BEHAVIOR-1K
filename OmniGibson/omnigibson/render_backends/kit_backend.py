"""
Isaac Sim / Omniverse Kit implementation of the RenderBackend interface.

This module owns everything about having a live Omniverse Kit application: launching it
(``launch_kit_app()``) and, once it exists, driving its rendering (Replicator, RTX, the viewport
system) behind the RenderBackend interface. Nothing outside this module needs to know Kit exists.

Kit is launched by ``PhysXBackend.create_app()``, since PhysX cannot run without it. PhysX then
owns the render tick itself (``PhysXBackend.render()`` -> ``SimulationContext.render()``), so this
class's own ``render()`` is only invoked by a physics backend that does not.
"""

import contextlib
import logging
import os
import shutil
import signal
import socket
import sys
import tempfile
import traceback
from contextlib import nullcontext
from pathlib import Path

import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.render_backends.base import RenderBackend
from omnigibson.utils.constants import LightingMode
from omnigibson.utils.ui_utils import create_module_logger, logo_small

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Maps each supported (major, minor, patch) Isaac Sim version to its `.kit` experience file.
m.KIT_FILES = {
    (5, 1, 0): "omnigibson_5_1_0.kit",
}


@contextlib.contextmanager
def suppress_kit_log(channels):
    """
    A context scope for temporarily suppressing logging for certain Kit channels.

    Args:
        channels (None or list of str): Logging channel(s) to suppress. If None, will globally disable logger
    """
    # Record the state to restore to after the context exists
    log = lazy.omni.log.get_log()

    if gm.DEBUG:
        # Do nothing
        pass
    elif channels is None:
        # Globally disable log
        log.enabled = False
    else:
        # For some reason, all enabled states always return False even if the logging is clearly enabled for the
        # given channel, so we assume all channels are enabled
        # We do, however, check what behavior was assigned to this channel, since we force an override during this context
        channel_behavior = {channel: log.get_channel_enabled(channel)[2] for channel in channels}

        # Suppress the channels
        for channel in channels:
            log.set_channel_enabled(channel, False, lazy.omni.log.SettingBehavior.OVERRIDE)

    yield

    if gm.DEBUG:
        # Do nothing
        pass
    elif channels is None:
        # Globally re-enable log
        log.enabled = True
    else:
        # Unsuppress the channels
        for channel in channels:
            log.set_channel_enabled(channel, True, channel_behavior[channel])


# Helper functions for starting omnigibson
def print_save_usd_warning(_):
    log.warning("Exporting individual USDs has been disabled in OG due to copyrights.")


class SuppressLogsUntilError:
    """
    Suppress stdout/stderr logs until an error occurs, at which point dump everything.
    """

    def __init__(self, _):
        self._old_stdout = None
        self._old_stderr = None
        self._tmpfile = None
        self._tmppath = None
        self._running = False

    def __enter__(self):
        # Temp file to buffer logs
        self._tmpfile = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        self._tmppath = self._tmpfile.name
        self._tmpfile.close()

        # Save original fds
        sys.stdout.flush()
        sys.stderr.flush()
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)

        # Redirect stdout/stderr → temp file
        fd = os.open(self._tmppath, os.O_WRONLY | os.O_APPEND)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)

        # Start background reader
        self._running = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop background reader
        self._running = False

        # Restore stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self._old_stdout)
        os.close(self._old_stderr)

        # On error → dump everything + traceback
        if exc_type is not None:
            print("\n=== Isaac Sim logs (dump on error) ===\n")
            with open(self._tmppath, "r") as f:
                print(f.read())
            print("=== End of Isaac Sim logs ===\n")

            print("Python traceback:\n")
            traceback.print_exception(exc_type, exc_val, exc_tb)

        # Cleanup
        try:
            os.remove(self._tmppath)
        except OSError:
            pass

        return False  # let exception propagate


def launch_kit_app():
    """
    Launches the Isaac Sim / Omniverse Kit application and returns its `SimulationApp`.

    Called from `PhysXBackend.create_app()` / `KitRenderBackend.create_app()`, so a configuration in
    which no backend needs Kit never reaches any of this code.

    Returns:
        isaacsim.SimulationApp: The launched Kit application.
    """
    log.info(f"{'-' * 5} Starting {logo_small()}. This will take 10-30 seconds... {'-' * 5}")

    # If multi_gpu is used, og.sim.render() will cause a segfault when called during on_contact callbacks,
    # e.g. when an attachment joint is being created due to contacts (create_joint calls og.sim.render() internally).
    gpu_id = None if gm.GPU_ID is None else int(gm.GPU_ID)
    config_kwargs = {"headless": gm.HEADLESS or bool(gm.REMOTE_STREAMING), "multi_gpu": False}
    if gpu_id is not None:
        config_kwargs["active_gpu"] = gpu_id
        config_kwargs["physics_gpu"] = gpu_id

    # Clear the argv - Isaac Sim unfortunately reads from it directly, so we need to clear it to avoid issues.
    # Otherwise it will inherit the arguments of the entrypoint script.
    _saved_argv = sys.argv[:]
    try:
        sys.argv = [
            _saved_argv[0]
        ]  # The script filename needs to be included - otherwise the first arg will get skipped.

        # Omni's logging is super annoying and overly verbose, so suppress it by modifying the logging levels
        if not gm.DEBUG:
            import warnings

            try:
                from numba.core.errors import NumbaPerformanceWarning

                warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            except ImportError:
                pass

            # Find a more elegant way to prune omni logging
            if gm.NO_OMNI_LOGS:
                sys.argv.append("--/log/level=error")
                sys.argv.append("--/log/fileLogLevel=error")
                sys.argv.append("--/log/outputStreamLevel=error")

        # Try to import the isaacsim module that only shows up in Isaac Sim 4.0.0. This ensures that
        # if we are using the pip installed version, all the ISAAC_PATH etc. env vars are set correctly.
        # On the regular omniverse launcher version this should not have any impact.
        try:
            os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
            import isaacsim  # noqa: F401
        except ImportError:
            pass

        # First obtain the Isaac Sim version
        isaac_path = os.environ["ISAAC_PATH"]
        version_file_path = os.path.join(isaac_path, "VERSION")
        assert os.path.exists(version_file_path), f"Isaac Sim version file not found at {version_file_path}"
        with open(version_file_path, "r") as file:
            version_content = file.read().strip()
            isaac_version_str = version_content.split("-")[0]
            isaac_version_tuple = tuple(map(int, isaac_version_str.split(".")[:3]))
            assert isaac_version_tuple in m.KIT_FILES, f"Isaac Sim version must be one of {list(m.KIT_FILES.keys())}"
            kit_file_name = m.KIT_FILES[isaac_version_tuple]
            if gm.ENABLE_VR:
                kit_file_name = kit_file_name.replace(".kit", "_vr.kit")

        # Copy the OmniGibson kit file and icon file to the Isaac Sim apps directory. This is necessary because the Isaac Sim app
        # expects the extensions to be reachable in the parent directory of the kit file. We copy on every launch to
        # ensure that the kit file is always up to date.
        assert (
            "EXP_PATH" in os.environ
        ), "The EXP_PATH variable is not set. Are you in an Isaac Sim installed environment?"
        exp_path = os.environ["EXP_PATH"]
        kit_file = Path(__file__).parents[1] / kit_file_name
        kit_file_target = Path(exp_path) / kit_file_name
        icon_file = Path(__file__).parents[3] / "docs" / "assets" / "OmniGibson_logo.png"
        icon_file_target = Path(exp_path) / "OmniGibson_logo.png"

        try:
            shutil.copyfile(kit_file, kit_file_target)
            shutil.copyfile(icon_file, icon_file_target)
        except Exception as e:
            raise e from ValueError(f"Failed to copy {kit_file_name} or {icon_file.name} to Isaac Sim apps directory.")

        # Set the MDL search path so that our OmniGibsonVrayMtl can be found.
        os.environ["MDL_USER_PATH"] = str((Path(__file__).parents[1] / "materials").resolve())

        launch_context = nullcontext if gm.DEBUG else SuppressLogsUntilError if gm.NO_OMNI_LOGS else suppress_kit_log

        # Prepare the directories where Omniverse will store its appdata (logs, caches, etc.)
        local_appdata = Path(gm.APPDATA_PATH) / "local"
        local_appdata.mkdir(parents=True, exist_ok=True)
        sys.argv.extend(["--portable-root", str(local_appdata)])

        global_cache_dir = Path(gm.APPDATA_PATH) / "global" / "cache"
        global_cache_dir.mkdir(parents=True, exist_ok=True)
        sys.argv.append(f"--/app/tokens/omni_global_cache={global_cache_dir}")

        global_data_dir = Path(gm.APPDATA_PATH) / "global" / "data"
        global_data_dir.mkdir(parents=True, exist_ok=True)
        sys.argv.append(f"--/app/tokens/omni_global_data={str(global_data_dir)}")

        # Persist warp's JIT-compiled kernel cache under gm.APPDATA_PATH so it survives
        # across runs (warp's default ~/.cache/warp is per-user/ephemeral on some
        # self-hosted CI runners, which means every run pays the full NVRTC JIT cost).
        global_warp_cache_dir = Path(gm.APPDATA_PATH) / "global" / "warp_cache"
        global_warp_cache_dir.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(global_warp_cache_dir)

        with launch_context(None):
            app = lazy.isaacsim.SimulationApp(config_kwargs, experience=str(kit_file_target.resolve(strict=True)))
    finally:
        # Always restore the caller's argv, even if Isaac Sim startup raises.
        sys.argv = _saved_argv

    # Close the stage so that we can create a new one when a Simulator Instance is created
    assert lazy.isaacsim.core.utils.stage.close_stage()

    # Omni overrides the global logger to be DEBUG, which is very annoying, so we re-override it to the default WARN
    # TODO: Remove this once omniverse fixes it
    logging.getLogger().setLevel(logging.WARNING)

    # Default Livestream settings
    if gm.REMOTE_STREAMING:
        app.set_setting("/app/window/drawMouse", True)
        app.set_setting("/app/livestream/proto", "ws")
        app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        app.set_setting("/ngx/enabled", False)

        # Find our IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()

        # Note: Only one livestream extension can be enabled at a time
        if gm.REMOTE_STREAMING == "native":
            # Enable Native Livestream extension
            # Default App: Streaming Client from the Omniverse Launcher
            lazy.isaacsim.core.utils.extensions.enable_extension("omni.kit.livestream.native")
            print(f"Now streaming on {ip} via Omniverse Streaming Client")
        elif gm.REMOTE_STREAMING == "webrtc":
            # Enable WebRTC Livestream extension
            app.set_setting("/exts/omni.services.transport.server.http/port", gm.HTTP_PORT)
            app.set_setting("/app/livestream/port", gm.WEBRTC_PORT)
            lazy.isaacsim.core.utils.extensions.enable_extension("omni.services.streamclient.webrtc")
            print(f"Now streaming on: http://{ip}:{gm.HTTP_PORT}/streaming/webrtc-client?server={ip}")
        else:
            raise ValueError(
                f"Invalid REMOTE_STREAMING option {gm.REMOTE_STREAMING}. Must be one of None, native, webrtc."
            )

    # If we're headless, suppress all warnings about GLFW
    if gm.HEADLESS:
        og_log = lazy.omni.log.get_log()
        og_log.set_channel_enabled("carb.windowing-glfw.plugin", False, lazy.omni.log.SettingBehavior.OVERRIDE)

    # Globally suppress certain logging modules (unless we're in debug mode) since they produce spurious warnings
    if not gm.DEBUG:
        og_log = lazy.omni.log.get_log()
        for channel in ["omni.hydra.scene_delegate.plugin", "omni.kit.manipulator.prim.model"]:
            og_log.set_channel_enabled(channel, False, lazy.omni.log.SettingBehavior.OVERRIDE)

    # Possibly hide windows if in debug mode
    hide_window_names = []
    if not gm.RENDER_VIEWER_CAMERA:
        hide_window_names.append("Viewport")
    if gm.GUI_VIEWPORT_ONLY:
        hide_window_names.extend(
            [
                "Console",
                "Main ToolBar",
                "Stage",
                "Layer",
                "Property",
                "Render Settings",
                "Content",
                "Flow",
                "Semantics Schema Editor",
                "VR",
                "Isaac Sim Assets [Beta]",
            ]
        )

    for name in hide_window_names:
        window = lazy.omni.ui.Workspace.get_window(name)
        if window is not None:
            window.visible = False
            app.update()

    lazy.omni.kit.widget.stage.context_menu.ContextMenu.save_prim = print_save_usd_warning

    # Let the hotkeys propagate.
    app.update()

    # Disable all hotkeys for now. These are not exactly helpful and they cause collisions with
    # the OmniGibson-provided hotkeys.
    hotkey_registry = lazy.omni.kit.hotkeys.core.get_hotkey_registry()
    for hotkey in list(hotkey_registry.get_all_hotkeys()):
        hotkey_registry.deregister_hotkey(hotkey)

    # TODO: Automated cleanup in callback doesn't work for some reason. Need to investigate.
    shutdown_stream = lazy.omni.kit.app.get_app().get_shutdown_event_stream()
    shutdown_stream.create_subscription_to_pop(og.cleanup, name="og_cleanup", order=0)

    # Loading Isaac Sim disables Ctrl+C, so we need to re-enable it
    signal.signal(signal.SIGINT, og.shutdown_handler)

    return app


class KitRenderBackend(RenderBackend):
    """
    Kit-backed render backend. Needs a live Isaac Sim / Omniverse Kit application, same as
    ``PhysXBackend`` does on the physics side.
    """

    runs_inside_kit = True
    supports_camera_capture = True
    supports_viewport = True
    supports_materials = True
    supports_bbox = True
    supports_pointcloud = True
    supports_tiled_rendering = True

    def apply_renderer_settings(self):
        self.sim._set_renderer_settings()
        # Set the lighting mode to be stage by default
        self.sim.set_lighting_mode(mode=LightingMode.STAGE)

    def create_fabric_hierarchy(self):
        usdrt_stage = lazy.isaacsim.core.utils.stage.get_current_stage(fabric=True)
        return usdrt_stage, lazy.usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            usdrt_stage.GetFabricId(), usdrt_stage.GetStageIdAsStageId()
        )

    def setup_viewer_camera(self, viewer_width, viewer_height):
        # _set_viewer_camera() also applies the default viewer pose -- see its own implementation.
        self.sim._set_viewer_camera(viewer_width=viewer_width, viewer_height=viewer_height)

    def before_play(self, sim):
        # Kit's RTX renderer only actually draws new frames for a render product while its own
        # timeline is playing (a separate, Kit-global concept from any physics backend's own play/pause
        # state). PhysX's SimulationContext.play() happens to also start the timeline as a side effect,
        # so this is redundant today -- but a physics backend that does not touch Kit's timeline
        # would otherwise leave the render product attached and non-erroring yet permanently blank.
        lazy.omni.timeline.get_timeline_interface().play()

    # ---- Stage / application services ----
    #
    # Kit's own versions of these, which OmniGibson used to call directly whenever it believed a Kit
    # application was present. See RenderBackend's own section comment for why they live on the
    # render backend.

    def create_default_pbr_material(self, scope_path, material_path, target_prim_path):
        from omnigibson.utils.physx_utils import bind_material
        from omnigibson.utils.render_utils import create_pbr_material

        with self.sim.editing_usd():
            self.sim.stage.DefinePrim(scope_path, "Scope")
        create_pbr_material(prim_path=material_path)
        bind_material(prim_path=target_prim_path, material_path=material_path)

    def get_stage(self):
        return lazy.isaacsim.core.utils.stage.get_current_stage()

    def copy_prim(self, source_prim_path, dest_prim_path):
        # Kit's CopyPrim carries the whole subtree, so it covers copy_mesh_prim() too.
        lazy.omni.kit.commands.execute("CopyPrim", path_from=source_prim_path, path_to=dest_prim_path)

    def copy_mesh_prim(self, source_prim_path, dest_prim_path):
        self.copy_prim(source_prim_path, dest_prim_path)

    def deactivate_prim(self, prim):
        lazy.omni.usd.commands.DeletePrimsCommand([str(prim.GetPath())], destructive=False).do()

    def delete_prim(self, prim_path, destructive=True):
        lazy.omni.usd.commands.DeletePrimsCommand([prim_path], destructive=destructive).do()

    def is_prim_ancestral(self, prim):
        return lazy.isaacsim.core.utils.prims.is_prim_ancestral(str(prim.GetPath()))

    def add_reference_to_stage(self, asset_path, prim_path):
        lazy.isaacsim.core.utils.stage.add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        return lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)

    def create_primitive_mesh(self, primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
        MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
            "Sphere": lazy.omni.kit.primitive.mesh.evaluators.sphere.SphereEvaluator,
            "Disk": lazy.omni.kit.primitive.mesh.evaluators.disk.DiskEvaluator,
            "Plane": lazy.omni.kit.primitive.mesh.evaluators.plane.PlaneEvaluator,
            "Cylinder": lazy.omni.kit.primitive.mesh.evaluators.cylinder.CylinderEvaluator,
            "Torus": lazy.omni.kit.primitive.mesh.evaluators.torus.TorusEvaluator,
            "Cone": lazy.omni.kit.primitive.mesh.evaluators.cone.ConeEvaluator,
            "Cube": lazy.omni.kit.primitive.mesh.evaluators.cube.CubeEvaluator,
        }

        evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
        u_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
        v_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
        hs_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)
        stage = self.sim.stage if stage is None else stage

        # Default half_scale (i.e. half-extent, half_height, radius) is 1.
        # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
        #  (currently it only accepts integer-number and floors 0.5 into 0).
        lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)
        kwargs = dict(prim_type=primitive_type, prim_path=prim_path, stage=stage)
        if u_patches is not None and v_patches is not None:
            kwargs["u_patches"] = u_patches
            kwargs["v_patches"] = v_patches

        # Import now to avoid too-eager load of Omni classes due to inheritance
        from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand

        CreateMeshPrimWithDefaultXformCommand(**kwargs).do()

        lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)

    def compute_world_aabb(self, prim_path):
        return lazy.omni.usd.get_context().compute_path_world_bounding_box(prim_path)

    def recompute_extents(self, prim):
        lazy.isaacsim.core.utils.bounds.recompute_extents(prim=prim)

    def add_semantic_labels(self, prim, label, instance_name="class"):
        lazy.isaacsim.core.utils.semantics.upgrade_prim_semantics_to_labels(prim=prim)
        lazy.isaacsim.core.utils.semantics.add_labels(prim=prim, labels=[label], instance_name=instance_name)

    def suppress_log(self, channels=None):
        return suppress_kit_log(channels)

    def render(self):
        lazy.omni.kit.app.get_app().update()

    def create_camera_resource(self, prim_path, resolution, force_new=False):
        return lazy.omni.replicator.core.create.render_product(prim_path, resolution, force_new=force_new)

    def destroy_camera_resource(self, camera_resource):
        camera_resource.destroy()

    def create_annotator(self, raw_modality_name):
        return lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(raw_modality_name)

    def attach_modality(self, annotator, camera_resource):
        annotator.attach([camera_resource])

    def detach_modality(self, annotator, camera_resource):
        try:
            annotator.detach(camera_resource)
        except TypeError:
            # omni.syntheticdata's node deactivation walk is fragile once a tiled render product
            # has existed in the session: cached graph node handles become invalid and detach raises
            # "Invalid NodeObj object". The annotator's nodes are already gone at that point, so we
            # just drop our reference (teardown-only; nothing further to clean up).
            log.warning("Failed to cleanly detach annotator; skipping")

    def get_modality_data(self, annotator, device=None):
        return annotator.get_data(device=device) if device is not None else annotator.get_data()
