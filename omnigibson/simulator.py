import atexit
import contextlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import signal
import socket
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros, gm
from omnigibson.object_states.contact_subscribed_state_mixin import ContactSubscribedStateMixin
from omnigibson.object_states.factory import get_states_by_dependency_order
from omnigibson.object_states.joint_break_subscribed_state_mixin import JointBreakSubscribedStateMixin
from omnigibson.object_states.update_state_mixin import GlobalUpdateStateMixin, UpdateStateMixin
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.objects.light_object import LightObject
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.prims import XFormPrim
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.scenes import Scene
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.constants import LightingMode
from omnigibson.utils.python_utils import Serializable
from omnigibson.utils.python_utils import clear as clear_python_utils
from omnigibson.utils.python_utils import create_object_from_init_info, recursively_convert_to_torch
from omnigibson.utils.ui_utils import (
    CameraMover,
    create_module_logger,
    disclaimer,
    logo_small,
    print_icon,
    print_logo,
    suppress_omni_log,
)
from omnigibson.utils.usd_utils import (
    CollisionAPI,
    ControllableObjectViewAPI,
    FlatcacheAPI,
    GripperRigidContactAPI,
    PoseAPI,
    RigidContactAPI,
)
from omnigibson.utils.usd_utils import clear as clear_usd_utils

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_VIEWER_CAMERA_POS = (-0.201028, -2.72566, 1.0654)
m.DEFAULT_VIEWER_CAMERA_QUAT = (0.68196617, -0.00155408, -0.00166678, 0.73138017)

m.OBJECT_GRAVEYARD_POS = (100.0, 100.0, 100.0)

m.SCENE_MARGIN = 10.0
m.INITIAL_SCENE_PRIM_Z_OFFSET = -100.0

m.KIT_FILES = {
    (4, 0, 0): "omnigibson_4_0_0.kit",
    (4, 1, 0): "omnigibson_4_1_0.kit",
}


# Helper functions for starting omnigibson
def print_save_usd_warning(_):
    log.warning("Exporting individual USDs has been disabled in OG due to copyrights.")


def _launch_app():
    log.setLevel(logging.DEBUG if gm.DEBUG else logging.INFO)

    log.info(f"{'-' * 5} Starting {logo_small()}. This will take 10-30 seconds... {'-' * 5}")

    # If multi_gpu is used, og.sim.render() will cause a segfault when called during on_contact callbacks,
    # e.g. when an attachment joint is being created due to contacts (create_joint calls og.sim.render() internally).
    gpu_id = None if gm.GPU_ID is None else int(gm.GPU_ID)
    config_kwargs = {"headless": gm.HEADLESS or bool(gm.REMOTE_STREAMING), "multi_gpu": False}
    if gpu_id is not None:
        config_kwargs["active_gpu"] = gpu_id
        config_kwargs["physics_gpu"] = gpu_id

    # Omni's logging is super annoying and overly verbose, so suppress it by modifying the logging levels
    if not gm.DEBUG:
        import sys
        import warnings

        try:
            from numba.core.errors import NumbaPerformanceWarning

            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        except ImportError:
            pass

        # TODO: Find a more elegant way to prune omni logging
        # sys.argv.append("--/log/level=warning")
        # sys.argv.append("--/log/fileLogLevel=warning")
        # sys.argv.append("--/log/outputStreamLevel=error")

    # Try to import the isaacsim module that only shows up in Isaac Sim 4.0.0. This ensures that
    # if we are using the pip installed version, all the ISAAC_PATH etc. env vars are set correctly.
    # On the regular omniverse launcher version this should not have any impact.
    try:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        import isaacsim  # noqa: F401
    except ImportError:
        isaacsim = None

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

    # Copy the OmniGibson kit file to the Isaac Sim apps directory. This is necessary because the Isaac Sim app
    # expects the extensions to be reachable in the parent directory of the kit file. We copy on every launch to
    # ensure that the kit file is always up to date.
    assert "EXP_PATH" in os.environ, "The EXP_PATH variable is not set. Are you in an Isaac Sim installed environment?"
    exp_path = os.environ["EXP_PATH"]
    kit_file = Path(__file__).parent / kit_file_name
    kit_file_target = Path(exp_path) / kit_file_name

    try:
        shutil.copy(kit_file, kit_file_target)
    except Exception as e:
        raise e from ValueError(f"Failed to copy {kit_file_name} to Isaac Sim apps directory.")

    launch_context = nullcontext if gm.DEBUG else suppress_omni_log

    with launch_context(None):
        app = lazy.omni.isaac.kit.SimulationApp(config_kwargs, experience=str(kit_file_target.resolve(strict=True)))

    # Close the stage so that we can create a new one when a Simulator Instance is created
    assert lazy.omni.isaac.core.utils.stage.close_stage()

    # Omni overrides the global logger to be DEBUG, which is very annoying, so we re-override it to the default WARN
    # TODO: Remove this once omniverse fixes it
    logging.getLogger().setLevel(logging.WARNING)

    # Enable additional extensions we need
    lazy.omni.isaac.core.utils.extensions.enable_extension("omni.flowusd")
    lazy.omni.isaac.core.utils.extensions.enable_extension("omni.particle.system.bundle")

    # Additional import for windows
    if os.name == "nt":
        lazy.omni.isaac.core.utils.extensions.enable_extension("omni.kit.window.viewport")

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
            lazy.omni.isaac.core.utils.extensions.enable_extension("omni.kit.livestream.native")
            print(f"Now streaming on {ip} via Omniverse Streaming Client")
        elif gm.REMOTE_STREAMING == "webrtc":
            # Enable WebRTC Livestream extension
            app.set_setting("/exts/omni.services.transport.server.http/port", gm.HTTP_PORT)
            app.set_setting("/app/livestream/port", gm.WEBRTC_PORT)
            lazy.omni.isaac.core.utils.extensions.enable_extension("omni.services.streamclient.webrtc")
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
            ]
        )

    for name in hide_window_names:
        window = lazy.omni.ui.Workspace.get_window(name)
        if window is not None:
            window.visible = False
            app.update()

    lazy.omni.kit.widget.stage.context_menu.ContextMenu.save_prim = print_save_usd_warning

    # TODO: Automated cleanup in callback doesn't work for some reason. Need to investigate.
    shutdown_stream = lazy.omni.kit.app.get_app().get_shutdown_event_stream()
    sub = shutdown_stream.create_subscription_to_pop(og.cleanup, name="og_cleanup", order=0)

    # Loading Isaac Sim disables Ctrl+C, so we need to re-enable it
    signal.signal(signal.SIGINT, og.shutdown_handler)

    return app


def _launch_simulator(*args, **kwargs):
    if not og.app:
        og.app = _launch_app()

    class Simulator(lazy.omni.isaac.core.simulation_context.SimulationContext, Serializable):
        """
        Simulator class for directly interfacing with the physx physics engine.

        NOTE: This is a monolithic class.
            All created Simulator() instances will reference the same underlying Simulator object

        Args:
            gravity (float): gravity on z direction.
            physics_dt (None or float): dt between physics steps. If None, will use default value
                1 / gm.DEFAULT_PHYSICS_FREQ
            rendering_dt (None or float): dt between rendering steps. Note: rendering means rendering a frame of the
                current application and not only rendering a frame to the viewports/ cameras. So UI elements of
                Isaac Sim will be refreshed with this dt as well if running non-headless. If None, will use default
                value 1 / gm.DEFAULT_RENDERING_FREQ
            sim_step_dt (None or float): dt between self.step() calls. This is the amount of simulation time that
                passes every time step() is called. Note: This must be a multiple of @rendering_dt. If None, will
                use default value 1 / gm.DEFAULT_SIM_STEP_FREQ
            viewer_width (int): width of the camera image, in pixels
            viewer_height (int): height of the camera image, in pixels
            device (None or str): specifies the device to be used if running on the gpu with torch backend
        """

        def __init__(
            self,
            gravity=9.81,
            physics_dt=None,
            rendering_dt=None,
            sim_step_dt=None,
            viewer_width=gm.DEFAULT_VIEWER_WIDTH,
            viewer_height=gm.DEFAULT_VIEWER_HEIGHT,
            device=None,
        ):
            assert (
                lazy.omni.isaac.core.utils.stage.get_current_stage() is None
            ), "Stage should not exist when creating a new Simulator instance"

            # Here we assign self as the Simulator instance and as og.sim, because certain functions
            # called downstream during the initialization of this object will try to access og.sim.
            # This makes that possible (and also introduces possible issues around circular dependencies)
            assert og.sim is None, "Only one Simulator instance can be created at a time!"
            og.sim = self

            # Sanity check physics vs. rendering vs. sim step dt
            physics_dt = 1.0 / gm.DEFAULT_PHYSICS_FREQ if physics_dt is None else physics_dt
            rendering_dt = 1.0 / gm.DEFAULT_RENDERING_FREQ if rendering_dt is None else rendering_dt
            sim_step_dt = 1.0 / gm.DEFAULT_SIM_STEP_FREQ if sim_step_dt is None else sim_step_dt
            self._validate_dts(physics_dt, rendering_dt, sim_step_dt)

            # Store vars needed for initialization
            self.gravity = gravity
            self._sim_step_dt = sim_step_dt
            self._n_steps_per_loop = int(sim_step_dt // rendering_dt)
            self._viewer_camera = None
            self._camera_mover = None
            self._render_on_step = True

            self._floor_plane = None
            self._skybox = None
            self._last_scene_edge = None
            self._stage_id = None

            # Run super init
            super().__init__(
                physics_dt=physics_dt,
                rendering_dt=rendering_dt,
                backend="torch",
                device=device,
            )

            # Store other references to variables that will be initialized later
            self._scenes = []
            self._physx_interface = lazy.omni.physx.get_physx_interface()
            self._physx_simulation_interface = lazy.omni.physx.get_physx_simulation_interface()
            self._physx_scene_query_interface = lazy.omni.physx.get_physx_scene_query_interface()
            self._physx_fabric_interface = None
            self._contact_callback = self._physics_context._physx_sim_interface.subscribe_contact_report_events(
                self._on_contact
            )
            self._physics_step_callback = self._physics_context._physx_interface.subscribe_physics_step_events(
                lambda _: self._on_physics_step()
            )
            self._simulation_event_callback = (
                self._physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop(
                    self._on_simulation_event
                )
            )

            # List of objects that need to be initialized during whenever the next sim step occurs
            self._objects_to_initialize = []
            self._objects_require_contact_callback = False
            self._objects_require_joint_break_callback = False

            # Maps callback name to callback
            self._callbacks_on_play = dict()
            self._callbacks_on_stop = dict()
            self._callbacks_on_add_obj = dict()
            self._callbacks_on_remove_obj = dict()
            self._callbacks_on_system_init = dict()
            self._callbacks_on_system_clear = dict()

            # Update internal settings
            self._set_physics_engine_settings()
            self._set_renderer_settings()

            # Set the lighting mode to be stage by default
            self.set_lighting_mode(mode=LightingMode.STAGE)

            # Mapping from link IDs assigned from omni to the object that they reference
            self._link_id_to_objects = dict()

            # Set of categories that can be grasped by assisted grasping
            self.object_state_types = get_states_by_dependency_order()
            self.object_state_types_requiring_update = [
                state
                for state in self.object_state_types
                if (issubclass(state, UpdateStateMixin) or issubclass(state, GlobalUpdateStateMixin))
            ]
            self.object_state_types_on_contact = {
                state for state in self.object_state_types if issubclass(state, ContactSubscribedStateMixin)
            }
            self.object_state_types_on_joint_break = {
                state for state in self.object_state_types if issubclass(state, JointBreakSubscribedStateMixin)
            }

            # Create world prim
            self.stage.DefinePrim("/World", "Xform")

            self.stop()

            for state in self.object_state_types_requiring_update:
                if issubclass(state, GlobalUpdateStateMixin):
                    state.global_initialize()

            # Now start rebuilding everything
            # Create collision group for fixed base objects' non root links, root links, and building structures
            CollisionAPI.create_collision_group(col_group="fixed_base_nonroot_links", filter_self_collisions=False)
            # Disable collision between root links of fixed base objects
            CollisionAPI.create_collision_group(col_group="fixed_base_root_links", filter_self_collisions=True)
            # Disable collision between building structures
            CollisionAPI.create_collision_group(col_group="structures", filter_self_collisions=True)
            # Disable collision between building structures and fixed base objects
            CollisionAPI.add_group_filter(col_group="structures", filter_group="fixed_base_nonroot_links")
            CollisionAPI.add_group_filter(col_group="structures", filter_group="fixed_base_root_links")

            # Store stage ID
            self._stage_id = lazy.pxr.UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()

            # Set the viewer camera, and then set its default pose
            if gm.RENDER_VIEWER_CAMERA:
                self._set_viewer_camera()
                self.viewer_camera.set_position_orientation(
                    position=th.tensor(m.DEFAULT_VIEWER_CAMERA_POS),
                    orientation=th.tensor(m.DEFAULT_VIEWER_CAMERA_QUAT),
                )
                self.viewer_width = viewer_width
                self.viewer_height = viewer_height

        def _set_viewer_camera(self, relative_prim_path="/viewer_camera", viewport_name="Viewport"):
            """
            Creates a camera prim dedicated for this viewer at @prim_path if it doesn't exist,
            and sets this camera as the active camera for the viewer

            Args:
                prim_path (str): Path to check for / create the viewer camera
                viewport_name (str): Name of the viewport this camera should attach to. Default is "Viewport", which is
                    the default viewport's name in Isaac Sim
            """
            self._viewer_camera = VisionSensor(
                relative_prim_path=relative_prim_path,
                name=relative_prim_path.split("/")[-1],  # Assume name is the lowest-level name in the prim_path
                modalities="rgb",
                image_height=self.viewer_height,
                image_width=self.viewer_width,
                viewport_name=viewport_name,
            )
            self._viewer_camera.load(None)

            # We update its clipping range and focal length so we get a good FOV and so that it doesn't clip
            # nearby objects (default min is 1 m)
            self._viewer_camera.clipping_range = [0.001, 10000000.0]
            self._viewer_camera.focal_length = 17.0

            # Initialize the sensor
            self._viewer_camera.initialize()

            # Also need to potentially update our camera mover if it already exists
            if self._camera_mover is not None:
                self._camera_mover.set_cam(cam=self._viewer_camera)

        def _set_physics_engine_settings(self):
            """
            Set the physics engine with specified settings
            """
            assert self.is_stopped(), f"Cannot set simulator physics settings while simulation is playing!"
            self._physics_context.set_gravity(value=-self.gravity)
            # Also make sure we don't invert the collision group filter settings so that different collision groups by
            # default collide with each other, and modify settings for speed optimization
            self._physics_context.set_invert_collision_group_filter(False)
            self._physics_context.enable_ccd(gm.ENABLE_CCD)
            self._physics_context.enable_fabric(gm.ENABLE_FLATCACHE)

            # Enable GPU dynamics based on whether we need omni particles feature
            if gm.USE_GPU_DYNAMICS:
                self._physics_context.enable_gpu_dynamics(True)
                self._physics_context.set_broadphase_type("GPU")
            else:
                self._physics_context.enable_gpu_dynamics(False)
                self._physics_context.set_broadphase_type("MBP")

            # Set GPU Pairs capacity and other GPU settings
            self._physics_context.set_gpu_found_lost_pairs_capacity(gm.GPU_PAIRS_CAPACITY)
            self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(gm.GPU_AGGR_PAIRS_CAPACITY)
            self._physics_context.set_gpu_total_aggregate_pairs_capacity(gm.GPU_AGGR_PAIRS_CAPACITY)
            self._physics_context.set_gpu_max_particle_contacts(gm.GPU_MAX_PARTICLE_CONTACTS)
            self._physics_context.set_gpu_max_rigid_contact_count(gm.GPU_MAX_RIGID_CONTACT_COUNT)
            self._physics_context.set_gpu_max_rigid_patch_count(gm.GPU_MAX_RIGID_PATCH_COUNT)

        def _set_renderer_settings(self):
            if gm.ENABLE_HQ_RENDERING:
                lazy.carb.settings.get_settings().set_bool("/rtx/reflections/enabled", True)
                lazy.carb.settings.get_settings().set_bool("/rtx/indirectDiffuse/enabled", True)
                lazy.carb.settings.get_settings().set_int("/rtx/post/dlss/execMode", 3)  # "Auto"
                lazy.carb.settings.get_settings().set_bool("/rtx/ambientOcclusion/enabled", True)
                lazy.carb.settings.get_settings().set_bool("/rtx/directLighting/sampledLighting/enabled", False)
            else:
                lazy.carb.settings.get_settings().set_bool("/rtx/reflections/enabled", False)
                lazy.carb.settings.get_settings().set_bool("/rtx/indirectDiffuse/enabled", False)
                lazy.carb.settings.get_settings().set_int("/rtx/post/dlss/execMode", 0)  # "Performance"
                lazy.carb.settings.get_settings().set_bool("/rtx/ambientOcclusion/enabled", False)
                lazy.carb.settings.get_settings().set_bool("/rtx/directLighting/sampledLighting/enabled", True)
            lazy.carb.settings.get_settings().set_int("/rtx/raytracing/showLights", 1)
            lazy.carb.settings.get_settings().set_float("/rtx/sceneDb/ambientLightIntensity", 0.1)
            lazy.carb.settings.get_settings().set_bool("/app/renderer/skipMaterialLoading", False)

            # Below settings are for improving performance: we use the USD / Fabric only for poses.
            lazy.carb.settings.get_settings().set_bool("/physics/updateToUsd", not gm.ENABLE_FLATCACHE)
            lazy.carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", False)
            lazy.carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd", False)
            lazy.carb.settings.get_settings().set_bool("/physics/updateForceSensorsToUsd", False)
            lazy.carb.settings.get_settings().set_bool("/physics/updateResidualsToUsd", False)
            lazy.carb.settings.get_settings().set_bool("/physics/outputVelocitiesLocalSpace", False)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUpdateVelocities", False)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUpdateForceSensors", False)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUpdateJointStates", False)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUpdateResiduals", False)
            lazy.carb.settings.get_settings().set_bool("/physics/fabricUseGPUInterop", True)

        def _validate_dts(self, physics_dt, rendering_dt, sim_step_dt):
            """
            Validates that @physics_dt, @rendering_dt, and @sim_step_dt are all valid with respect to each other

            Args:
                physics_dt (float): Physics timestep
                rendering_dt (float): Rendering timestep
                sim_step_dt (float): Simulation step timestep
            """
            render_physics_ratio = rendering_dt / physics_dt
            sim_render_ratio = sim_step_dt / rendering_dt
            assert math.isclose(
                render_physics_ratio, round(render_physics_ratio)
            ), f"Rendering dt ({rendering_dt}) must be a multiple of physics dt ({physics_dt})"
            assert (
                rendering_dt >= physics_dt
            ), f"Rendering dt ({rendering_dt}) cannot be smaller than physics dt ({rendering_dt})"
            assert math.isclose(
                sim_render_ratio, round(sim_render_ratio)
            ), f"Simulation step dt ({sim_step_dt}) must be a multiple of rendering dt ({rendering_dt})"
            assert (
                sim_step_dt >= rendering_dt
            ), f"Simulation step dt ({sim_step_dt}) cannot be smaller than rendering dt ({rendering_dt})"

            # If we're headless, we also enforce that sim_step_dt == rendering_dt because it doesn't make sense
            # to waste rendering that is not observed by the user
            if gm.HEADLESS:
                assert sim_step_dt == rendering_dt, (
                    f"Simulation step dt ({sim_step_dt}) must be equal to rendering dt ({rendering_dt}) when "
                    f"gm.HEADLESS is set!"
                )

        @property
        def viewer_visibility(self):
            """
            Returns:
                bool: Whether the viewer is visible or not
            """
            return self._viewer_camera.viewer_visibility

        @viewer_visibility.setter
        def viewer_visibility(self, visible):
            """
            Sets whether the viewer should be visible or not in the Omni UI

            Args:
                visible (bool): Whether the viewer should be visible or not
            """
            self._viewer_camera.viewer_visibility = visible

        @property
        def viewer_height(self):
            """
            Returns:
                int: viewer height of this sensor, in pixels
            """
            # If the viewer camera hasn't been created yet, utilize the default width
            return gm.DEFAULT_VIEWER_HEIGHT if self._viewer_camera is None else self._viewer_camera.image_height

        @viewer_height.setter
        def viewer_height(self, height):
            """
            Sets the viewer height @height for this sensor

            Args:
                height (int): viewer height, in pixels
            """
            self._viewer_camera.image_height = height

        @property
        def viewer_width(self):
            """
            Returns:
                int: viewer width of this sensor, in pixels
            """
            # If the viewer camera hasn't been created yet, utilize the default height
            return gm.DEFAULT_VIEWER_WIDTH if self._viewer_camera is None else self._viewer_camera.image_width

        @viewer_width.setter
        def viewer_width(self, width):
            """
            Sets the viewer width @width for this sensor

            Args:
                width (int): viewer width, in pixels
            """
            self._viewer_camera.image_width = width

        def add_ground_plane(self, floor_plane_visible=True, floor_plane_color=None):
            """
            Generate a ground plane into the simulator.
            """
            if self._floor_plane is not None:
                return
            ground_plane_relative_path = "/ground_plane"
            plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
                prim_path="/World" + ground_plane_relative_path,
                name="ground_plane",
                z_position=0,
                size=None,
                color=None if floor_plane_color is None else th.tensor(floor_plane_color),
                visible=floor_plane_visible,
                # TODO: update with new PhysicsMaterial API
                # static_friction=static_friction,
                # dynamic_friction=dynamic_friction,
                # restitution=restitution,
            )

            self._floor_plane = XFormPrim(
                relative_prim_path=ground_plane_relative_path,
                name=plane.name,
            )
            self._floor_plane.load(None)

            # Assign floors category to the floor plane
            lazy.omni.isaac.core.utils.semantics.add_update_semantics(
                prim=self._floor_plane.prim,
                semantic_label="floors",
                type_label="class",
            )

        def add_skybox(self):
            """
            Generate a skybox into the simulator.
            """
            if self._skybox is not None:
                return
            self._skybox = LightObject(
                relative_prim_path="/skybox",
                name="skybox",
                category="background",
                light_type="Dome",
                intensity=1500,
                fixed_base=True,
            )
            self._skybox.load(None)
            self._skybox.color = (1.07, 0.85, 0.61)
            self._skybox.texture_file_path = f"{gm.ASSET_PATH}/models/background/sky.jpg"

        def get_sim_step_dt(self):
            """
            Gets the internal simulation step timestep size

            Returns:
                float: Simulation timestep size
            """
            return self._sim_step_dt

        def set_sim_step_dt(self, sim_step_dt):
            """
            Sets the internal simulation step timestep size

            Args:
                sim_step_dt (float): Timestep size to apply
            """
            # Sanity check this new dt with respect to the rendering dt
            rendering_dt = self.get_rendering_dt()
            self._validate_dts(self.get_physics_dt(), rendering_dt, sim_step_dt)

            self._sim_step_dt = sim_step_dt
            self._n_steps_per_loop = int(sim_step_dt // rendering_dt)

        def set_simulation_dt(self, physics_dt=None, rendering_dt=None):
            # Call super first
            super().set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)

            # Sanity check all new dts
            rendering_dt = self.get_rendering_dt()
            physics_dt = self.get_physics_dt()
            self._validate_dts(physics_dt, rendering_dt, self._sim_step_dt)

        def set_lighting_mode(self, mode):
            """
            Sets the active lighting mode in the current simulator. Valid options are one of LightingMode

            Args:
                mode (LightingMode): Lighting mode to set
            """
            lazy.omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode=mode)

        def enable_viewer_camera_teleoperation(self):
            """
            Enables keyboard control of the active viewer camera for this simulation
            """
            assert gm.RENDER_VIEWER_CAMERA, "Viewer camera must be enabled to enable teleoperation!"
            self._camera_mover = CameraMover(cam=self._viewer_camera)
            self._camera_mover.print_info()
            return self._camera_mover

        def import_scene(self, scene):
            """
            Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

            Args:
                scene (Scene): a scene object to load
            """
            assert self.is_stopped(), "Simulator must be stopped while importing a scene!"
            assert isinstance(scene, Scene), "import_scene can only be called with Scene"

            # Check that the scene is not already imported
            if scene.loaded:
                raise ValueError("Scene is already loaded!")

            self._last_scene_edge = scene.load(
                idx=len(self.scenes),
                last_scene_edge=self._last_scene_edge,
                initial_scene_prim_z_offset=m.INITIAL_SCENE_PRIM_Z_OFFSET,
                scene_margin=m.SCENE_MARGIN,
            )

            # Load the scene.
            self._scenes.append(scene)

            # Make sure simulator is not running, then start it so that we can initialize the scene
            assert self.is_stopped(), "Simulator must be stopped after importing a scene!"
            self.play()

            # Initialize the scene
            scene.initialize()

            # Need to one more step for particle systems to work
            self.step()
            self.stop()
            log.info(f"Imported scene {scene.idx}.")

        @contextlib.contextmanager
        def adding_objects(self, objs):
            """
            Adds a set of objects from the simulator. This is a context manager that handles low-level simulator state
            and should be called externally. Note that this method does not explicitly add the object from
            the simulator; it is assumed that this is handled externally

            Args:
                objs (Iterable[BaseObject]): list of objects to add
            """
            # Immediately yield
            yield

            # Run all post-processing on all newly added objects
            for obj in objs:
                self._post_import_object(obj=obj)

        def _post_import_object(self, obj):
            """
            Post import an object into the simulator, handling any additional setup that needs to be done.

            Args:
                obj (BaseObject): an object to load
            """
            assert isinstance(obj, BaseObject), "_post_import_object can only be called with BaseObject"

            # Run any callbacks
            for callback in self._callbacks_on_add_obj.values():
                callback(obj)

            # Cache the mapping from link IDs to object
            for link in obj.links.values():
                self._link_id_to_objects[lazy.pxr.PhysicsSchemaTools.sdfPathToInt(link.prim_path)] = obj

            # Lastly, additionally add this object automatically to be initialized as soon as another simulator step occurs
            self._objects_to_initialize.append(obj)

        def batch_add_objects(self, objs, scenes):
            """
            Add a set of objects from the simulator.

            Args:
                objs (Iterable[BaseObject]): list of objects to add
                scenes (Iterable[BaseScene]): list of scenes corresponding to each object to load
            """
            with self.adding_objects(objs=objs):
                for obj, scene in zip(objs, scenes):
                    scene.add_object(obj, _batched_call=True)

        @contextlib.contextmanager
        def removing_objects(self, objs):
            """
            Remove a set of objects from the simulator. This is a context manager that handles low-level simulator state
            and should be called externally. Note that this method does not explicitly remove the object from
            the simulator; it is assumed that this is handled externally

            Args:
                objs (Iterable[BaseObject]): list of objects to remove
            """
            playing = self.is_playing()
            if playing:
                state = self.dump_state()

                # Omniverse has a strange bug where if GPU dynamics is on and the object to remove is in contact with
                # with another object (in some specific configuration only, not always), the simulator crashes. Therefore,
                # we first move the object to a safe location, then remove it.
                pos = list(m.OBJECT_GRAVEYARD_POS)
                for ob in objs:
                    ob.set_position_orientation(pos, th.tensor([0, 0, 0, 1], dtype=th.float32))
                    pos[0] += max(ob.aabb_extent)

                # One physics timestep will elapse
                self.step_physics()

            # Run all pre-processing for all objects and record which scenes have been modified
            scenes_modified = set()
            for obj in objs:
                scenes_modified.add(obj.scene)
                self._pre_remove_object(obj)
                # Prune from the state if recorded
                if playing:
                    obj_registry = state[obj.scene.idx]["object_registry"]
                    if (
                        obj.name in obj_registry
                    ):  # a particle system template object might not exist in the registry when it's empty
                        obj_registry.pop(obj.name)

            # Run the main method
            yield

            # Run post-processing required if we were playing
            if playing:
                # Update all handles that are now broken because objects have changed
                self.update_handles()

                if gm.ENABLE_TRANSITION_RULES:
                    # Prune the transition rules that are currently active
                    for scene in scenes_modified:
                        scene.transition_rule_api.prune_active_rules()

                # Load the state back
                self.load_state(state)

        def _pre_remove_object(self, obj):
            """
            Remove a non-robot object from the simulator. Should not be called directly by the user.

            Args:
                obj (BaseObject): a non-robot object to remove
            """
            # Run any callbacks
            for callback in self._callbacks_on_remove_obj.values():
                callback(obj)

            # pop all link ids
            for link in obj.links.values():
                self._link_id_to_objects.pop(lazy.pxr.PhysicsSchemaTools.sdfPathToInt(link.prim_path))

            # If it was queued up to be initialized, remove it from the queue as well
            for i, initialize_obj in enumerate(self._objects_to_initialize):
                if obj.name == initialize_obj.name:
                    self._objects_to_initialize.pop(i)
                    break

        def batch_remove_objects(self, objs):
            """
            Remove a set of objects from the simulator.

            Args:
                objs (Iterable[BaseObject]): list of objects to remove
            """
            with self.removing_objects(objs=objs):
                for obj in objs:
                    obj.scene.remove_object(obj, _batched_call=True)

        def remove_prim(self, prim):
            """
            Remove a prim from the simulator.

            Args:
                prim (BasePrim): a prim to remove
            """
            # [omni.physx.tensors.plugin] prim '[prim_path]' was deleted while being used by a shape in a tensor view
            # class. The physics.tensors simulationView was invalidated.
            with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
                # Remove prim
                prim.remove()

            # Update all handles that are now broken because prims have changed
            self.update_handles()

        def _reset_variables(self):
            """
            Reset internal variables when a new stage is loaded
            """

        def render(self):
            super().render()
            # During rendering, the Fabric API is updated, so we can mark it as clean
            PoseAPI.mark_valid()

        def update_handles(self):
            # Handles are only relevant when physx is running
            if not self.is_playing():
                return

            # First, refresh the physics sim view
            self._physics_sim_view = lazy.omni.physics.tensors.create_simulation_view(self.backend)
            self._physics_sim_view.set_subspace_roots("/")

            # Then update the handles for all objects
            for scene in self.scenes:
                if scene is not None and scene.initialized:
                    for obj in scene.objects:
                        # Only need to update if object is already initialized as well
                        if obj.initialized:
                            obj.update_handles()
                    for system in scene.active_systems.values():
                        if isinstance(system, MacroPhysicalParticleSystem):
                            system.refresh_particles_view()

            # Finally update any unified views
            RigidContactAPI.initialize_view()
            GripperRigidContactAPI.initialize_view()
            ControllableObjectViewAPI.initialize_view()

        def _non_physics_step(self):
            """
            Complete any non-physics steps such as state updates.
            """
            # If we don't have a valid scene, immediately return
            if len(self.scenes) == 0:
                return

            # Update omni
            self._omni_update_step()

            # If we're playing we, also run additional logic
            if self.is_playing():
                # Check to see if any objects should be initialized (only done IF we're playing)
                n_objects_to_initialize = len(self._objects_to_initialize)
                if n_objects_to_initialize > 0 and self.is_playing():
                    # We iterate through the objects to initialize
                    # Note that we don't explicitly do for obj in self._objects_to_initialize because additional objects
                    # may be added mid-iteration!!
                    # For this same reason, after we finish the loop, we keep any objects that are yet to be initialized
                    # First call zero-physics step update, so that handles are properly propagated
                    og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
                    scenes_modified = set()
                    for i in range(n_objects_to_initialize):
                        obj = self._objects_to_initialize[i]
                        obj.initialize()
                        scenes_modified.add(obj.scene)
                        if len(obj.states.keys() & self.object_state_types_on_contact) > 0:
                            self._objects_require_contact_callback = True
                        if len(obj.states.keys() & self.object_state_types_on_joint_break) > 0:
                            self._objects_require_joint_break_callback = True

                    self._objects_to_initialize = self._objects_to_initialize[n_objects_to_initialize:]

                    # Re-initialize the physics view because the number of objects has changed
                    self.update_handles()

                    if gm.ENABLE_TRANSITION_RULES:
                        # Refresh the transition rules
                        for scene in scenes_modified:
                            scene.transition_rule_api.refresh_all_rules()

                # Update any system-related state
                for scene in self.scenes:
                    for system in scene.active_systems.values():
                        system.update()

                # Propagate states if the feature is enabled
                if gm.ENABLE_OBJECT_STATES:
                    # Step the object states in global topological order (if the scene exists)
                    for state_type in self.object_state_types_requiring_update:
                        if issubclass(state_type, GlobalUpdateStateMixin):
                            state_type.global_update()
                        if issubclass(state_type, UpdateStateMixin):
                            for scene in self.scenes:
                                for obj in scene.get_objects_with_state(state_type):
                                    # Update the state (object should already be initialized since
                                    # this step will only occur after objects are initialized and sim
                                    # is playing
                                    obj.states[state_type].update()

                    for scene in self.scenes:
                        for obj in scene.objects:
                            # Only update visuals for objects that have been initialized so far
                            if isinstance(obj, StatefulObject) and obj.initialized:
                                obj.update_visuals()

                # Possibly run transition rule step
                if gm.ENABLE_TRANSITION_RULES:
                    for scene in self.scenes:
                        scene.transition_rule_api.step()

        def _omni_update_step(self):
            """
            Step any omni-related things
            """
            # Clear the bounding box and contact caches so that they get updated during the next time they're called
            RigidContactAPI.clear()
            GripperRigidContactAPI.clear()
            ControllableObjectViewAPI.clear()

        def play(self):
            if not self.is_playing():
                # Track whether we're starting the simulator fresh -- i.e.: whether we were stopped previously
                was_stopped = self.is_stopped()

                # Run super first
                # We suppress warnings from omni.usd because it complains about values set in the native USD
                # These warnings occur because the native USD file has some type mismatch in the `scale` property,
                # where the property expects a double but for whatever reason the USD interprets its values as floats
                # We suppress omni.physicsschema.plugin when kinematic_only objects are placed with scale ~1.0, to suppress
                # the following error:
                # [omni.physicsschema.plugin] ScaleOrientation is not supported for rigid bodies, prim path: [...] You may
                #   ignore this if the scale is close to uniform.
                # We also need to suppress the following error when flat cache is used:
                # [omni.physx.plugin] Transformation change on non-root links is not supported.
                channels = ["omni.usd", "omni.physicsschema.plugin"]
                if gm.ENABLE_FLATCACHE:
                    channels.append("omni.physx.plugin")
                with suppress_omni_log(channels=channels):
                    super().play()

                # Take a render step -- this is needed so that certain (unknown, maybe omni internal state?) is populated
                # correctly.
                self.render()

                # Update all object handles, unless this is a play during initialization
                if og.sim is not None:
                    self.update_handles()

                if was_stopped:
                    # We need to update controller mode because kp and kd were set to the original (incorrect) values when
                    # sim was stopped. We need to reset them to default_kp and default_kd defined in ControllableObject.
                    # We also need to take an additional sim step to make sure simulator is functioning properly.
                    # We need to do this because for some reason omniverse exhibits strange behavior if we do certain
                    # operations immediately after playing; e.g.: syncing USD poses when flatcache is enabled
                    if len(self.scenes) > 0 and all([scene.initialized for scene in self.scenes]):
                        for scene in self.scenes:
                            for robot in scene.robots:
                                if robot.initialized:
                                    robot.update_controller_mode()

                            # Also refresh any transition rules that became stale while sim was stopped
                            if gm.ENABLE_TRANSITION_RULES:
                                scene.transition_rule_api.refresh_all_rules()

                # Additionally run non physics things
                self._non_physics_step()

            # Run all callbacks
            for callback in self._callbacks_on_play.values():
                callback()

        def pause(self):
            if not self.is_paused():
                super().pause()

        def stop(self):
            if not self.is_stopped():
                super().stop()

            # If we're using flatcache, we also need to reset its API
            if gm.ENABLE_FLATCACHE:
                FlatcacheAPI.reset()

            # Run all callbacks
            for callback in self._callbacks_on_stop.values():
                callback()

        @property
        def n_physics_timesteps_per_render(self):
            """
            Number of physics timesteps per rendering timestep. rendering_dt has to be a multiple of physics_dt.

            Returns:
                int: Discrete number of physics timesteps to take per step
            """
            n_physics_timesteps_per_render = self.get_rendering_dt() / self.get_physics_dt()
            assert n_physics_timesteps_per_render.is_integer(), "render_timestep must be a multiple of physics_timestep"
            return int(n_physics_timesteps_per_render)

        def step(self):
            """
            Step the simulation at self.get_sim_step_dt() rate
            """
            render = self._render_on_step
            if self.stage is None:
                raise Exception("There is no stage currently opened, init_stage needed before calling this func")

            # If we have imported any objects within the last timestep, we render the app once, since otherwise calling
            # step() may not step physics
            if len(self._objects_to_initialize) > 0:
                self.render()

            # Clear all scenes' updated objects
            for scene in self.scenes:
                scene.clear_updated_objects()

            for _ in range(self._n_steps_per_loop):
                if render:
                    super().step(render=True)
                else:
                    for i in range(self.n_physics_timesteps_per_render):
                        super().step(render=False)

            # Additionally run non physics things
            self._non_physics_step()

            # TODO (eric): After stage changes (e.g. pose, texture change), it will take two super().step(render=True) for
            #  the result to propagate to the rendering. We could have called super().render() here but it will introduce
            #  a big performance regression.

        def step_physics(self):
            """
            Step the physics a single step.
            """
            self._physics_context._step(current_time=self.current_time)

            # Update all APIs
            self._omni_update_step()
            PoseAPI.invalidate()

        def _on_physics_step(self):
            # Make the controllable object view API refresh
            ControllableObjectViewAPI.clear()

            # Run the controller step on every controllable object
            for scene in self.scenes:
                for obj in scene.objects:
                    if isinstance(obj, ControllableObject):
                        obj.step()

            # Flush the controls from the ControllableObjectViewAPI
            ControllableObjectViewAPI.flush_control()

        def _on_contact(self, contact_headers, contact_data):
            """
            This callback will be invoked after every PHYSICS step if there is any contact.
            For each of the pair of objects in each contact, we invoke the on_contact function for each of its states
            that subclass ContactSubscribedStateMixin. These states update based on contact events.
            """
            if gm.ENABLE_OBJECT_STATES and self._objects_require_contact_callback:
                headers = defaultdict(list)
                for contact_header in contact_headers:
                    actor0_obj = self._link_id_to_objects.get(contact_header.actor0, None)
                    actor1_obj = self._link_id_to_objects.get(contact_header.actor1, None)
                    # If any of the objects cannot be found, skip
                    if actor0_obj is None or actor1_obj is None:
                        continue
                    # If any of the objects is not initialized, skip
                    if not actor0_obj.initialized or not actor1_obj.initialized:
                        continue
                    # If any of the objects is not stateful, skip
                    if not isinstance(actor0_obj, StatefulObject) or not isinstance(actor1_obj, StatefulObject):
                        continue
                    # If any of the objects doesn't have states that require on_contact callbacks, skip
                    if (
                        len(actor0_obj.states.keys() & self.object_state_types_on_contact) == 0
                        or len(actor1_obj.states.keys() & self.object_state_types_on_contact) == 0
                    ):
                        continue
                    headers[tuple(sorted((actor0_obj, actor1_obj), key=lambda x: x.uuid))].append(contact_header)

                for actor0_obj, actor1_obj in headers:
                    for obj0, obj1 in [(actor0_obj, actor1_obj), (actor1_obj, actor0_obj)]:
                        for state_type in self.object_state_types_on_contact:
                            if state_type in obj0.states:
                                obj0.states[state_type].on_contact(
                                    obj1, headers[(actor0_obj, actor1_obj)], contact_data
                                )

        def get_obj_at_prim_path(self, prim_path):
            for scene in self.scenes:
                obj = scene.object_registry("prim_path", prim_path)
                if obj is not None:
                    return obj
            return None

        def _on_simulation_event(self, event):
            """
            This callback will be invoked if there is any simulation event. Currently it only processes JOINT_BREAK event.
            """
            if gm.ENABLE_OBJECT_STATES:
                if (
                    event.type == int(lazy.omni.physx.bindings._physx.SimulationEvent.JOINT_BREAK)
                    and self._objects_require_joint_break_callback
                ):
                    joint_path = str(
                        lazy.pxr.PhysicsSchemaTools.decodeSdfPath(
                            event.payload["jointPath"][0], event.payload["jointPath"][1]
                        )
                    )
                    obj = None

                    tokens = joint_path.split("/")
                    for i in range(2, len(tokens) + 1):
                        obj = self.get_obj_at_prim_path("/".join(tokens[:i]))
                        if obj is not None:
                            break

                    if obj is None or not obj.initialized or not isinstance(obj, StatefulObject):
                        return
                    if len(obj.states.keys() & self.object_state_types_on_joint_break) == 0:
                        return
                    for state_type in self.object_state_types_on_joint_break:
                        if state_type in obj.states:
                            obj.states[state_type].on_joint_break(joint_path)

        def is_paused(self):
            """
            Returns:
                bool: True if the simulator is paused, otherwise False
            """
            return not (self.is_stopped() or self.is_playing())

        @contextlib.contextmanager
        def render_on_step(self, value):
            """
            A context scope for setting whether rendering should occur on each simulator step.
            """
            # Store the original value, set the new value, yield, and then reset the original value
            original_value = self._render_on_step
            self._render_on_step = value
            yield
            self._render_on_step = original_value

        @contextlib.contextmanager
        def stopped(self):
            """
            A context scope for making sure the simulator is stopped during execution within this scope.
            Upon leaving the scope, the prior simulator state is restored.
            """
            # Infer what state we're currently in, then stop, yield, and then restore the original state
            sim_is_playing, sim_is_paused = self.is_playing(), self.is_paused()
            if sim_is_playing or sim_is_paused:
                self.stop()
            yield
            if sim_is_playing:
                self.play()
            elif sim_is_paused:
                self.pause()

        @contextlib.contextmanager
        def playing(self):
            """
            A context scope for making sure the simulator is playing during execution within this scope.
            Upon leaving the scope, the prior simulator state is restored.
            """
            # Infer what state we're currently in, then stop, yield, and then restore the original state
            sim_is_stopped, sim_is_paused = self.is_stopped(), self.is_paused()
            if sim_is_stopped or sim_is_paused:
                self.play()
            yield
            if sim_is_stopped:
                self.stop()
            elif sim_is_paused:
                self.pause()

        @contextlib.contextmanager
        def paused(self):
            """
            A context scope for making sure the simulator is paused during execution within this scope.
            Upon leaving the scope, the prior simulator state is restored.
            """
            # Infer what state we're currently in, then stop, yield, and then restore the original state
            sim_is_stopped, sim_is_playing = self.is_stopped(), self.is_playing()
            if sim_is_stopped or sim_is_playing:
                self.pause()
            yield
            if sim_is_stopped:
                self.stop()
            elif sim_is_playing:
                self.play()

        @contextlib.contextmanager
        def slowed(self, dt):
            """
            A context scope for making the simulator simulation dt slowed, e.g.: for taking micro-steps for propagating
            instantaneous kinematics with minimal impact on physics propagation.

            NOTE: This will set both the physics dt and rendering dt to the same value during this scope.

            Upon leaving the scope, the prior simulator state is restored.
            """
            # Set dt, yield, then restore the original dt
            physics_dt, rendering_dt = self.get_physics_dt(), self.get_rendering_dt()
            self.set_simulation_dt(physics_dt=dt, rendering_dt=dt)
            yield
            self.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)

        def add_callback_on_play(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time sim.play() is called

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback() --> None
            """
            self._callbacks_on_play[name] = callback

        def add_callback_on_stop(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time sim.stop() is called

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback() --> None
            """
            self._callbacks_on_stop[name] = callback

        def add_callback_on_add_obj(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time
            sim._post_import_object() is called

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback(obj: BaseObject) --> None
            """
            self._callbacks_on_add_obj[name] = callback

        def add_callback_on_remove_obj(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time
            sim._pre_remove_object() is called

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback(obj: BaseObject) --> None
            """
            self._callbacks_on_remove_obj[name] = callback

        def add_callback_on_system_init(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time a system is initialized

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback(system: System) --> None
            """
            self._callbacks_on_system_init[name] = callback

        def add_callback_on_system_clear(self, name, callback):
            """
            Adds a function @callback, referenced by @name, to be executed every time a system is cleared

            Args:
                name (str): Name of the callback
                callback (function): Callback function. Function signature is expected to be:

                    def callback(system: System) --> None
            """
            self._callbacks_on_system_clear[name] = callback

        def remove_callback_on_play(self, name):
            """
            Remove play callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_play.pop(name, None)

        def remove_callback_on_stop(self, name):
            """
            Remove stop callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_stop.pop(name, None)

        def remove_callback_on_add_obj(self, name):
            """
            Remove add obj callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_add_obj.pop(name, None)

        def remove_callback_on_remove_obj(self, name):
            """
            Remove remove obj callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_remove_obj.pop(name, None)

        def remove_callback_on_system_init(self, name):
            """
            Remove system init callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_system_init.pop(name, None)

        def remove_callback_on_system_clear(self, name):
            """
            Remove system clear callback whose reference is @name

            Args:
                name (str): Name of the callback
            """
            self._callbacks_on_system_clear.pop(name, None)

        @property
        def pi(self):
            """
            Returns:
                PhysX: Physx Interface (pi) for controlling low-level physx engine
            """
            return self._physx_interface

        @property
        def psi(self):
            """
            Returns:
                IPhysxSimulation: Physx Simulation Interface (psi) for controlling low-level physx simulation
            """
            return self._physx_simulation_interface

        @property
        def psqi(self):
            """
            Returns:
                PhysXSceneQuery: Physx Scene Query Interface (psqi) for running low-level scene queries
            """
            return self._physx_scene_query_interface

        @property
        def scenes(self):
            """
            Returns:
                Empty list or [Scene]: Scenes currently loaded in this simulator. If no scenes are loaded, returns empty list
            """
            return self._scenes

        @property
        def viewer_camera(self):
            """
            Returns:
                VisionSensor: Active camera sensor corresponding to the active viewport window instance shown in the omni UI
            """
            return self._viewer_camera

        @property
        def camera_mover(self):
            """
            Returns:
                None or CameraMover: If enabled, the teleoperation interface for controlling the active viewer camera
            """
            return self._camera_mover

        @property
        def world_prim(self):
            """
            Returns:
                Usd.Prim: Prim at /World
            """
            return lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path="/World")

        @property
        def floor_plane(self):
            return self._floor_plane

        @property
        def skybox(self):
            return self._skybox

        def get_callbacks_on_system_init(self):
            return self._callbacks_on_system_init

        def get_callbacks_on_system_clear(self):
            return self._callbacks_on_system_clear

        def write_metadata(self, key, data):
            """
            Writes metadata @data to the current global metadata dict using key @key

            Args:
                key (str): Keyword entry in the global metadata dictionary to use
                data (dict): Data to write to @key in the global metadata dictionary
            """
            self.world_prim.SetCustomDataByKey(key, data)

        def get_metadata(self, key):
            """
            Grabs metadata from the current global metadata dict using key @key

            Args:
                key (str): Keyword entry in the global metadata dictionary to use
            """
            return self.world_prim.GetCustomDataByKey(key)

        def restore(self, scene_files):
            """
            Restore simulation environments from @json_paths.

            Args:
                scene_files (List[str] or List[dict]): Full paths of either JSON files or loaded scene files to load,
                    which contains information to recreate a scene.
            """
            # Note whether we're loading from scratch or not
            load_from_scratch = len(self.scenes) == 0

            # We don't support smart diff'ing if there's a mismatch in number of scenes
            if not load_from_scratch and len(self.scenes) != len(scene_files):
                log.error(
                    "There is a mismatch between the number of active scenes and number of json_paths to be "
                    "loaded. Please call og.clear() to relaunch the simulator first."
                )
                return

            # Parse each json path individually
            states = []
            og.sim.stop()
            for i, scene_file in enumerate(scene_files):
                if isinstance(scene_file, str):
                    if not scene_file.endswith(".json"):
                        log.error(f"You have to define the full json_path to load from. Got: {scene_file}")
                        return

                    # Load the info from the json
                    with open(scene_file, "r") as f:
                        scene_info = json.load(f)
                else:
                    scene_info = scene_file
                init_info = scene_info["init_info"]
                # The saved state are lists, convert them to torch tensors
                state = recursively_convert_to_torch(scene_info["state"])
                states.append(state)

                if load_from_scratch:
                    # Override the init info with our json path
                    init_info["args"]["scene_file"] = scene_file

                    # Also make sure we have any additional modifications necessary from the specific scene
                    og.REGISTERED_SCENES[init_info["class_name"]].modify_init_info_for_restoring(init_info=init_info)

                    # Recreate and import the saved scene
                    recreated_scene = create_object_from_init_info(init_info)
                    self.import_scene(scene=recreated_scene)

                else:
                    scene = self.scenes[i]
                    # Make sure the class type is the same
                    if scene.__class__.__name__ != init_info["class_name"]:
                        log.error(
                            f"Got mismatch in scene type: current is type {scene.__class__.__name__}, trying to load type {init_info['class_name']}"
                        )

                    current_obj_names = set(scene.object_registry.get_dict("name").keys())
                    load_obj_names = set(scene_info["objects_info"]["init_info"].keys())

                    objs_to_remove = current_obj_names - load_obj_names
                    objs_to_add = load_obj_names - current_obj_names

                    # Delete any extra objects that currently exist in the scene stage
                    objects_to_remove = [
                        scene.object_registry("name", obj_to_remove) for obj_to_remove in objs_to_remove
                    ]
                    og.sim.batch_remove_objects(objects_to_remove)

                    # Add any extra objects that do not currently exist in the scene stage
                    objects_to_add = [
                        create_object_from_init_info(scene_info["objects_info"]["init_info"][obj_to_add])
                        for obj_to_add in objs_to_add
                    ]
                    og.sim.batch_add_objects(objects_to_add, scenes=[scene] * len(objects_to_add))

            # Start the simulation and restore the dynamic state of the scene
            self.play()
            for i, state in enumerate(states):
                self.scenes[i].load_state(state, serialized=False)

            log.info("The saved simulation environment loaded.")

        def save(self, json_paths=None):
            """
            Saves the current simulation environment to @json_path.

            Args:
                json_paths (None or List[str]): Full path of JSON files to save (should end with .json), each of which
                    contain information to recreate the current scenes, if specified. List should have one element per
                    currently loaded scene. If None, will return a list of JSON strings instead.

            Returns:
                None or list of str: If @json_paths is None, returns list of dumped json strings. Else, None
            """
            # Make sure the sim is not stopped, since we need to grab joint states
            assert not self.is_stopped(), "Simulator cannot be stopped when saving to USD!"

            # Make sure there are no objects in the initialization queue, if not, terminate early and notify user
            # Also run other sanity checks before saving
            if len(self._objects_to_initialize) > 0:
                log.error("There are still objects to initialize! Please take one additional sim step and then save.")
                return
            if not self.scenes:
                log.warning("Scene has not been loaded. Nothing to save.")
                return
            if json_paths is not None:
                if isinstance(json_paths, str):
                    log.error(
                        f"You must define a list of .json paths, one for each scene. Number of scenes: {len(self.scenes)}"
                    )
                    return
                if not all([json_path.endswith(".json") for json_path in json_paths]):
                    log.error(f"You have to define the full json_path to save the scene to. Got: {json_paths}")
                    return

            if not json_paths:
                json_paths = [None] * len(self.scenes)

            assert len(json_paths) == len(self.scenes), "Number of json paths should match the number of scenes"

            # Update scene info
            jsons = []
            for scene, json_path in zip(self.scenes, json_paths):
                scene.update_objects_info()

                # Dump saved current state and also scene init info
                scene_info = {
                    "metadata": self.world_prim.GetCustomData(),
                    "state": scene.dump_state(serialized=False),
                    "init_info": scene.get_init_info(),
                    "objects_info": scene.get_objects_info(),
                }

                # Write this to the json file
                if json_path is None:
                    jsons.append(json.dumps(scene_info, cls=TorchEncoder, indent=4))

                else:
                    Path(os.path.dirname(json_path)).mkdir(parents=True, exist_ok=True)
                    with open(json_path, "w+") as f:
                        json.dump(scene_info, f, cls=TorchEncoder, indent=4)

                    log.info(f"Scene {scene.idx} saved.")

            if jsons:
                return jsons

            return None

        def _partial_clear(self):
            """Partial clear clearing all components owned by the Simulator. Rest is completed in og.clear."""
            # Stop the physics
            self.stop()

            # Clear all scenes
            for scene in self.scenes:
                scene.clear()

            # Remove the skybox, floor plane and viewer camera
            if self._skybox is not None:
                self._skybox.remove()

            if self._floor_plane is not None:
                self._floor_plane.remove()

            if self._viewer_camera is not None:
                self._viewer_camera.remove()

            if self._camera_mover is not None:
                self._camera_mover.clear()

            # Clear the vision sensor cache
            VisionSensor.clear()

            # Clear all global update states
            for state in self.object_state_types_requiring_update:
                if issubclass(state, GlobalUpdateStateMixin):
                    state.global_initialize()

            # Clear all materials
            MaterialPrim.clear()

            # Clear pose API cache
            PoseAPI.clear()

            # Clear uniquely named items and other internal states
            clear_python_utils()
            clear_usd_utils()

        def close(self):
            """
            Shuts down the OmniGibson application
            """
            self._app.shutdown()

        @property
        def stage_id(self):
            """
            Returns:
                int: ID of the current active stage
            """
            return self._stage_id

        @property
        def device(self):
            """
            Returns:
                device (None or str): Device used in simulation backend
            """
            return self._device

        @device.setter
        def device(self, device):
            """
            Sets the device used for sim backend

            Args:
                device (None or str): Device to set for the simulation backend
            """
            self._device = device
            if self._device is not None and "cuda" in self._device:
                device_id = self._settings.get_as_int("/physics/cudaDevice")
                self._device = f"cuda:{device_id}"

        @property
        def initial_physics_dt(self):
            """
            Returns:
                float: Physics timestep
            """
            return self._initial_physics_dt

        @property
        def initial_rendering_dt(self):
            """
            Returns:
                float: Rendering timestep
            """
            return self._initial_rendering_dt

        def _dump_state(self):
            # Default state is from the scene
            return {i: scene.dump_state(serialized=False) for i, scene in enumerate(self.scenes)}

        def _load_state(self, state):
            # Default state is from the scene
            for i, scene in enumerate(self.scenes):
                scene.load_state(state=state[i], serialized=False)

        def load_state(self, state, serialized=False):
            # We need to make sure the simulator is playing since joint states only get updated when playing
            assert self.is_playing()

            # Run super
            super().load_state(state=state, serialized=serialized)

            # Highlight that at the current step, the non-kinematic states are potentially inaccurate because a sim
            # step is needed to propagate specific states in physics backend
            # TODO: This should be resolved in a future omniverse release!
            disclaimer(
                "Attempting to load simulator state.\n"
                "Currently, omniverse does not support exclusively stepping kinematics, so we cannot update some "
                "of our object states relying on updated kinematics until a simulator step is taken!\n"
                "Object states such as OnTop, Inside, etc. relying on relative spatial information will inaccurate"
                "until a single sim step is taken.\n"
                "This should be resolved by the next NVIDIA Isaac Sim release."
            )

        def serialize(self, state):
            # Default state is from the scene
            return th.cat([scene.serialize(state=state[i]) for i, scene in enumerate(self.scenes)], dim=0)

        def deserialize(self, state):
            # Default state is from the scene
            dicts = {}
            total_state_size = 0
            for i, scene in enumerate(self.scenes):
                scene_dict, deserialized_items = scene.deserialize(state=state[total_state_size:])
                dicts[i] = scene_dict
                total_state_size += deserialized_items
            return dicts, total_state_size

    if not og.sim:
        # The simulator init function saves itself as og.sim.
        Simulator(*args, **kwargs)

        print()
        print_icon()
        print_logo()
        print()
        log.info(f"{'-' * 10} Welcome to {logo_small()}! {'-' * 10}")

    return og.sim
