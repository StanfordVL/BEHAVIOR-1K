"""Common simulator interface for OmniGibson backend implementations."""

import contextlib
from abc import ABC


class UnsupportedSimulatorFeature(NotImplementedError):
    """Raised when a simulator backend cannot implement a legacy simulator API yet."""


class AbstractSimulator(ABC):
    """Compatibility surface shared by PhysX and Newton simulators.

    The methods mirror the legacy ``omnigibson.simulator.Simulator`` API. Backends
    should implement the methods they support and leave the rest as explicit
    ``NotImplementedError`` / no-op placeholders while the Newton migration
    progresses.
    """

    @classmethod
    def from_environment_configs(cls, configs):
        raise UnsupportedSimulatorFeature("from_environment_configs is not implemented for this simulator backend.")

    @staticmethod
    def should_auto_build_environment(configs):
        return True

    def build_environment(self):
        raise UnsupportedSimulatorFeature("build_environment is not implemented for this simulator backend.")

    def apply_environment_action(self, action):
        raise UnsupportedSimulatorFeature("apply_environment_action is not implemented for this simulator backend.")

    def add_ground_plane(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("add_ground_plane is not implemented for this simulator backend.")

    def add_skybox(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("add_skybox is not implemented for this simulator backend.")

    def get_sim_step_dt(self):
        raise UnsupportedSimulatorFeature("get_sim_step_dt is not implemented for this simulator backend.")

    def set_simulation_dt(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("set_simulation_dt is not implemented for this simulator backend.")

    def set_lighting_mode(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("set_lighting_mode is not implemented for this simulator backend.")

    def enable_viewer_camera_teleoperation(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("enable_viewer_camera_teleoperation is not implemented for this backend.")

    def import_scene(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("import_scene is not implemented for this simulator backend.")

    @contextlib.contextmanager
    def adding_objects(self, *args, **kwargs):
        yield

    def batch_add_objects(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("batch_add_objects is not implemented for this simulator backend.")

    @contextlib.contextmanager
    def removing_objects(self, *args, **kwargs):
        yield

    def batch_remove_objects(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("batch_remove_objects is not implemented for this simulator backend.")

    def remove_prim(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("remove_prim is not implemented for this simulator backend.")

    def get_physics_context(self):
        raise UnsupportedSimulatorFeature("get_physics_context is not implemented for this simulator backend.")

    @property
    def stage(self):
        raise UnsupportedSimulatorFeature("stage is not implemented for this simulator backend.")

    @property
    def current_time(self):
        raise UnsupportedSimulatorFeature("current_time is not implemented for this simulator backend.")

    @property
    def current_time_step_index(self):
        raise UnsupportedSimulatorFeature("current_time_step_index is not implemented for this simulator backend.")

    def is_playing(self):
        return True

    def is_stopped(self):
        return False

    def is_paused(self):
        return False

    def get_physics_dt(self):
        raise UnsupportedSimulatorFeature("get_physics_dt is not implemented for this simulator backend.")

    def get_rendering_dt(self):
        raise UnsupportedSimulatorFeature("get_rendering_dt is not implemented for this simulator backend.")

    @property
    def physics_sim_view(self):
        raise UnsupportedSimulatorFeature("physics_sim_view is not implemented for this simulator backend.")

    @property
    def pi(self):
        raise UnsupportedSimulatorFeature("pi is not implemented for this simulator backend.")

    @property
    def psi(self):
        raise UnsupportedSimulatorFeature("psi is not implemented for this simulator backend.")

    @property
    def psqi(self):
        raise UnsupportedSimulatorFeature("psqi is not implemented for this simulator backend.")

    def render(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("render is not implemented for this simulator backend.")

    def sync_physx_to_fabric(self):
        pass

    def update_handles(self):
        pass

    def play(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass

    def step(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("step is not implemented for this simulator backend.")

    def step_physics(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("step_physics is not implemented for this simulator backend.")

    def get_obj_at_prim_path(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("get_obj_at_prim_path is not implemented for this simulator backend.")

    @contextlib.contextmanager
    def render_on_step(self, *args, **kwargs):
        yield

    @contextlib.contextmanager
    def slowed(self, *args, **kwargs):
        yield

    @contextlib.contextmanager
    def editing_usd(self, *args, **kwargs):
        yield

    def add_callback_on_play(self, *args, **kwargs):
        pass

    def add_callback_on_stop(self, *args, **kwargs):
        pass

    def add_callback_on_add_obj(self, *args, **kwargs):
        pass

    def add_callback_on_remove_obj(self, *args, **kwargs):
        pass

    def add_callback_on_system_init(self, *args, **kwargs):
        pass

    def add_callback_on_system_clear(self, *args, **kwargs):
        pass

    def remove_callback_on_play(self, *args, **kwargs):
        pass

    def remove_callback_on_stop(self, *args, **kwargs):
        pass

    def remove_callback_on_add_obj(self, *args, **kwargs):
        pass

    def remove_callback_on_remove_obj(self, *args, **kwargs):
        pass

    def remove_callback_on_system_init(self, *args, **kwargs):
        pass

    def remove_callback_on_system_clear(self, *args, **kwargs):
        pass

    def get_callbacks_on_system_init(self):
        return {}

    def get_callbacks_on_system_clear(self):
        return {}

    @property
    def scenes(self):
        raise UnsupportedSimulatorFeature("scenes is not implemented for this simulator backend.")

    @property
    def viewer_camera(self):
        viewer_camera = getattr(self, "_viewer_camera", None)
        if viewer_camera is None:
            raise UnsupportedSimulatorFeature("viewer_camera is not implemented for this simulator backend.")
        return viewer_camera

    @viewer_camera.setter
    def viewer_camera(self, viewer_camera):
        self._viewer_camera = viewer_camera

    @property
    def camera_mover(self):
        return None

    @property
    def world_prim(self):
        raise UnsupportedSimulatorFeature("world_prim is not implemented for this simulator backend.")

    @property
    def floor_plane(self):
        return None

    @property
    def skybox(self):
        return None

    def restore(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("restore is not implemented for this simulator backend.")

    def save(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("save is not implemented for this simulator backend.")

    def close(self):
        pass

    @property
    def stage_id(self):
        raise UnsupportedSimulatorFeature("stage_id is not implemented for this simulator backend.")

    @property
    def device(self):
        raise UnsupportedSimulatorFeature("device is not implemented for this simulator backend.")

    @property
    def initial_physics_dt(self):
        raise UnsupportedSimulatorFeature("initial_physics_dt is not implemented for this simulator backend.")

    @property
    def initial_rendering_dt(self):
        raise UnsupportedSimulatorFeature("initial_rendering_dt is not implemented for this simulator backend.")

    def dump_state(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("dump_state is not implemented for this simulator backend.")

    def load_state(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("load_state is not implemented for this simulator backend.")

    def serialize(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("serialize is not implemented for this simulator backend.")

    def deserialize(self, *args, **kwargs):
        raise UnsupportedSimulatorFeature("deserialize is not implemented for this simulator backend.")


SimulatorBase = AbstractSimulator


__all__ = ["AbstractSimulator", "SimulatorBase", "UnsupportedSimulatorFeature"]
