import omnigibson.lazy as lazy


class StandaloneSimulationContext:
    def __init__(self, backend, physics_dt, rendering_dt, device=None):
        self._backend = backend
        self._stage = lazy.pxr.Usd.Stage.CreateInMemory()
        # A bare stage defaults to Y-up, which makes Newton's add_usd() rotate every imported frame
        lazy.pxr.UsdGeom.SetStageUpAxis(self._stage, lazy.pxr.UsdGeom.Tokens.z)
        lazy.pxr.UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        self._initial_physics_dt = physics_dt
        self._initial_rendering_dt = rendering_dt
        self._physics_dt = physics_dt
        self._rendering_dt = rendering_dt
        self.device = device

        self._state = "stopped"
        self._current_time = 0.0
        self._current_time_step_index = 0

        self.physics_sim_view = None

    @property
    def stage(self):
        return self._stage

    @property
    def current_time(self):
        return self._current_time

    @property
    def current_time_step_index(self):
        return self._current_time_step_index

    def is_playing(self):
        return self._state == "playing"

    def is_stopped(self):
        return self._state == "stopped"

    def is_paused(self):
        return self._state == "paused"

    def get_physics_dt(self):
        return self._physics_dt

    def get_rendering_dt(self):
        return self._rendering_dt

    def set_simulation_dt(self, physics_dt=None, rendering_dt=None):
        if physics_dt is not None:
            self._physics_dt = physics_dt
        if rendering_dt is not None:
            self._rendering_dt = rendering_dt

    def get_physics_context(self):
        raise NotImplementedError(
            "No Kit physics context exists for a standalone backend; physics settings are applied "
            "directly by the backend instead."
        )

    def play(self):
        self._state = "playing"

    def pause(self):
        self._state = "paused"

    def stop(self):
        self._state = "stopped"
        self._current_time = 0.0
        self._current_time_step_index = 0

    def render(self):
        pass

    def step(self, render=True):
        # Like Isaac's SimulationContext.step(), render=True advances a whole rendering frame of substeps
        n_substeps = int(round(self._rendering_dt / self._physics_dt)) if render else 1
        for _ in range(max(n_substeps, 1)):
            self._backend.step_physics_once(current_time=self._current_time)
            self._current_time += self._physics_dt
            self._current_time_step_index += 1
