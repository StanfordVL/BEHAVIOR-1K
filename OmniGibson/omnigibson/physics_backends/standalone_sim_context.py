"""
Drop-in replacement for Isaac Sim's `isaacsim.core.api.SimulationContext`, used by non-PhysX physics
backends. Provides only the subset of that class's interface a
`PhysicsBackend` actually needs (dt/time bookkeeping, a USD stage, and play/pause/stop state) without
launching Isaac Sim's Kit application itself (that decision belongs to whichever backend -- physics or
render -- actually needs Kit; see `simulator.py::_launch_app()`).

There is no Kit event system to drive physics stepping standalone: `step()` calls directly into the
owning backend's `step_physics_once()`.
"""

import omnigibson.lazy as lazy


class StandaloneSimulationContext:
    def __init__(self, backend, physics_dt, rendering_dt, device=None):
        self._backend = backend
        self._stage = lazy.pxr.Usd.Stage.CreateInMemory()
        # Kit's create_new_stage() authors these. A bare stage falls back to Y-up, and Newton's
        # add_usd() then turns every frame it imports 90 degrees about x to convert to Z-up.
        lazy.pxr.UsdGeom.SetStageUpAxis(self._stage, lazy.pxr.UsdGeom.Tokens.z)
        lazy.pxr.UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        self._initial_physics_dt = physics_dt
        self._initial_rendering_dt = rendering_dt
        self._physics_dt = physics_dt
        self._rendering_dt = rendering_dt
        self.device = device

        # Explicit tri-state -- "playing" / "paused" / "stopped" -- rather than a single bool, so
        # pause() and stop() are actually distinguishable.
        self._state = "stopped"
        self._current_time = 0.0
        self._current_time_step_index = 0

        # Populated by the owning backend's refresh_physics_sim_view(); no Kit omni.physics.tensors
        # view exists standalone.
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
        # No Kit viewport/renderer exists standalone. Rendering and sensors are out of scope for the
        # standalone physics-backend path (gm.PHYSICS_BACKEND != "physx").
        pass

    def step(self, render=True):
        # Match Isaac's SimulationContext.step() contract, which this class stands in for:
        # step(render=True) advances one *rendering* frame, not one physics step. Isaac implements that
        # by calling self._app.update(), which ticks Kit once and lets the omni.physx extension take
        # however many physics substeps that frame spans (rendering_dt / physics_dt of them); only
        # step(render=False) is a single physics step. Simulator.step() relies on exactly this split:
        # its render branch calls backend.step(render=True) ONCE per loop, while its non-render branch
        # does its own `for _ in range(n_physics_timesteps_per_render)` around step(render=False).
        #
        # Taking a single substep here regardless of `render` therefore dilated simulated time by
        # rendering_dt / physics_dt (4x at OmniGibson's default 120 Hz physics / 30 Hz rendering) for
        # every stepped frame, since _render_on_step is True by default and nothing ever unsets it.
        # Measured on a teleop recording before this fix: every body's reported velocity was exactly
        # 4.0x its own finite-differenced d(pos)/dt, and a free body's vertical velocity decayed by
        # 0.0818 m/s per stepped frame (g / 120) instead of 0.327 (g / 30) -- i.e. the whole scene,
        # robot articulation included, ran in quarter-speed slow motion.
        n_substeps = int(round(self._rendering_dt / self._physics_dt)) if render else 1
        for _ in range(max(n_substeps, 1)):
            self._backend.step_physics_once(current_time=self._current_time)
            self._current_time += self._physics_dt
            self._current_time_step_index += 1
