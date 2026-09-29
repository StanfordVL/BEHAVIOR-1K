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
    def __init__(self, backend, physics_dt, rendering_dt, device=None, use_kit_stage=False):
        """
        Args:
            use_kit_stage (bool): if True, attach to Kit's own USD context stage (via
                `isaacsim.core.utils.stage.create_new_stage()`, i.e. `omni.usd.get_context().new_stage()`)
                instead of a private in-memory one -- needed whenever a live Kit application is also
                present (e.g. the physics=Newton + render=Kit hybrid) so Kit's own Fabric/renderer has
                a real stage to read from. A Kit-less run (the historical default) keeps using a private
                in-memory stage, invisible to (and independent of) any Kit application.
        """
        self._backend = backend
        if use_kit_stage:
            # create_new_stage() (== omni.usd.get_context().new_stage()) returns a bool success flag,
            # not the Stage itself (confirmed empirically -- its own docstring example is misleading);
            # fetch the actual Usd.Stage object separately afterward.
            #
            # Note: once Kit's timeline plays (needed for KitRenderBackend to render at all), the
            # omni.physx extension can auto-attach to this stage and independently simulate any
            # UsdPhysics.RigidBodyAPI-tagged prim on it, entirely disconnected from Newton's own
            # physics -- see KitRenderBackend.render(), which detaches PhysX from the stage every tick
            # to prevent this. That's handled there (every render tick) rather than here (construction
            # time only) because the attach can also happen later, e.g. when an object is added after
            # the timeline is already playing.
            lazy.isaacsim.core.utils.stage.create_new_stage()
            self._stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        else:
            self._stage = lazy.pxr.Usd.Stage.CreateInMemory()

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
