"""Newton-native OmniGibson Environment entry point."""

from __future__ import annotations


class Environment:
    """Canonical OmniGibson environment facade backed by the Newton simulator."""

    def __init__(self, configs, in_vec_env=False):
        from omnigibson.simulator import Simulator

        self.configs = configs if isinstance(configs, (list, tuple)) else [configs]
        self.in_vec_env = in_vec_env
        self.render_mode = "human"
        self.metadata = {"render.modes": ["human"]}
        self.action_space = None
        self.observation_space = None

        self.simulator = Simulator.from_environment_configs(self.configs)
        self._built = False
        self._closed = False

        if self.simulator.should_auto_build_environment(self.configs):
            self._ensure_built()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise NotImplementedError("Newton Environment does not support deterministic seeding yet.")
        if options:
            raise NotImplementedError("Newton Environment reset options are not implemented yet.")
        self._ensure_built()
        return {}, {"summary": self.summary()}

    def step(self, action=None):
        self._ensure_built()
        self.simulator.apply_environment_action(action)
        self.simulator.step()
        if self.simulator.viewer is not None:
            self.simulator.render()
        return {}, 0.0, False, False, {"summary": self.summary()}

    def render(self):
        self._ensure_built()
        if self.simulator.viewer is None:
            self.simulator.attach_viewer()
        self.simulator.render()

    def close(self):
        if not self._closed:
            self.simulator.close()
            self._closed = True
            self._built = False

    def summary(self):
        self._ensure_built()
        return self.simulator.summary()

    @property
    def scene(self):
        self._ensure_built()
        return self.simulator.scene

    @property
    def robots(self):
        self._ensure_built()
        return list(self.simulator.robots)

    @property
    def objects(self):
        self._ensure_built()
        return list(self.simulator.objects)

    def _ensure_built(self):
        if self._closed:
            raise RuntimeError("Cannot use a closed Environment.")
        if not self._built:
            self.simulator.build_environment()
            self._built = True

    def __enter__(self):
        self._ensure_built()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["Environment"]
