from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import EVAL_CAMERA_ROLES, HEAD_RESOLUTION, WRIST_RESOLUTION


class RGBDFullResWrapper(EnvironmentWrapper):
    """
    Eval wrapper: full-resolution RGB-D observations (head at HEAD_RESOLUTION, wrists at
    WRIST_RESOLUTION) matching the data-collection cameras, plus a depth modality.

    As with :class:`~omnigibson.eval.wrappers.default_wrapper.DefaultWrapper`, camera resolution and
    modalities are declared via :meth:`camera_spec` and baked into the robot config at env CREATION
    (multi-env tiled rendering fixes resolution at creation, so a runtime resize would not apply).

    Args:
        env (og.Environment): The environment to wrap.
    """

    @classmethod
    def camera_spec(cls) -> dict:
        """Returns {"modalities": [...], "resolution": {camera_role: (H, W)}} for this eval profile."""
        resolution = {role: (HEAD_RESOLUTION if role == "head" else WRIST_RESOLUTION) for role in EVAL_CAMERA_ROLES}
        return {"modalities": ["rgb", "depth_linear"], "resolution": resolution}

    def __init__(self, env: Environment):
        super().__init__(env=env)
