"""Policy evaluation module for vid2scene.

Integrates LeRobot policy evaluation with BEHAVIOR-1K OmniGibson environments.
Supports any LeRobot-compatible policy (Diffusion, ACT, etc.).

Usage:
    from vid2scene_policy.policy_evaluation import evaluate_policy, EvalConfig

    config = EvalConfig(
        policy_path="path/to/checkpoint",
        scene_model="Rs_int",
        n_episodes=10,
    )
    results = evaluate_policy(config)
"""

from .omnigibson_eval_env import OmniGibsonEvalEnv, EpisodeMetadata, EnvConfig
from .eval import evaluate_policy, EvalConfig, load_policy, rollout_episode

__all__ = [
    "OmniGibsonEvalEnv",
    "EpisodeMetadata",
    "EnvConfig",
    "evaluate_policy",
    "EvalConfig",
    "load_policy",
    "rollout_episode",
]
