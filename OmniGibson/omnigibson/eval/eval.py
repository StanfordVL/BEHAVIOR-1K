"""Websocket evaluation runner for the BEHAVIOR-1K challenge.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval \
        --task-name turning_on_radio \
        --robot-config omnigibson/eval/r1pro.yaml \
        --mode public_test \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval --write-video
"""

import argparse
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help=(
            "Optional path to YAML/JSON file containing one complete robot config dictionary with canonical "
            "'model' and 'name' fields. Add eval.camera_sensor_names to configure eval camera roles."
        ),
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Instance indices for the selected mode. For train these are direct train instance IDs; "
            "for public_test / hidden_test these index into that 20-instance split."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "public_test", "hidden_test"),
        default="public_test",
        help="Instance split to evaluate. Default: public_test.",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel env slots. Must equal the number of selected instance indices.",
    )
    parser.add_argument(
        "--replay-action-chunk-size",
        type=int,
        default=0,
        help=(
            "Opt in to exact client-side replay of a complete server-returned action chunk. "
            "Use the receding-horizon length (for example 16); 0 keeps one websocket request per step."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 1.5x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument(
        "--policy",
        choices=("websocket", "local"),
        default="websocket",
        help="Policy backend to use. local emits zero actions and is intended for eval smoke tests.",
    )
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an MP4 rollout video (head + wrist cameras) per rollout under <output-dir>/videos.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for saved rollout videos.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gm.HEADLESS = args.headless
    # Headless evaluation consumes robot-camera observations through VisionSensor / TiledVisionSensor render
    # products, not the separate 1280x720 debug viewport. Avoid constructing that unused viewer product: at
    # N=10 it accounts for two of five RTX tile passes while policy camera shapes and traces remain unchanged.
    # The environment override restores the old path only for controlled performance A/B runs.
    if args.headless and os.getenv("OMNIGIBSON_KEEP_VIEWER_CAMERA", "0") != "1":
        gm.RENDER_VIEWER_CAMERA = False

    seed = seed_everything(DEFAULT_EVAL_SEED)
    logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    logger.info(f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}")
    if len(instance_ids) != args.num_envs:
        raise ValueError(
            "Evaluation requires exactly one instance per environment slot: "
            f"got {len(instance_ids)} instance indices and --num-envs={args.num_envs}."
        )

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
            "action_chunk_size": args.replay_action_chunk_size,
        }
    else:
        model_cfg = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "mode": args.mode,
            "seed": seed,
            "num_envs": args.num_envs,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    results = []
    with Evaluator(cfg) as evaluator:
        # Each run() pass evaluates the selected instances together in one parallel batch. The outer
        # loop repeats that same batch for additional rollouts.
        for rollout_id in range(args.num_rollouts):
            rollout_results = evaluator.run(
                [int(i) for i in instance_ids],
                write_video=args.write_video,
                video_path=video_dir,
                metrics_dir=json_dir,
                rollout_id=rollout_id,
                video_fps=args.video_fps,
            )
            results.extend(rollout_results.values())

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
    logger.info(f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
