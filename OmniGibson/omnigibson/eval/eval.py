"""Websocket evaluation runner for the BEHAVIOR-1K challenge.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval \
        --task-name turning_on_radio \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval
"""

import argparse
import json
import os
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Indices into the task's Public Test Instance IDs (test_instances.csv).",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 2x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.RGBLowResWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from omnigibson.macros import gm

    gm.HEADLESS = args.headless

    # Imported after macros are set; this also pulls in OmniGibson + gello.
    from omegaconf import OmegaConf

    from omnigibson.eval.adapter import BehaviorEvalEnv, Evaluator

    instance_ids = BehaviorEvalEnv._resolve_instance_ids(args.task_name, args.instance_indices)
    print(f"Resolved test instance ids for {args.task_name}: {instance_ids}")

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": "websocket",
            "model": {
                "_target_": "omnigibson.eval.policies.WebsocketPolicy",
                "host": args.host,
                "port": args.port,
            },
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": False,
            "test_hidden": False,
            "task": {"name": args.task_name},
            "robot": {"type": "R1Pro", "controllers": None},
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)

    results = []
    with Evaluator(cfg) as evaluator:
        for instance_id in instance_ids:
            for rollout_id in range(args.num_rollouts):
                try:
                    evaluator.load_task_instance(int(instance_id), test_hidden=False)
                    evaluator.reset()
                    terminated = truncated = False
                    steps = 0
                    while not (terminated or truncated):
                        terminated, truncated = evaluator.step()
                        steps += 1

                    success = bool(evaluator.env.task.success)
                    metrics = {}
                    for metric in evaluator.metrics:
                        metrics.update(metric.aggregate(evaluator.env))

                    result = {
                        "task": args.task_name,
                        "instance_id": int(instance_id),
                        "rollout_id": rollout_id,
                        "steps": steps,
                        "success": success,
                        **metrics,
                    }
                    out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    print(
                        f"[result] instance={instance_id} rollout={rollout_id} steps={steps} "
                        f"success={success} q_score={q_score} -> {out_path}"
                    )
                    results.append(result)
                except Exception:
                    print(f"[error] instance={instance_id} rollout={rollout_id} failed:")
                    traceback.print_exc()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
    print(f"\n=== EVAL SUMMARY: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name} ===")


if __name__ == "__main__":
    main()
