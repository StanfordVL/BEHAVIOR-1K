"""
Run LLM agents with the Individual topology.

Uses LLMIndividualAgent from comm_topology.llm_individual
with NO communication - each agent plans independently using LLM.

Decision Flow:
    1. Observe environment
    2. Generate plan using LLM (no waiting, no messages)
    3. Execute plan
"""

import sys
import os
import json
import threading
import time
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.behavior_env.coop_env import CooperativeEnv
from coop2.cognitive import (
    PlanningEnvWrapper, 
    LLMClient,
    print_plan_summary,
    visualize_comprehensive_timeline,
    build_repair_intervention_report,
    compute_all_metrics,
    print_metrics_summary,
    save_metrics,
    save_metrics_csv,
)
from coop2.cognitive.viz import RealtimeVisualizationWrapper
from coop2.cognitive.agent.llm_io_log import LLMIORecorder
from coop2.experiment.agent_timeline import plot_from_run_dir
from coop2.comm_topology import LLMIndividualAgent, create_llm_individual_topology
try:
    from llm_usage import print_llm_usage_summary
except ImportError:
    from coop2.experiment.llm_usage import print_llm_usage_summary


def run_individual_experiment(
    n_agents=2,
    max_steps=200,
    verbose=True,
    show_viz=False,
    llm_verbose=True,
    coop2_precheck_enabled=False,
    goal_instruction="",
    llm_model=None,
    time_limit_seconds=120,
    seed=42,
    record_video=True,
    output_root=None,
    scene_model=None,
    room=None,
    objects=None,
    bddl_activity=None,
    bddl_instance_id=0,
    headless=True,
    keep_viewer=False,
):
    """
    Run experiment with individual LLM agents (no communication).
    
    Args:
        n_agents: Number of agents
        max_steps: Maximum environment steps
        verbose: Whether to print verbose output
        show_viz: Whether to show live visualization
        llm_verbose: If True, print LLM API inputs and outputs
        coop2_precheck_enabled: If True, enable COOP2 pre-execution repair
        goal_instruction: Optional global goal added to agent prompts
        llm_model: Optional model/deployment override
        time_limit_seconds: Wall-clock budget for the score task. Disable with 0 or None.
        seed: Environment random seed.
        record_video: If True, record and save an episode GIF.
        output_root: Optional directory where this run directory should be created.
    """
    print("="*80)
    print("INDIVIDUAL TOPOLOGY WITH LLM AGENTS")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    repair_label = "repair_on" if coop2_precheck_enabled else "repair_off"
    # coop2/runs/ -- one folder per run, next to the code rather than in
    # /tmp, so a reboot does not take the experiment data with it.
    results_root = output_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs'
    )
    output_dir = os.path.join(
        results_root,
        f'individual_agents{n_agents}_{repair_label}_seed{seed}_{timestamp}',
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}\n")
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm_client = LLMClient.from_env(model=llm_model, verbose=llm_verbose)
    # Every prompt and completion of the run, one JSON object per line, flushed
    # as it goes. Attached to the client because all the agents share it; the
    # agent supplies its own id and env_step when it records.
    llm_client.io_recorder = LLMIORecorder(os.path.join(output_dir, "llm_calls.jsonl"))
    print(f"  Model: {llm_client.model}")
    if llm_verbose:
        print("  LLM verbose mode: ON (showing API inputs/outputs)")
    
    # Create environment
    agent_names = [f"agent_{i}" for i in range(n_agents)]
    env_kwargs = dict(
        area=(64, 64),
        view=(9, 9),
        size=(84, 84),
        reward=True,
        length=max_steps,
        n_players=n_agents,
        seed=seed,
        coop_config_path="paper",
    )
    # BEHAVIOR-only knobs. The crafter runner had no scene and no objects at
    # all, so leaving these unset yields a scene with none of the goal's
    # objects in it: every plan then fails at grounding with INVALID_TARGET
    # before the engine ever assigns a primitive.
    #
    # `objects` has no CLI flag. It used to (--n-objects, which scattered N
    # apples in the team's room) and that only made sense before BDDL: an
    # activity's cached template already contains the objects its goal refers
    # to, placed to satisfy its own initial conditions, so adding more from
    # the command line produced four apples for a two-apple task and nothing
    # in the code stopped it. Programmatic callers that genuinely have no
    # activity -- the offline runner harness -- still pass a list here.
    if scene_model is not None:
        env_kwargs["scene_model"] = scene_model
    if room is not None:
        env_kwargs["room"] = room
    if objects is not None:
        env_kwargs["objects"] = objects
    if record_video:
        env_kwargs["video_path"] = os.path.join(output_dir, "episode.mp4")
    # Explicit, because CooperativeBehaviorEnv._build sets gm.HEADLESS itself
    # when this is True -- so OMNIGIBSON_HEADLESS alone cannot open a window.
    env_kwargs["headless"] = headless
    if bddl_activity is not None:
        # The activity's goal expression is the only thing that can end an
        # episode early; there is no proxy check any more.
        env_kwargs["bddl_activity"] = bddl_activity
        env_kwargs["bddl_instance_id"] = bddl_instance_id
    base_env = CooperativeEnv(**env_kwargs)
    # Diagnostic hook: keep a handle so a harness can inspect counters after
    # the run without threading a return value through.
    import coop2.experiment.run_individual as _self
    _self._last_base_env = base_env
    if hasattr(base_env, "set_team_score_time_limit"):
        base_env.set_team_score_time_limit(time_limit_seconds)
    
    # Create individual LLM agents
    print(f"\nCreating individual topology with {n_agents} agents...")
    print(f"  All agents operate independently (no communication)")
    if coop2_precheck_enabled:
        print("  COOP2 repair: enabled")
    if time_limit_seconds:
        print(f"  Team-score deadline: {max_steps} env steps or {time_limit_seconds} seconds, whichever comes first")
    if goal_instruction:
        print(f"  Goal: {goal_instruction}")
    
    agents = create_llm_individual_topology(
        n_agents=n_agents,
        llm_client=llm_client,
        temperature=0.7,
        verbose=verbose,
        goal_instruction=goal_instruction,
    )
    
    # Wrap with planning environment
    plan_env = PlanningEnvWrapper(
        base_env,
        agent_names=agent_names,
        agents=agents,
        log_file=os.path.join(output_dir, "plan_logs.json"),
        interrupt_on_message=False,  # No messages in individual topology
        coop2_precheck_enabled=coop2_precheck_enabled,
    )
    
    # Wrap with visualization
    env = RealtimeVisualizationWrapper(plan_env, record_video=record_video, show=show_viz)
    
    # Run the experiment
    print("\n" + "="*80)
    print("STARTING EXPERIMENT")
    print("="*80)
    
    obs_dict, info = env.reset()
    done = False
    timed_out = False
    deadline = None
    if time_limit_seconds and time_limit_seconds > 0:
        deadline = time.monotonic() + float(time_limit_seconds)
    
    running_threads: Dict[str, threading.Thread] = {}
    
    try:
        while env.current_step < max_steps and not done:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break

            while not all(env.agents[aid].ready for aid in agent_names):
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break

                for aid in agent_names:
                    agent = env.agents[aid]
                    if aid not in running_threads or not running_threads[aid].is_alive():
                        running_threads[aid] = threading.Thread(target=agent.create_agent_thread)
                        running_threads[aid].start()
                env.wait_for_state_change(timeout=0.05)

            if timed_out:
                break
            
            for t in running_threads.values():
                t.join()
            running_threads.clear()

            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            
            obs_dict, rewards, terminated, truncated, info = env.step()
            
            if env.current_step % 10 == 0:
                print(f"\n--- Step {env.current_step}/{max_steps} ---")
                for agent_id, agent in agents.items():
                    print(f"  {agent_id}: {agent.api_calls} API calls, {agent.total_tokens_used} tokens")
            
            done = any(terminated.values()) or any(truncated.values())
    
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    
    last_step = env.current_step
    
    print(f"\n{'='*80}")
    print(f"Episode complete at step {last_step}")
    if timed_out:
        print(f"Stopped because wall-clock limit reached ({time_limit_seconds} seconds)")
    print(f"{'='*80}")
    
    env.terminate_unfinished_plans()
    env.save_logs(output_dir)

    if coop2_precheck_enabled:
        report_path = build_repair_intervention_report(output_dir)
        if report_path:
            print(f"COOP2 repair process figure: {report_path}")

    stats = env.get_plan_statistics()
    print(f"\nPlan Statistics:")
    print(f"  Total Plans: {stats['total_plans']}")
    if stats['total_plans'] > 0:
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"  Average Duration: {stats['average_duration']:.1f} steps")
    
    usage_summary = print_llm_usage_summary(
        agents,
        role_getter=lambda _agent_id, _agent: "individual",
    )
    total_api_calls = usage_summary["total_api_calls"]
    total_tokens = usage_summary["total_tokens"]
    
    print_plan_summary(env.logger.plan_history)
    
    # save_logs() above wrote agent_states.json; the timeline is drawn from that
    # file rather than from plan_history, so it shows the FSM (R/W/X/I) and not
    # just plan boundaries.
    plot_from_run_dir(output_dir)
    
    comprehensive_timeline_path = os.path.join(output_dir, 'comprehensive_timeline.png')
    visualize_comprehensive_timeline(
        env.agents, env.logger.plan_history, last_step,
        env.message_broker.get_message_log(),
        output_path=comprehensive_timeline_path
    )
    
    # No save_video call here: nothing implements it. The facade records as it
    # ticks (video_path below) and finalises the file in close(), because the
    # encoder has to be flushed and og.shutdown() never unwinds.
    env.close()
    
    llm_stats_path = os.path.join(output_dir, 'llm_usage.json')
    llm_stats = {
        'model': llm_client.model,
        'topology': 'individual',
        'max_steps': max_steps,
        'time_limit_seconds': time_limit_seconds,
        'timed_out': timed_out,
        'seed': seed,
        'record_video': record_video,
        **usage_summary,
    }
    with open(llm_stats_path, 'w') as f:
        json.dump(llm_stats, f, indent=2)

    # Compute and save COOP² metrics after llm_usage.json exists.
    print("\nComputing COOP² metrics...")
    metrics = compute_all_metrics(output_dir)
    print_metrics_summary(metrics)
    save_metrics(metrics, os.path.join(output_dir, 'coop2_metrics.json'))
    save_metrics_csv(metrics, os.path.join(output_dir, 'coop2_metrics.csv'))
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")

    # Last, so that everything above is already on disk: this blocks until the
    # user interrupts it, and Ctrl+C during it must not cost them the run.
    if keep_viewer:
        base_env.keep_viewer_open()

    return agents, env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run individual LLM experiment")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--steps", type=int, default=200, help="Maximum environment steps")
    parser.add_argument("--time-limit-seconds", type=float, default=120, help="Wall-clock time budget; use 0 to disable")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--show", action="store_true", help="Show live visualization")
    parser.add_argument("--coop2-repair", action="store_true", help="Enable COOP2 pre-execution repair")
    parser.add_argument("--goal", type=str, default="", help="Global goal instruction for agent prompts")
    parser.add_argument("--model", type=str, default=None, help="LLM model/deployment override")
    parser.add_argument("--llm-quiet", action="store_true", help="Hide raw LLM API prompts and responses")
    parser.add_argument("--seed", type=int, default=42, help="Environment random seed")
    parser.add_argument("--no-video", action="store_true", help="Skip episode GIF recording/saving")
    parser.add_argument("--output-root", type=str, default=None, help="Directory where the run result folder is created")
    parser.add_argument("--scene", type=str, default=None, help="Scene model override")
    parser.add_argument("--room", type=str, default=None, help="Room to place the team and objects in")
    parser.add_argument("--bddl-activity", type=str, default=None, metavar="NAME",
                        help="BDDL activity to load from its cached instance, e.g. "
                             "coop_two_apples_pomaria. Its goal expression is what decides "
                             "when the episode terminates; without it nothing does, and the "
                             "run ends only on the step or wall-clock limit.")
    parser.add_argument("--bddl-instance-id", type=int, default=0,
                        help="activity_instance_id of the cached template to load")
    parser.add_argument("--gui", action="store_true",
                        help="Open the Isaac Sim viewport. Needs a DISPLAY, and note that "
                             "--show is a no-op: the visualisation wrapper is a stub, and "
                             "OMNIGIBSON_HEADLESS is overridden by the env's own headless "
                             "default, so this flag is the only way to get a window.")
    parser.add_argument("--keep-viewer", action="store_true",
                        help="After the episode, keep the window open and keep printing "
                             "task-object positions until Ctrl+C. Implies --gui.")

    args = parser.parse_args()
    
    run_individual_experiment(
        n_agents=args.agents,
        max_steps=args.steps,
        verbose=not args.quiet,
        show_viz=args.show,
        llm_verbose=not args.llm_quiet,
        coop2_precheck_enabled=args.coop2_repair,
        goal_instruction=args.goal,
        llm_model=args.model,
        time_limit_seconds=args.time_limit_seconds,
        seed=args.seed,
        record_video=not args.no_video,
        output_root=args.output_root,
        scene_model=args.scene,
        room=args.room,
        bddl_activity=args.bddl_activity,
        bddl_instance_id=args.bddl_instance_id,
        headless=not (args.gui or args.keep_viewer),
        keep_viewer=args.keep_viewer,
    )
