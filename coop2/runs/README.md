# Experiment output

Default `--output-root` for `coop2/experiment/run_{individual,centralized,broadcast_chain}.py`.
One subdirectory per run, named
`<topology>_agents<N>_repair_<on|off>_seed<S>_<timestamp>`.

Each contains:

| file | what it is |
|---|---|
| `coop2_metrics.json` / `.csv` | the nine COOP² metric sections; `constraints` holds C⁺/C⁻ |
| `coop2_process_log.json` | per-step process trace (the largest file by far) |
| `task_states.json` | per-step task snapshots; `compute_metrics` reads constraint changes from here |
| `plan_logs.json` | one record per plan, with its actions and outcomes |
| `agent_states.json`, `llm_usage.json`, `message_log.json` | FSM history, token/API accounting, inter-agent messages |
| `metrics_timeline.png`, `agent_timeline.png`, `comprehensive_timeline.png` | the plots |

Contents are gitignored: reproducible from the code and the seed.
