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
| `llm_calls.jsonl` | **every LLM call: the prompt in, the completion out**, one JSON object per line, attributed to an agent and an env_step. The only record of what an agent was actually told -- the goal text reaches the run folder through nothing else. Flushed per call, so a killed run keeps the calls it had made; read it with `coop2.cognitive.agent.llm_io_log.read_llm_calls`, which skips a truncated final line. Budget ~35 kB per call. |
| `agent_timeline.png` | per-agent FSM state (R/W/X/I) against wall clock, drawn from `agent_states.json` |

`metrics_timeline.png` is no longer produced -- it plotted the COOP2 task
tracker's C+/C- series, which is out of scope for this project.

Contents are gitignored: reproducible from the code and the seed.

Delete accumulated runs with `clean_runs.py` (dry run by default):

```bash
python coop2/runs/clean_runs.py                  # show what would go
python coop2/runs/clean_runs.py --keep 3 --yes   # delete all but the 3 newest
```
