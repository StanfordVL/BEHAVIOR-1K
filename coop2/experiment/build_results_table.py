"""Build the current COOP2 experiment results table.

The reported constraint deficits are computed from concrete task attempts:
- spatial: 1 - min(agents near task / required_agents, 1)
- temporal: 1 - min(agents issuing targeted collect / required_agents, 1)
- dependency: 1 - min(agents with required tool or capability / required_agents, 1)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from coop2.cognitive.coop2_attempt_events import (
    ATTEMPT_CONSTRAINTS,
    aggregate_attempt_violation_rates,
    extract_attempt_constraint_events,
)


RESULT_RE = re.compile(
    r"^(?P<topology>individual|centralized|broadcast_chain)_agents(?P<agents>\d+)_"
    r"repair_(?P<repair>on|off)_seed(?P<seed>\d+)_"
)

TOPOLOGY_ORDER = ["individual", "centralized", "broadcast_chain"]
LLAMA_SCOUT_MODEL = "Llama-4-Scout-17B-16E-Instruct"
MODEL_ORDER = ["gpt-5.4-mini", LLAMA_SCOUT_MODEL, "gpt-5.4"]
MODEL_LABELS = {
    LLAMA_SCOUT_MODEL: "Llama-Scout",
}
TOPOLOGY_LABELS = {
    "individual": "Individual",
    "centralized": "Centralized",
    "broadcast_chain": "Broadcast Chain",
}
DEFAULT_AGENTS = [3]
TABLE_METRICS = [
    "score",
    "steps",
    "score_per_step",
    "total_plans",
    "plans_per_agent",
    "messages",
    "interruptions",
    "total_decision_time",
    "spatial_violation_rate",
    "temporal_violation_rate",
    "dependency_violation_rate",
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _plan_count(path: Path) -> int:
    plan_logs = _read_json(path)
    return len(plan_logs.get("plan_history") or [])


def _agent_interrupt_count(agent_states: Dict[str, Any]) -> int:
    """Count entries into the MAEIL interrupted stage from agent state logs."""
    if not isinstance(agent_states, dict):
        return 0
    count = 0
    for states in agent_states.values():
        if not isinstance(states, list):
            continue
        count += sum(1 for entry in states if len(entry) >= 3 and entry[2] == "interrupted")
    return count


def _process_constraint_violation_rates(folder: Path) -> Dict[str, float]:
    """Average constraint deficits over concrete task-attempt events."""
    process_log = _read_json(folder / "coop2_process_log.json")
    if not isinstance(process_log, list):
        process_log = []

    attempt_events = extract_attempt_constraint_events(process_log)
    if attempt_events:
        return aggregate_attempt_violation_rates(attempt_events)

    empty_rates = {"attempt_constraint_events": 0.0}
    for constraint in ATTEMPT_CONSTRAINTS:
        empty_rates[f"{constraint}_violation_rate"] = 0.0
        empty_rates[f"{constraint}_score"] = 0.0
        empty_rates[f"{constraint}_checks"] = 0
    return empty_rates


def load_rows(results_dir: Path, dedupe_latest: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        match = RESULT_RE.match(folder.name)
        if not match:
            continue

        metrics_path = folder / "coop2_metrics.json"
        score_path = folder / "team_score.json"
        usage_path = folder / "llm_usage.json"
        if not (metrics_path.exists() and score_path.exists() and usage_path.exists()):
            continue

        groups = match.groupdict()
        metrics = _read_json(metrics_path)
        team_score = _read_json(score_path)
        usage = _read_json(usage_path)
        agent_states = _read_json(folder / "agent_states.json")

        communication = metrics.get("communication") or {}
        decision = metrics.get("decision_overhead") or {}
        decision_per_agent = decision.get("decision_time_per_agent") or {}
        steps = _safe_int(team_score.get("current_step") or metrics.get("team_score", {}).get("current_step"))

        row = {
            "run_id": folder.name,
            "path": str(folder),
            "mtime": folder.stat().st_mtime,
            "agents": _safe_int(groups["agents"]),
            "model": str(usage.get("model") or "unknown"),
            "topology": groups["topology"],
            "repair": groups["repair"],
            "seed": _safe_int(groups["seed"]),
            "score": _safe_float(team_score.get("total_score") or metrics.get("team_score", {}).get("total_score")),
            "steps": steps,
            "total_plans": _plan_count(folder / "plan_logs.json"),
            "messages": _safe_int(communication.get("L_tot")),
            "interruptions": _agent_interrupt_count(agent_states),
            "total_decision_time": sum(_safe_float(value) for value in decision_per_agent.values()),
            **_process_constraint_violation_rates(folder),
        }
        row["score_per_step"] = row["score"] / row["steps"] if row["steps"] else 0.0
        row["plans_per_agent"] = row["total_plans"] / row["agents"] if row["agents"] else 0.0
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("mtime")
    if dedupe_latest:
        key_cols = ["agents", "model", "topology", "repair", "seed"]
        df = df.groupby(key_cols, as_index=False).tail(1).reset_index(drop=True)
    df["topology"] = pd.Categorical(df["topology"], TOPOLOGY_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER + sorted(set(df["model"]) - set(MODEL_ORDER)), ordered=True)
    return df.sort_values(["agents", "model", "topology", "repair", "seed"]).reset_index(drop=True)


def _filter_values(df: pd.DataFrame, column: str, values: Optional[Iterable[str]]) -> pd.DataFrame:
    if values is None or df.empty or column not in df.columns:
        return df
    requested = {str(value) for value in values}
    return df[df[column].astype(str).isin(requested)]


def filter_rows(
    df: pd.DataFrame,
    models: Optional[List[str]],
    topologies: Optional[List[str]],
    agents: Optional[List[int]],
    seeds: Optional[List[int]],
    repair: str,
) -> pd.DataFrame:
    filtered = _filter_values(df, "model", models)
    filtered = _filter_values(filtered, "topology", topologies)
    if agents is not None and not filtered.empty:
        filtered = filtered[filtered["agents"].isin(agents)]
    if seeds is not None and not filtered.empty:
        filtered = filtered[filtered["seed"].isin(seeds)]
    if repair != "both" and not filtered.empty:
        filtered = filtered[filtered["repair"].astype(str) == repair]
    return filtered.reset_index(drop=True)


def aggregate_runs(df: pd.DataFrame, aggregate_repeats: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    group_cols = ["agents", "model", "topology", "repair"]
    if not aggregate_repeats:
        group_cols.append("seed")

    grouped = df.groupby(group_cols, observed=True)
    mean_df = grouped[TABLE_METRICS].mean().reset_index()
    mean_df["runs"] = grouped.size().to_numpy()
    if aggregate_repeats:
        std_df = grouped[TABLE_METRICS].std(ddof=1).reset_index(drop=True).fillna(0.0)
        counts = mean_df["runs"].clip(lower=1)
        for metric in TABLE_METRICS:
            # Normal approximation is sufficient for a compact experiment
            # summary; the raw rows are saved for exact bootstrap/t-tests.
            mean_df[f"{metric}_ci95"] = 1.96 * std_df[metric] / (counts ** 0.5)
    else:
        for metric in TABLE_METRICS:
            mean_df[f"{metric}_ci95"] = 0.0
    ordered = group_cols + ["runs"] + [
        value
        for metric in TABLE_METRICS
        for value in (metric, f"{metric}_ci95")
    ]
    return mean_df[ordered]


def _fmt_mean_ci(row: pd.Series, metric: str, decimals: int = 1) -> str:
    mean = _safe_float(row.get(metric))
    ci = _safe_float(row.get(f"{metric}_ci95"))
    if ci <= 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {ci:.{decimals}f}"


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return pd.DataFrame(
        {
            "Agents": df["agents"].astype(int),
            "Model": df["model"].astype(str).map(lambda value: MODEL_LABELS.get(value, value)),
            "Topology": df["topology"].astype(str).map(lambda value: TOPOLOGY_LABELS.get(value, value.title())),
            "Runs": df["runs"].astype(int),
            "Score": df.apply(lambda row: _fmt_mean_ci(row, "score", 1), axis=1),
            "Steps": df.apply(lambda row: _fmt_mean_ci(row, "steps", 1), axis=1),
            "Score/Step": df.apply(lambda row: _fmt_mean_ci(row, "score_per_step", 2), axis=1),
            "Plans/Agent": df.apply(lambda row: _fmt_mean_ci(row, "plans_per_agent", 1), axis=1),
            "Msg.": df.apply(lambda row: _fmt_mean_ci(row, "messages", 1), axis=1),
            "Intr.": df.apply(lambda row: _fmt_mean_ci(row, "interruptions", 1), axis=1),
            "Total Dec.": df.apply(lambda row: _fmt_mean_ci(row, "total_decision_time", 1), axis=1),
            "Spat. Viol.": df.apply(lambda row: _fmt_mean_ci(row, "spatial_violation_rate", 2), axis=1),
            "Temp. Viol.": df.apply(lambda row: _fmt_mean_ci(row, "temporal_violation_rate", 2), axis=1),
            "Dep. Viol.": df.apply(lambda row: _fmt_mean_ci(row, "dependency_violation_rate", 2), axis=1),
        }
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple GitHub-flavored Markdown table without optional deps."""
    if df.empty:
        return ""
    headers = [str(column).replace("\n", " ") for column in df.columns]
    rows = [
        [str(value).replace("\n", " ") for value in row]
        for row in df.to_numpy().tolist()
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: List[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator] + [render_row(row) for row in rows])


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"$\pm$",
    }
    return "".join(replacements.get(char, char) for char in text)


def dataframe_to_latex(
    df: pd.DataFrame,
    *,
    caption: str = "MA-Crafter results.",
    label: str = "tab:macrafter_results",
) -> str:
    if df.empty:
        return ""
    column_spec = "lll" + "r" * (len(df.columns) - 3)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in df.columns) + r" \\",
        r"\midrule",
    ]
    previous_group = None
    for _, row in df.iterrows():
        group = (row.get("Agents"), row.get("Model"))
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        lines.append(" & ".join(_latex_escape(value) for value in row.tolist()) + r" \\")
        previous_group = group
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the current COOP2 aggregate results table.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=[Path(__file__).resolve().parent / "results"],
        help="One or more result roots containing completed run folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repair", choices=["off", "on", "both"], default="off")
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--topologies", nargs="*", default=TOPOLOGY_ORDER)
    parser.add_argument("--agents", nargs="*", type=int, default=DEFAULT_AGENTS)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42],
        help="Environment seeds to include. Default: 42 for matched-condition comparisons.",
    )
    parser.add_argument(
        "--include-repeats",
        action="store_true",
        help="Keep repeated runs with the same condition/seed instead of taking only the latest.",
    )
    parser.add_argument(
        "--aggregate-repeats",
        action="store_true",
        help="Aggregate over repeated runs by agents/model/topology/repair and report 95%% CIs.",
    )
    parser.add_argument("--hide-runs", action="store_true")
    parser.add_argument("--latex-caption", default="MA-Crafter results.")
    parser.add_argument("--latex-label", default="tab:macrafter_results")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir[0] / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "all" if args.seeds is None else "_".join(str(seed) for seed in args.seeds)

    loaded = [
        load_rows(results_dir, dedupe_latest=not args.include_repeats)
        for results_dir in args.results_dir
    ]
    rows = pd.concat([df for df in loaded if not df.empty], ignore_index=True) if loaded else pd.DataFrame()
    rows = filter_rows(rows, args.models, args.topologies, args.agents, args.seeds, args.repair)
    if rows.empty:
        print("No completed runs matched the requested filters.")
        return 1

    raw_path = output_dir / f"coop2_rows_repair_{args.repair}_seed_{seed_label}.csv"
    rows.to_csv(raw_path, index=False)

    aggregate = aggregate_runs(rows, aggregate_repeats=args.aggregate_repeats)
    aggregate_path = output_dir / f"coop2_table_repair_{args.repair}_seed_{seed_label}.csv"
    aggregate.to_csv(aggregate_path, index=False)

    table = format_table(aggregate)
    if args.hide_runs and "Runs" in table.columns:
        table = table.drop(columns=["Runs"])
    markdown_path = output_dir / f"coop2_table_repair_{args.repair}_seed_{seed_label}.md"
    markdown_path.write_text(dataframe_to_markdown(table), encoding="utf-8")
    latex_path = output_dir / f"coop2_table_repair_{args.repair}_seed_{seed_label}.tex"
    latex_path.write_text(
        dataframe_to_latex(table, caption=args.latex_caption, label=args.latex_label),
        encoding="utf-8",
    )

    print(f"Raw rows: {raw_path}")
    print(f"Aggregate CSV: {aggregate_path}")
    print(f"Markdown table: {markdown_path}")
    print(f"LaTeX table: {latex_path}")
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
