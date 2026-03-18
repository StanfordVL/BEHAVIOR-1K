#!/usr/bin/env python3
"""Convert a cProfile .prof file into an HTML summary.

This uses snakeviz to parse profile stats and render function tables inspired by
https://gist.github.com/MSSandroid/6402e2e99e31633386a312b283839e0d.
"""

import argparse
import html
import os
from pstats import Stats


def _format_time(seconds):
    return f"{seconds:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Path to .prof file")
    parser.add_argument("--output", required=True, help="Output html path")
    parser.add_argument("--limit", type=int, default=200, help="Number of functions to include")
    args = parser.parse_args()

    import snakeviz.stats as snakeviz_stats

    stats = Stats(args.profile)
    snake_stats = snakeviz_stats.json_stats(stats)

    rows = []
    for idx, (func_name, data) in enumerate(sorted(snake_stats.items(), key=lambda item: item[1]["stats"][3], reverse=True)):
        if idx >= args.limit:
            break
        ccalls, ncalls, tottime, cumtime = data["stats"]
        rows.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td><code>{html.escape(func_name)}</code></td>"
            f"<td>{ncalls}</td>"
            f"<td>{ccalls}</td>"
            f"<td>{_format_time(tottime)}</td>"
            f"<td>{_format_time(cumtime)}</td>"
            "</tr>"
        )

    total_calls = stats.total_calls
    primitive_calls = stats.prim_calls
    total_time = stats.total_tt

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(
            """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>cProfile report</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f5f5f5; text-align: left; }
    code { font-size: 12px; }
  </style>
</head>
<body>
"""
        )
        f.write(f"<h1>cProfile report: {html.escape(os.path.basename(args.profile))}</h1>\n")
        f.write("<p>Generated with <code>snakeviz.stats.json_stats</code> and <code>pstats.Stats</code>.</p>\n")
        f.write(
            f"<ul><li>Total calls: {total_calls}</li><li>Primitive calls: {primitive_calls}</li><li>Total time (s): {total_time:.6f}</li></ul>\n"
        )
        f.write(
            "<table><thead><tr><th>#</th><th>Function</th><th>Calls</th><th>Primitive Calls</th><th>Total Time (s)</th><th>Cumulative Time (s)</th></tr></thead><tbody>"
        )
        f.write("\n".join(rows))
        f.write("</tbody></table>\n</body></html>\n")


if __name__ == "__main__":
    main()
