from collections import defaultdict
import io
import dateutil.parser
import numpy as np
import requests
from pybadges import badge
import matplotlib.pyplot as plt
import json
from PIL import Image


REALTIME_FPS = 30


def get_profile_data():
    # Pull the data from the website.
    data = requests.get(
        "https://stanfordvl.github.io/OmniGibson/profiling/data.js"
    ).text

    # Update the data to be a valid json string.
    data = data.replace("window.BENCHMARK_DATA = ", "")

    # Load it as a JSON
    data = json.loads(data)

    # Get all the commit IDs, sort by time.
    commits = {
        (x["commit"]["id"], dateutil.parser.isoparse(x["commit"]["timestamp"]))
        for x in data["entries"]["Benchmark"]
    }
    commits = [x[0] for x in sorted(commits, key=lambda x: x[1])]

    # Now get the data in the {plot: {series: [values]}} format.
    plots = defaultdict(lambda: defaultdict(lambda: [None] * len(commits)))
    plot_units = defaultdict(set)
    for commit_entry in data["entries"]["Benchmark"]:
        commit = commit_entry["commit"]["id"]
        commit_index = commits.index(commit)
        for datapoint in commit_entry["benches"]:
            plot_name = datapoint["extra"][0]
            series_name = datapoint["name"]
            value = datapoint["value"]
            unit = datapoint["unit"]

            plots[plot_name][series_name][commit_index] = value
            plot_units[plot_name].add(unit)

    # Validate that each plot has datapoints from a single unit.
    for plot, units in plot_units.items():
        if len(units) > 1:
            raise ValueError(f"Plot {plot} has multiple units: {units}")
    plot_units = {plot: units.pop() for plot, units in plot_units.items()}

    # Postprocess a realtimeness plot from the FPS plot
    if "FPS" in plots:
        plots["Realtime Performance"] = {
            k: [v / REALTIME_FPS if v is not None else None for v in vs]
            for k, vs in plots["FPS"].items()
        }
        plot_units["Realtime Performance"] = "x"

    return plots, commits, plot_units


def plot_profile(plot_name, last_n_points, ignore_series=None):
    plots, commits, plot_units = get_profile_data()

    if plot_name not in plots:
        raise ValueError(f"Plot {plot_name} not found.")

    series = plots[plot_name]
    series_names = [
        x for x in series.keys() if ignore_series is None or x not in ignore_series
    ]

    # Plot the series.
    plt.figure(figsize=(10, 4), dpi=300)
    for series_name in series_names:
        plt.plot(series[series_name][-last_n_points:], label=series_name)
    plt.xlabel("Commit")
    tick_labels = [x[-8:] for x in commits[-last_n_points:]]
    plt.xticks(ticks=np.arange(len(tick_labels)), labels=tick_labels, rotation=90)
    plt.ylabel(plot_units[plot_name])
    plt.title(plot_name)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.axhline(1.0, linestyle="--", c="gray", alpha=0.5)
    plt.tight_layout()

    stream = io.BytesIO()
    plt.savefig(stream, format="png")
    plt.close()
    return Image.open(stream)


def make_realtime_badge(series_name):
    plots, commits, plot_units = get_profile_data()

    realtime_coeff = [
        x for x in plots["Realtime Performance"][series_name] if x is not None
    ][-1]
    if realtime_coeff > 1:
        color = "green"
    elif realtime_coeff > 0.5:
        color = "yellow"
    else:
        color = "red"
    return badge(
        left_text=f"{series_name} realtime performance",
        right_text=f"{realtime_coeff:.2f}x",
        right_color=color,
    )


def get_profile_badge_svg():
    """Generate profile badge SVG for static site generation."""
    try:
        badge_text = make_realtime_badge("Rs_int")
        return badge_text.encode("utf-8")
    except Exception:
        return None


def get_profile_plot_png():
    """Generate profile plot PNG for static site generation."""
    try:
        plot_img = plot_profile(
            "Realtime Performance", 10, ignore_series=["Empty scene"]
        )
        stream = io.BytesIO()
        plot_img.save(stream, format="PNG")
        return stream.getvalue()
    except Exception:
        return None
