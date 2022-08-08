""" 
Code to plot realtime graph. 

To use, run the iGibson simulator loop with simulator.step(print_stats=True)
and redirect stdout to a file:

python my_gibson_test.py > log.txt

and provide log.txt to the profiling script via the --filename arg
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Process timing data from step_simulation")
    parser.add_argument("--filename", type=str, nargs="?", help="path to dumped stdout")
    parser.add_argument("--plotname", type=str, nargs="?", help="name for plot")
    return parser.parse_args()


def main():
    args = parse_args()
    fname = args.filename
    if args.plotname != None:
        title = args.plotname
    else:
        title = os.path.splitext(os.path.basename(args.filename))[0]
    log = open(fname, "r")
    lines = log.readlines()
    lines = [l.strip() for l in lines]
    first_frame_break = lines.index("-------------------------")
    last_frame_break = len(lines) - lines[::-1].index("-------------------------")
    lines = lines[first_frame_break + 1 : last_frame_break]

    metrics = {
        "Out of Step": "Total out-of-step",
        "Physics": "Total physics",
        "Non-Physics": "Total non-physics",
        "Render": "Total render",
        "Sleep": "Total sleep",
        "VR system": "Total VR system",
        "Frame": "Total frame",
        "Realtime Factor": "Realtime",
    }

    met_data = {}
    for m in metrics.keys():
        met_data[m] = []

    for l in lines:
        for m in metrics.keys():
            if l.find(metrics[m]) == 0:
                c_idx = l.find(":")
                end_idx = None if m != "Frame" else l.find("and") - 1
                val = float(l[c_idx + 2 : end_idx])
                met_data[m].append(val)

    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    ax1, ax2 = axs
    ax1.set(xlabel="Frame", ylabel="Time (ms)")

    for m in metrics.keys():
        if m == "Realtime Factor":
            continue
        data = met_data[m]
        ax1.plot(range(len(data)), data, label="{} - av: {}".format(m, round(np.mean(np.array(data)), 2)))

    ax1.legend(loc="upper right")

    ax2.set(xlabel="Frame", ylabel="Realtime Factor")
    rt_factor = met_data["Realtime Factor"]
    ax2.plot(range(len(rt_factor)), rt_factor)
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


if __name__ == "__main__":
    main()
