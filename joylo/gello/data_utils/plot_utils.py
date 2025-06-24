import matplotlib.pyplot as plt
import numpy as np


def plot_in_grid(vals: np.ndarray, save_path: str):
    """Plot the trajectories in a grid.

    Args:
        vals: B x T x N, where
            B is the number of trajectories,
            T is the number of timesteps,
            N is the dimensionality of the values.
        save_path: path to save the plot.
    """
    B = len(vals)
    N = vals[0].shape[-1]
    # fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    rows = (N // 4) + (N % 4 > 0)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 10))
    for b in range(B):
        curr = vals[b]
        for i in range(N):
            T = curr.shape[0]
            # give them transparency
            axes[i // 4, i % 4].plot(np.arange(T), curr[:, i], alpha=0.5)

    for i in range(N):
        axes[i // 4, i % 4].set_title(f"Dim {i}")
        axes[i // 4, i % 4].set_ylim([-1.0, 1.0])

    plt.savefig(save_path)
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(141, projection="3d")
    for b in range(B):
        curr = vals[b]
        ax.plot(curr[:, 0], curr[:, 1], curr[:, 2], alpha=0.75)
        # scatter the start and end points
        ax.scatter(curr[0, 0], curr[0, 1], curr[0, 2], c="r")
        ax.scatter(curr[-1, 0], curr[-1, 1], curr[-1, 2], c="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(["Trajectory", "Start", "End"])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # get the 2D view of the XY plane, with X pointing downwards
    ax.view_init(270, 0)

    ax = fig.add_subplot(142, projection="3d")
    for b in range(B):
        curr = vals[b]
        ax.plot(curr[:, 0], curr[:, 1], curr[:, 2], alpha=0.75)
        ax.scatter(curr[0, 0], curr[0, 1], curr[0, 2], c="r")
        ax.scatter(curr[-1, 0], curr[-1, 1], curr[-1, 2], c="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(["Trajectory", "Start", "End"])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # get the 2D view of the XZ plane, with X pointing leftwards
    ax.view_init(0, 0)

    ax = fig.add_subplot(143, projection="3d")
    for b in range(B):
        curr = vals[b]
        ax.plot(curr[:, 0], curr[:, 1], curr[:, 2], alpha=0.75)
        ax.scatter(curr[0, 0], curr[0, 1], curr[0, 2], c="r")
        ax.scatter(curr[-1, 0], curr[-1, 1], curr[-1, 2], c="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(["Trajectory", "Start", "End"])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # get the 2D view of the YZ plane, with Y pointing leftwards
    ax.view_init(0, 90)

    ax = fig.add_subplot(144, projection="3d")
    for b in range(B):
        curr = vals[b]
        ax.plot(curr[:, 0], curr[:, 1], curr[:, 2], alpha=0.75)
        ax.scatter(curr[0, 0], curr[0, 1], curr[0, 2], c="r")
        ax.scatter(curr[-1, 0], curr[-1, 1], curr[-1, 2], c="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(["Trajectory", "Start", "End"])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    plt.savefig(save_path[:-4] + "_3d.png")

    plt.close()
