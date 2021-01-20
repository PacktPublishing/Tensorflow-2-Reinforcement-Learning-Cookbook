#!/usr/bin/env python
# Utility functions to visualize value functions
# Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_maze_values(q_table, env, isMaze=True, arrow=True):
    """Plot the Tabular Q-Value function

    Args:
        q_table (np.array): Tabular Q-Value function
        env (gym.Env): Gym environment with discrete space. E.g: MazeEnv
        isMaze (bool, optional): True for MazeEnv. Defaults to True.
        arrow (bool, optional): Set to True for drawing directional arrows. Defaults to True.
    """
    # (x,y) cooridnates
    direction = {
        0: (0, -0.4),
        1: (0, 0.4),
        2: (-0.4, 0),
        3: (0.4, 0),
    }
    v = np.max(q_table, axis=1)
    best_action = np.argmax(q_table, axis=1)
    if isMaze:
        idx2cell = env.index_to_coordinate_map
        for i in range(8):
            _, ax = plt.subplots()
            ax.set_axis_off()
            y_mat = np.zeros(env.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = v[8 * j + i]
                if arrow:
                    a = best_action[8 * j + i]
                    ax.arrow(
                        pos[1],
                        pos[0],
                        direction[a][0],
                        direction[a][1],
                        head_width=0.05,
                        head_length=0.1,
                        fc="g",
                        ec="g",
                    )
            y_mat[env.goal_pos] = max(v) + 0.1
            ax.imshow(y_mat, cmap="hot")
            plt.savefig(f"results/value_iter_{i}.png", bbox_inches="tight")

    else:
        n = int(np.sqrt(len(v)))
        state_value_func = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                if not (r == (n - 1) and c == (n - 1)):
                    state_value_func[r, c] = v[n * c + r]
                    if arrow:
                        d = direction[best_action[n * c + r]]
                        plt.arrow(
                            c,
                            r,
                            d[0],
                            d[1],
                            head_width=0.05,
                            head_length=0.1,
                            fc="r",
                            ec="r",
                        )
        state_value_func[env.goal_pos] = max(v[:-1]) + 0.1
        plt.imshow(state_value_func, cmap="hot")
    plt.show()


def visualize_grid_state_values(grid_state_values):
    """Visualizes the state value function for the grid"""
    plt.figure(figsize=(10, 5))
    p = sns.heatmap(
        grid_state_values,
        cmap="Greens",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 16},
        square=True,
    )
    p.set_ylim(len(grid_state_values) + 0.01, -0.01)
    plt.show()


def visualize_grid_action_values(grid_action_values):
    top = grid_action_values[:, 0].reshape((3, 4))
    top_value_positions = [
        (0.38, 0.25),
        (1.38, 0.25),
        (2.38, 0.25),
        (3.38, 0.25),
        (0.38, 1.25),
        (1.38, 1.25),
        (2.38, 1.25),
        (3.38, 1.25),
        (0.38, 2.25),
        (1.38, 2.25),
        (2.38, 2.25),
        (3.38, 2.25),
    ]
    right = grid_action_values[:, 1].reshape((3, 4))
    right_value_positions = [
        (0.65, 0.5),
        (1.65, 0.5),
        (2.65, 0.5),
        (3.65, 0.5),
        (0.65, 1.5),
        (1.65, 1.5),
        (2.65, 1.5),
        (3.65, 1.5),
        (0.65, 2.5),
        (1.65, 2.5),
        (2.65, 2.5),
        (3.65, 2.5),
    ]
    bottom = grid_action_values[:, 2].reshape((3, 4))
    bottom_value_positions = [
        (0.38, 0.8),
        (1.38, 0.8),
        (2.38, 0.8),
        (3.38, 0.8),
        (0.38, 1.8),
        (1.38, 1.8),
        (2.38, 1.8),
        (3.38, 1.8),
        (0.38, 2.8),
        (1.38, 2.8),
        (2.38, 2.8),
        (3.38, 2.8),
    ]
    left = grid_action_values[:, 3].reshape((3, 4))
    left_value_positions = [
        (0.05, 0.5),
        (1.05, 0.5),
        (2.05, 0.5),
        (3.05, 0.5),
        (0.05, 1.5),
        (1.05, 1.5),
        (2.05, 1.5),
        (3.05, 1.5),
        (0.05, 2.5),
        (1.05, 2.5),
        (2.05, 2.5),
        (3.05, 2.5),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_ylim(3, 0)
    tripcolor = plot_triangular(
        left,
        top,
        right,
        bottom,
        ax=ax,
        triplotkw={"color": "k", "lw": 1},
        tripcolorkw={"cmap": "rainbow_r"},
    )

    ax.margins(0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(tripcolor)

    for i, (xi, yi) in enumerate(top_value_positions):
        plt.text(xi, yi, round(top.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(right_value_positions):
        plt.text(xi, yi, round(right.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(left_value_positions):
        plt.text(xi, yi, round(left.flatten()[i], 2), size=11, color="w")
    for i, (xi, yi) in enumerate(bottom_value_positions):
        plt.text(xi, yi, round(bottom.flatten()[i], 2), size=11, color="w")

    plt.show()


def plot_triangular(left, bottom, right, top, ax=None, triplotkw={}, tripcolorkw={}):

    if not ax:
        ax = plt.gca()
    n = left.shape[0]
    m = left.shape[1]

    a = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]])
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))

    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5 : (k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4 : (k + 1) * 4, :] = tr + k * 5

    C = np.c_[
        left.flatten(), bottom.flatten(), right.flatten(), top.flatten()
    ].flatten()

    _ = ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor
