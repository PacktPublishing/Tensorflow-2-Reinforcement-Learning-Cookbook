import matplotlib.pyplot as plt
import numpy as np


def value_function_plot(q_table, env, isMaze=True, arrow=True):
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
        idx2cell = env.idx2cell
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
