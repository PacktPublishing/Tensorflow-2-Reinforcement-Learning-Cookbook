import numpy as np
from envs.gridworldv2 import Gridworldv2


def temporal_difference_learning(env, max_episodes):
    grid_state_values = np.zeros(len(env.distinct_states))
    grid_state_values[env.goal_state] = 1
    grid_state_values[env.bomb_state] = -1
    # v: state-value function
    v = grid_state_values
    gamma = 0.99  # Discount factor
    alpha = 0.01  # learning rate
    done = False

    for episode in range(max_episodes):
        state = env.reset()
        while not done:
            action = np.random.randint(4)  # random policy
            next_state, reward, done = env.step(action)

            # State-value function updates using TD(0)
            v[state] += alpha * (reward + gamma * v[next_state] - v[state])
            state = next_state


if __name__ == "__main__":
    # init the environment:
    max_episodes = 4000
    env = Gridworldv2(step_cost=-0.1, max_ep_length=30)
    temporal_difference_learning(env, max_episodes)

