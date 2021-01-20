#!/usr/bin/env python
# Q-learning algorithm and agent
# Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import numpy as np
import random
from value_function_utils import visualize_grid_action_values
from envs.gridworldv2 import GridworldV2Env


def q_learning(env, max_episodes):
    grid_action_values = np.zeros((len(env.distinct_states), env.action_space.n))
    grid_action_values[env.goal_state] = 1
    grid_action_values[env.bomb_state] = -1
    gamma = 0.99  # discounting factor
    alpha = 0.01  # learning rate
    # q: state-action-value function
    q = grid_action_values
    for episode in range(max_episodes):
        step_num = 1
        done = False
        state = env.reset()
        while not done:
            decayed_epsilon = 1 * gamma ** step_num  # Doesn't have to be gamma
            action = greedy_policy(q[state], decayed_epsilon)
            next_state, reward, done = env.step(action)

            # Q-Learning update
            grid_action_values[state][action] += alpha * (
                reward + gamma * max(q[next_state]) - q[state][action]
            )

            step_num += 1
            state = next_state
    visualize_grid_action_values(grid_action_values)


def greedy_policy(q_values, epsilon):
    """Epsilon-greedy policy """

    if random.random() >= epsilon:
        return np.argmax(q_values)
    else:
        return random.randint(0, 3)


if __name__ == "__main__":
    max_episodes = 4000
    env = GridworldV2Env(step_cost=-0.1, max_ep_length=30)
    q_learning(env, max_episodes)
