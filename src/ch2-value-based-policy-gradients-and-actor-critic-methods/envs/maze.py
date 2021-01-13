#!/usr/bin/env python
# Maze RL environment with image observations
# Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

from typing import List

import gym
import numpy as np


class MazeEnv(gym.Env):
    def __init__(self, stochastic=True):
        """Stochastic Maze environment with coins, obstacles/walls and a goal state.
        Actions: 0: Move Up, 1: Move Down, 2: Move Left, 3: Move Right
        Reward is based on the number of coins collected by the agent before reaching goal state.
        Stochasticity in the env arises from the `slip_probability` which alters the action.
          The slip action will be the clockwise directinal action (LEFT -> UP, UP -> RIGHT etc)
          Example with `slip_probability=0.2`: With 0.2 probabilty a "RIGHT" action may result in "DOWN"
        """
        self.map = np.asarray(["SWFWG", "OOOOO", "WOOOW", "FOWFW"])
        self.observation_space = gym.spaces.Discrete(1)
        self.dim = (4, 5)  # Used for plotting policy & value function
        self.img_map = np.ones(self.dim)
        self.obstacles = [(0, 1), (0, 3), (2, 0), (2, 4), (3, 2), (3, 4)]
        for x in self.obstacles:
            self.img_map[x[0]][x[1]] = 0
        if stochastic:
            self.slip = True
        self.distinct_states = 112  # Number of unique states in the env
        self.action_space = gym.spaces.Discrete(4)
        # Clock-wise action slip for stochasticity
        self.slip_action_map = {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
        }
        self.slip_probability = 0.1
        self.start_pos = (0, 0)
        self.goal_pos = (0, 4)
        self.index_to_coordinate_map = {
            0: (0, 0),
            1: (1, 0),
            2: (3, 0),
            3: (1, 1),
            4: (2, 1),
            5: (3, 1),
            6: (0, 2),
            7: (1, 2),
            8: (2, 2),
            9: (1, 3),
            10: (2, 3),
            11: (3, 3),
            12: (0, 4),
            13: (1, 4),
        }
        self.coordinate_to_index_map = dict(
            (val, key) for key, val in self.index_to_coordinate_map.items()
        )
        # Start state
        self.state = self.coordinate_to_index_map[self.start_pos]

    def set_state(self, state: int) -> None:
        """Set the current state of the environment. Useful for value iteration

        Args:
            state (int): A valid state in the Maze env int: [0, 112]
        """
        self.state = state

    def step(self, action, slip=True):
        """Run one step into the Maze env

        Args:
            state (Any): Current index state of the maze
            action (int): Discrete action for up, down, left, right
            slip (bool, optional): Stochasticity in the env. Defaults to True.

        Raises:
            ValueError: If invalid action is provided as input

        Returns:
            Tuple : Next state, reward, done, _
        """
        self.slip = slip
        if self.slip:
            if np.random.rand() < self.slip_probability:
                action = self.slip_action_map[action]

        cell = self.index_to_coordinate_map[int(self.state / 8)]
        if action == 0:
            c_next = cell[1]
            r_next = max(0, cell[0] - 1)
        elif action == 1:
            c_next = cell[1]
            r_next = min(self.dim[0] - 1, cell[0] + 1)
        elif action == 2:
            c_next = max(0, cell[1] - 1)
            r_next = cell[0]
        elif action == 3:
            c_next = min(self.dim[1] - 1, cell[1] + 1)
            r_next = cell[0]
        else:
            raise ValueError(f"Invalid action:{action}")

        if (r_next == self.goal_pos[0]) and (
            c_next == self.goal_pos[1]
        ):  # Check if goal reached
            v_coin = self.num2coin(self.state % 8)
            self.state = (
                8 * self.coordinate_to_index_map[(r_next, c_next)] + self.state % 8
            )
            return (
                self.state,
                float(sum(v_coin)),
                True,
            )
        else:
            if (r_next, c_next) in self.obstacles:  # obstacle tuple list
                return self.state, 0.0, False
            else:  # Coin locations
                v_coin = self.num2coin(self.state % 8)
                if (r_next, c_next) == (0, 2):
                    v_coin[0] = 1
                elif (r_next, c_next) == (3, 0):
                    v_coin[1] = 1
                elif (r_next, c_next) == (3, 3):
                    v_coin[2] = 1
                self.state = 8 * self.coordinate_to_index_map[
                    (r_next, c_next)
                ] + self.coin2num(v_coin)
                return (
                    self.state,
                    0.0,
                    False,
                )

    def num2coin(self, n: int):
        # Each element of the below tuple correspond to a status of each coin. 0 for not collected, 1 for collected.
        coinlist = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
        return list(coinlist[n])

    def coin2num(self, v: List):
        if sum(v) < 2:
            return np.inner(v, [1, 2, 3])
        else:
            return np.inner(v, [1, 2, 3]) + 1

    def reset(self):
        # Return the initial state
        self.state = self.coordinate_to_index_map[self.start_pos]
        return self.state

    def render(self):
        cell = self.index_to_coordinate_map[int(self.state / 8)]
        desc = self.map.tolist()

        desc[cell[0]] = (
            desc[cell[0]][: cell[1]]
            + "\x1b[1;34m"  # Blue font
            + "\x1b[4m"  # Underline
            + "\x1b[1m"  # Bold
            + "\x1b[7m"  # Reversed
            + desc[cell[0]][cell[1]]
            + "\x1b[0m"
            + desc[cell[0]][cell[1] + 1 :]
        )

        print("\n".join("".join(row) for row in desc))


if __name__ == "__main__":
    env = MazeEnv()
    obs = env.reset()
    env.render()
    done = False
    step_num = 1
    action_list = ["UP", "DOWN", "LEFT", "RIGHT"]
    # Run one episode
    while not done:
        # Sample a random action from the action space
        action = env.action_space.sample()
        next_obs, reward, done = env.step(action)
        print(
            f"step#:{step_num} action:{action_list[action]} reward:{reward} done:{done}"
        )
        step_num += 1
        env.render()
    env.close()
