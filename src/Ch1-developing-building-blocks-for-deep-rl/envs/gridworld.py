import copy
import sys

import gym
import numpy as np

# Grid cell state and color mapping
EMPTY = BLACK = 0
WALL = GRAY = 1
AGENT = BLUE = 2
MINE = RED = 3
TARGET = GREEN = 4
SUCCESS = PINK = 5

# RGB color value table
COLOR_MAP = {
    BLACK: [0.0, 0.0, 0.0],
    GRAY: [0.5, 0.5, 0.5],
    BLUE: [0.0, 0.0, 1.0],
    RED: [1.0, 0.0, 0.0],
    GREEN: [0.0, 1.0, 0.0],
    PINK: [1.0, 0.0, 1.0],
}

# Action mapping
NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GridworldEnv(gym.Env):
    def __init__(self):
        # Observations
        self.img_shape = [256, 256, 3]
        self.grid_layout = """
        1 1 1 1 1 1 1 1
        1 2 0 0 0 0 0 1
        1 0 1 1 1 0 0 1
        1 0 1 0 1 0 0 1
        1 0 1 4 1 0 0 1
        1 0 3 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 1 1 1 1 1 1 1
        """
        self.initial_grid_state = np.fromstring(self.grid_layout, dtype=int, sep=" ")
        self.initial_grid_state = self.initial_grid_state.reshape(8, 8)
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=self.grid_state.shape
        )
        self.metadata = {"render.modes": ["human"]}
        # Actions
        self.action_space = gym.spaces.Discrete(5)
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.action_pos_dict = {
            NOOP: [0, 0],
            UP: [-1, 0],
            DOWN: [1, 0],
            LEFT: [0, -1],
            RIGHT: [0, 1],
        }
        (self.agent_start_state, self.agent_target_state,) = self.get_state()

        self.viewer = None

    def step(self, action):
        """ return next observation, reward, done , info"""
        action = int(action)
        info = {"success": True}
        done = False
        reward = 0.0
        next_obs = (
            self.agent_state[0] + self.action_pos_dict[action][0],
            self.agent_state[1] + self.action_pos_dict[action][1],
        )

        if action == NOOP:
            return self.grid_state, reward, False, info
        next_state_valid = (
            next_obs[0] < 0 or next_obs[0] >= self.grid_state.shape[0]
        ) or (next_obs[1] < 0 or next_obs[1] >= self.grid_state.shape[1])
        if next_state_valid:
            info["success"] = False
            return self.grid_state, reward, False, info

        next_state = self.grid_state[next_obs[0], next_obs[1]]

        if next_state == EMPTY:
            self.grid_state[next_obs[0], next_obs[1]] = AGENT
        elif next_state == WALL:
            info["success"] = False
            reward = -0.1
            return self.grid_state, reward, False, info
        elif next_state == TARGET:
            done = True
            reward = 1
        elif next_state == MINE:
            done = True
            reward = -1

        # self._render("human")
        self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(next_obs)

        return self.grid_state, reward, done, info

    def reset(self):
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        (self.agent_state, self.agent_target_state,) = self.get_state()
        return self.grid_state

    def get_state(self):
        start_state = np.where(self.grid_state == AGENT)
        goal_state = np.where(self.grid_state == TARGET)

        start_or_goal_not_found = not (start_state[0] and goal_state[0])
        if start_or_goal_not_found:
            sys.exit(
                "Start and/or Goal state not present in the Gridworld. "
                "Check the Grid layout"
            )
        start_state = (start_state[0][0], start_state[1][0])
        goal_state = (goal_state[0][0], goal_state[1][0])

        return start_state, goal_state

    def gridarray_to_image(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        scale_x = int(observation.shape[0] / self.grid_state.shape[0])
        scale_y = int(observation.shape[1] / self.grid_state.shape[1])
        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                for k in range(3):  # 3-channel RGB image
                    pixel_value = COLOR_MAP[self.grid_state[i, j]][k]
                    observation[
                        i * scale_x : (i + 1) * scale_x,
                        j * scale_y : (j + 1) * scale_y,
                        k,
                    ] = pixel_value
        return (255 * observation).astype(np.uint8)

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self.gridarray_to_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    @staticmethod
    def get_action_meanings():
        return ["NOOP", "DOWN", "UP", "LEFT", "RIGHT"]
