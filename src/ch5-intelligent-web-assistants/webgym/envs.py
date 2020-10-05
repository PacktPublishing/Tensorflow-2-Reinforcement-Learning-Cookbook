import gym.spaces
import numpy as np

from miniwob_env import MiniWoBVisualEnv
from miniwob.action import MiniWoBCoordClick


class MiniWoBClickButtonVisualEnv(MiniWoBVisualEnv):
    def __init__(self, num_instances=1):
        self.miniwob_env_name = "click-button"
        super().__init__(self.miniwob_env_name, num_instances)
        self.task_width = 320
        self.task_height = 320
        self.num_channels = 3  # RGB
        self.observation_space = gym.spaces.Box(
            0, 255, (self.task_width, self.task_height, self.num_channels), dtype=int
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.task_width, self.task_height]),
            shape=(2,),
            dtype=int,
        )

    def step(self, actions):
        """Applies an action on each instance and returns the results.

        Args:
            actions (list[(x, y) or None]);
              - x is the number of pixels from the left of browser window
              - y is the number of pixels from the top of browser window

        Returns:
            tuple (states, rewards, dones, info)
                states (list[PIL.Image.Image])
                rewards (list[float])
                dones (list[bool])
                info (dict): additional debug information.
                    Global debug information is directly in the root level
                    Local information for instance i is in info['n'][i]
        """
        assert (
            len(actions) == self.num_instances
        ), f"Expected len(actions)={self.num_instances}. Got {len(actions)}."

        def clamp(action, low=self.action_space.low, high=self.action_space.high):
            low_x, low_y = low
            high_x, high_y = high
            return (
                max(low_x, min(action[0], high_x)),
                max(low_y, min(action[1], high_y)),
            )

        miniwob_actions = [
            MiniWoBCoordClick(*clamp(action)) if action is not None else None
            for action in actions
        ]
        return super().step(miniwob_actions)
