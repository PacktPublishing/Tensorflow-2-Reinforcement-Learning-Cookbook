import gym.spaces
import numpy as np

from miniwob_env import MiniWoBVisualEnv
from miniwob.action import MiniWoBCoordClick


class MiniWoBClickButtonVisualEnv(MiniWoBVisualEnv):
    def __init__(self, num_instances=1):
        self.miniwob_env_name = "click-button"
        self.task_width = 160
        self.task_height = 210
        self.obs_im_width = 64
        self.obs_im_height = 64
        self.num_channels = 3  # RGB
        self.obs_im_size = (self.obs_im_width, self.obs_im_height)
        super().__init__(self.miniwob_env_name, self.obs_im_size, num_instances)

        self.observation_space = gym.spaces.Box(
            0,
            255,
            (self.obs_im_width, self.obs_im_height, self.num_channels),
            dtype=int,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.task_width, self.task_height]),
            shape=(2,),
            dtype=int,
        )

    def reset(self, seeds=[1]):
        """Forces stop and start all instances.

        Args:
            seeds (list[object]): Random seeds to set for each instance;
                If specified, len(seeds) must be equal to the number of instances.
                A None entry in the list = do not set a new seed.
        Returns:
            states (list[PIL.Image])
        """
        obs = super().reset(seeds)
        # Click somewhere to Start!
        # miniwob_state, _, _, _ = super().step(
        #    self.num_instances * [MiniWoBCoordClick(10, 10)]
        # )
        return obs

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
