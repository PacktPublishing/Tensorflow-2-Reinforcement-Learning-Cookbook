#!/usr/bin/env python
# Visual WebGym RL envs based on miniwob++ for agents learning to interact with the WWW
# Chapter 6, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import gym.spaces
import numpy as np
import string

from miniwob_env import MiniWoBEnv
from miniwob.action import MiniWoBCoordClick, MiniWoBType


class MiniWoBVisualClickEnv(MiniWoBEnv):
    def __init__(self, name, num_instances=1):
        """RL environment with visual observations and touch/mouse-click action space
            Two dimensional, continuous-valued action space allows agents
            to specify (x, y) coordinates on the visual rendering to click/touch
            to interact with the world-of bits


        Args:
            name (str): Name of the supported MiniWoB-PlusPlus environment
            num_instances (int, optional): Number of parallel env instances. Defaults to 1.
        """
        self.miniwob_env_name = name
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


class MiniWoBVisualClickAndTypeEnv(MiniWoBEnv):
    def __init__(self, name, num_instances=1):
        """RL environment with visual observations and click & type actions
        Three dimensional, continuous-valued action space allows agents
        to specify (x, y) coordinates on the visual rendering to click
        and a character to be typed in a text/typable field.

        Args:
            name (str): Name of the supported MiniWoB-PlusPlus environment
            num_instances (int, optional): Number of parallel env instances. Defaults to 1.
        """
        self.miniwob_env_name = name
        self.task_width = 160
        self.task_height = 210
        self.obs_im_width = 64
        self.obs_im_height = 64
        self.num_channels = 3  # RGB
        self.obs_im_size = (self.obs_im_width, self.obs_im_height)
        super().__init__(self.miniwob_env_name, self.obs_im_size, num_instances)
        # Generate an index mapped character list: ["", a, b, c, ... x, y, z, " "]
        self.key_action_map = {
            i: x for (i, x) in enumerate(list("" + string.ascii_uppercase + " "))
        }
        self.num_allowed_chars = len(self.key_action_map) - 1

        self.observation_space = gym.spaces.Box(
            0,
            255,
            (self.obs_im_width, self.obs_im_height, self.num_channels),
            dtype=int,
        )
        # Action: [x, y, character]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.task_width, self.task_height, self.num_allowed_chars]),
            shape=(3,),
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
        return obs

    def step(self, actions):
        """Applies an action on each instance and returns the results.

        Args:
            actions (list[(x, y, char) or None]);
              - x is the number of pixels from the left of browser window
              - y is the number of pixels from the top of browser window
              - char is the character from `self.key_action_map` to be typed

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
            low_x, low_y, low_char_idx = low
            high_x, high_y, high_char_idx = high
            return (
                max(low_x, min(action[0], high_x)),
                max(low_y, min(action[1], high_y)),
                max(low_char_idx, min(action[2], high_char_idx)),
            )

        clamped_actions = [
            clamp(action) if action is not None else None for action in actions
        ]

        miniwob_click_actions = [
            MiniWoBCoordClick(*clamped_action[:2])
            if clamped_action is not None
            else None
            for clamped_action in clamped_actions
        ]
        # Execute the click action first
        _ = super().step(miniwob_click_actions)
        # Execute the type action next & return
        miniwob_type_actions = [
            MiniWoBType(self.key_action_map.get(clamped_action[2], ""))
            if clamped_action is not None
            else None
            for clamped_action in clamped_actions
        ]
        return super().step(miniwob_type_actions)


class MiniWoBClickButtonVisualEnv(MiniWoBVisualClickEnv):
    def __init__(self, num_instances=1):
        super().__init__("click-button", num_instances)


class MiniWoBEmailInboxImportantVisualEnv(MiniWoBVisualClickEnv):
    """Train RL agents to automate email-management tasks
    E.g.: Find emails from specific person and mark them as important
    """

    def __init__(self, num_instances=1):
        super().__init__("email-inbox-important", num_instances)


class MiniWoBSocialMediaMuteUserVisualEnv(MiniWoBVisualClickEnv):
    def __init__(self, num_instances=1):
        super().__init__("social-media", num_instances)


class MiniWoBSocialMediaReplyVisualEnv(MiniWoBVisualClickEnv):
    def __init__(self, num_instances=1):
        super().__init__("social-media-some", num_instances)


class MiniWoBBookFlightVisualEnv(MiniWoBVisualClickAndTypeEnv):
    def __init__(self, num_instances=1):
        super().__init__("book-flight", num_instances)


class MiniWoBLoginUserVisualEnv(MiniWoBVisualClickAndTypeEnv):
    def __init__(self, num_instances=1):
        super().__init__("login-user", num_instances)