#!/usr/bin/env python
# WebGym's base environment definition for RL agent interacting with web pages
# Chapter 6, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import os

import gym
from PIL import Image

from miniwob.action import MiniWoBCoordClick
from miniwob.environment import MiniWoBEnvironment

cur_path_dir = os.path.dirname(os.path.realpath(__file__))
miniwob_dir = os.path.join(cur_path_dir, "miniwob", "html", "miniwob")


class MiniWoBEnv(MiniWoBEnvironment, gym.Env):
    def __init__(
        self,
        env_name: str,
        obs_im_shape,
        num_instances: int = 1,
        miniwob_dir: str = miniwob_dir,
        seeds: list = [1],
    ):
        super().__init__(env_name)
        self.base_url = f"file://{miniwob_dir}"
        self.configure(num_instances=num_instances, seeds=seeds, base_url=self.base_url)
        # self.set_record_screenshots(True)
        self.obs_im_shape = obs_im_shape

    def reset(self, seeds=[1], mode=None, record_screenshots=False):
        """Forces stop and start all instances.

        Args:
            seeds (list[object]): Random seeds to set for each instance;
                If specified, len(seeds) must be equal to the number of instances.
                A None entry in the list = do not set a new seed.
            mode (str): If specified, set the data mode to this value before
                starting new episodes.
            record_screenshots (bool): Whether to record screenshots of the states.
        Returns:
            states (list[MiniWoBState])
        """
        miniwob_state = super().reset(seeds, mode, record_screenshots=True)

        return [
            state.screenshot.resize(self.obs_im_shape, Image.ANTIALIAS)
            for state in miniwob_state
        ]

    def step(self, actions):
        """Applies an action on each instance and returns the results.

        Args:
            actions (list[MiniWoBAction or None])

        Returns:
            tuple (states, rewards, dones, info)
                states (list[PIL.Image.Image])
                rewards (list[float])
                dones (list[bool])
                info (dict): additional debug information.
                    Global debug information is directly in the root level
                    Local information for instance i is in info['n'][i]
        """
        states, rewards, dones, info = super().step(actions)
        # Obtain screenshot & Resize image obs to match config
        img_states = [
            state.screenshot.resize(self.obs_im_shape) if not dones[i] else None
            for i, state in enumerate(states)
        ]
        return img_states, rewards, dones, info


if __name__ == "__main__":
    env = MiniWoBEnv("click-pie", obs_im_shape=(320, 240))
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = [MiniWoBCoordClick(90, 150)]
            obs, reward, done, info = env.step(action)
            [ob.show() for ob in obs if ob is not None]
    env.close()
