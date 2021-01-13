#!/usr/bin/env python
# WebGym Visual MiniWoB environment registration script
# Chapter 6, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import sys
import os

from gym.envs.registration import register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


_AVAILABLE_ENVS = {
    "MiniWoBClickButtonVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBClickButtonVisualEnv",
        "discription": "Click the button on a web page",
    },
    "MiniWoBEmailInboxImportantVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBEmailInboxImportantVisualEnv",
        "discription": "Mark email as important",
    },
    "MiniWoBBookFlightVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBBookFlightVisualEnv",
        "discription": "Book flight",
    },
    "MiniWoBSocialMediaMuteUserVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBSocialMediaMuteUserVisualEnv",
        "discription": "Mute User on Social Media (Twitter-like) webpages",
    },
    "MiniWoBSocialMediaReplyVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBSocialMediaReplyVisualEnv",
        "discription": "Click Reply to users on Social Media (Twitter-like) webpages",
    },
    "MiniWoBLoginUserVisualEnv-v0": {
        "entry_point": "webgym.envs:MiniWoBLoginUserVisualEnv",
        "discription": "Login user",
    },
}


for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))
