#!/usr/bin/env python
# Simple test script for the deployed Trading Bot-as-a-Service
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy


import os
import sys

import gym
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tradegym  # Register tradegym envs with OpenAI Gym registry

host_ip = "127.0.0.1"
host_port = 5555
endpoint = "v1/act"
env = gym.make("StockTradingContinuousEnv-v0")

post_data = {"observation": env.observation_space.sample().tolist()}
res = requests.post(f"http://{host_ip}:{host_port}/{endpoint}", json=post_data)
if res.ok:
    print(f"Received Agent action:{res.json()}")
