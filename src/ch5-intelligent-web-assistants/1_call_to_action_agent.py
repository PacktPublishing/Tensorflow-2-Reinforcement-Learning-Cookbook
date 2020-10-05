import argparse
import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Input,
    Lambda,
    Conv2D,
    Activation,
    MaxPool2D,
    Dropout,
    Flatten,
)


tf.keras.backend.set_floatx("float64")


parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch5-Click-To-Action-Agent")
parser.add_argument("--env", default="MiniWoBClickButtonVisualEnv-v0")
parser.add_argument("--update-freq", type=int, default=5)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--actor-lr", type=float, default=0.0005)
parser.add_argument("--critic-lr", type=float, default=0.001)
parser.add_argument("--clip-ratio", type=float, default=0.1)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

