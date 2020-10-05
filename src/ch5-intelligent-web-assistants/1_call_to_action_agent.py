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


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)
        flat = Flatten(pool2)
        dense1 = Dense(128, activation="relu")(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation="relu")(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output_val = Dense(self.action_dim, activation="tanh")(dropout2)
        mu_output = Lambda(lambda x: x * self.action_bound)(output_val)
        std_output = Dense(self.action_dim, activation="softplus")(dropout2)
        return tf.keras.models.Model(obs_input, [mu_output, std_output])
