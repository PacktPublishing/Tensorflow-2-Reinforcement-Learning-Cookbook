#!/usr/bin/env python
# Train PPO agents to book flight tickets on the web
# Chapter 6, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPool2D,
)

import webgym  # Used to register webgym environments

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch6-Flight-Booking-Agent")
parser.add_argument("--env", default="MiniWoBBookFlightVisualEnv-v0")
parser.add_argument("--update-freq", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--actor-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=1e-4)
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
        self.action_bound = np.array(action_bound)
        self.std_bound = std_bound
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.eps = 1e-5
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Actor model
        self.opt = tf.keras.optimizers.Nadam(args.actor_lr)

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
        pool1 = MaxPool2D(pool_size=(3, 3), strides=1)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=1)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        # action_dim[0] = 2
        output_val = Dense(
            self.action_dim[0],
            activation="relu",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        # Scale & clip x[i] to be in range [0, action_bound[i]]
        mu_output = Lambda(
            lambda x: tf.clip_by_value(x * self.action_bound, 1e-9, self.action_bound)
        )(output_val)
        std_output_1 = Dense(
            self.action_dim[0],
            activation="softplus",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        std_output = Lambda(
            lambda x: tf.clip_by_value(
                x * self.action_bound, 1e-9, self.action_bound / 2
            )
        )(std_output_1)
        return tf.keras.models.Model(
            inputs=obs_input, outputs=[mu_output, std_output], name="Actor"
        )

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        mu, std = self.model.predict(state_np)
        action = np.random.normal(mu, std + self.eps, size=self.action_dim).astype(
            "int"
        )
        # Clip action to be between 0 and max obs screen size
        action = np.clip(action, 0, self.action_bound)
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        action = (action,)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        # Avoid INF in exp by setting 80 as the upper bound since,
        # tf.exp(x) for x>88 yeilds NaN (float32)
        ratio = tf.exp(
            tf.minimum(log_new_policy - tf.stop_gradient(log_old_policy), 80)
        )
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
        )
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Critic model
        self.opt = tf.keras.optimizers.Nadam(args.critic_lr)

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
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        value = Dense(
            1, activation="linear", kernel_initializer=self.weight_initializer
        )(dropout2)

        return tf.keras.models.Model(inputs=obs_input, outputs=value, name="Critic")

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape
        # Set action_bounds to be within the actual task-window/browser-view of the agent
        self.action_bound = [
            self.env.task_width,
            self.env.task_height,
            self.env.num_allowed_chars,
        ]
        self.std_bound = [1e-2, 1.0]

        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def get_typed_char(self, action_2):
        """Get character value represented by action index

        Args:
            action (int): Character index represented by action[2]
        """
        return self.env.key_action_map.get(int(action_2), "")

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                state_batch = []
                action_batch = []
                reward_batch = []
                old_policy_batch = []

                episode_reward, done = 0, False

                state = self.env.reset()
                prev_state = state
                step_num = 0

                while not done:
                    # self.env.render()
                    log_old_policy, action = self.actor.get_action(state)

                    next_state, reward, dones, _ = self.env.step(action)
                    step_num += 1

                    # Convert action[2] from int idx to char for verbose printing
                    action_print = []
                    for a in action:  # Map apply
                        action_verbose = (a[:2], self.get_typed_char(a[2]))
                        action_print.append(action_verbose)
                    print(
                        f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action_print} dones:{dones}"
                    )
                    done = np.all(dones)
                    if done:
                        next_state = prev_state
                    else:
                        prev_state = next_state
                    state = np.array([np.array(s) for s in state])
                    next_state = np.array([np.array(s) for s in next_state])
                    reward = np.reshape(reward, [1, 1])
                    log_old_policy = np.reshape(log_old_policy, [1, 1])

                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append((reward + 8) / 8)
                    old_policy_batch.append(log_old_policy)

                    if len(state_batch) >= args.update_freq or done:
                        states = np.array([state.squeeze() for state in state_batch])
                        # Convert ([x, y],) to [x, y]
                        actions = np.array([action[0] for action in action_batch])
                        rewards = np.array(
                            [reward.squeeze() for reward in reward_batch]
                        )
                        old_policies = np.array(
                            [old_pi.squeeze() for old_pi in old_policy_batch]
                        )

                        v_values = self.critic.model.predict(states)
                        next_v_value = self.critic.model.predict(next_state)

                        gaes, td_targets = self.gae_target(
                            rewards, v_values, next_v_value, done
                        )
                        actor_losses, critic_losses = [], []
                        for epoch in range(args.epochs):
                            actor_loss = self.actor.train(
                                old_policies, states, actions, gaes
                            )
                            actor_losses.append(actor_loss)
                            critic_loss = self.critic.train(states, td_targets)
                            critic_losses.append(critic_loss)
                        # Plot mean actor & critic losses on every update
                        tf.summary.scalar("actor_loss", np.mean(actor_losses), step=ep)
                        tf.summary.scalar(
                            "critic_loss", np.mean(critic_losses), step=ep
                        )

                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        old_policy_batch = []

                    episode_reward += reward[0][0]
                    state = next_state[0]

                print(f"Episode#{ep} Reward:{episode_reward} Actions:{action_batch}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)


if __name__ == "__main__":
    env_name = "MiniWoBBookFlightVisualEnv-v0"
    env = gym.make(env_name)
    cta_agent = PPOAgent(env)
    cta_agent.train()
