#!/usr/bin/env python
# Multi-GPU PPO agent training script; Image observations, discrete actions
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
import os
from datetime import datetime

import gym
import gym.wrappers
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)

import procgen  # Import & register procgen Gym envs

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch9-Distributed-RL-Agent")
parser.add_argument("--env", default="procgen:procgen-coinrun-v0")
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
    def __init__(self, state_dim, action_dim, execution_strategy):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.execution_strategy = execution_strategy
        with self.execution_strategy.scope():
            self.weight_initializer = tf.keras.initializers.he_normal()
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
        output_discrete_action = Dense(
            self.action_dim,
            activation="softmax",
            kernel_initializer=self.weight_initializer,
        )(dropout2)
        return tf.keras.models.Model(
            inputs=obs_input, outputs=output_discrete_action, name="Actor"
        )

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        logits = self.model.predict(state_np)  # shape: (batch_size, self.action_dim)
        action = np.random.choice(self.action_dim, p=logits[0])
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        # action = (action,)
        return logits, action

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        log_old_policy = tf.math.log(tf.reduce_sum(old_policy * actions))
        log_old_policy = tf.stop_gradient(log_old_policy)
        log_new_policy = tf.math.log(tf.reduce_sum(new_policy * actions))
        # Avoid INF in exp by setting 80 as the upper bound since,
        # tf.exp(x) for x>88 yeilds NaN (float32)
        ratio = tf.exp(
            tf.minimum(log_new_policy - tf.stop_gradient(log_old_policy), 80)
        )
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
        )
        gaes = tf.stop_gradient(gaes)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)  # One-hot encoding
        actions = tf.reshape(actions, [-1, self.action_dim])  # Add batch dimension
        actions = tf.cast(actions, tf.float64)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(old_policy, logits, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def train_distributed(self, old_policy, states, actions, gaes):
        per_replica_losses = self.execution_strategy.run(
            self.train, args=(old_policy, states, actions, gaes)
        )
        return self.execution_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )


class Critic:
    def __init__(self, state_dim, execution_strategy):
        self.state_dim = state_dim
        self.execution_strategy = execution_strategy
        with self.execution_strategy.scope():
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
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def train_distributed(self, states, td_targets):
        per_replica_losses = self.execution_strategy.run(
            self.train, args=(states, td_targets)
        )
        return self.execution_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )


class PPOAgent:
    def __init__(self, env):
        """Distributed PPO Agent for image observations and discrete action-space Gym envs

        Args:
            env (gym.Env): OpenAI Gym I/O compatible RL environment with discrete action space
        """
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        # Create a Distributed execution strategy
        self.distributed_execution_strategy = tf.distribute.MirroredStrategy()
        print(
            f"Number of devices: {self.distributed_execution_strategy.num_replicas_in_sync}"
        )
        # Create Actor & Critic networks under the distributed execution strategy scope
        with self.distributed_execution_strategy.scope():
            self.actor = Actor(
                self.state_dim, self.action_dim, tf.distribute.get_strategy()
            )
            self.critic = Critic(self.state_dim, tf.distribute.get_strategy())

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

    def train(self, max_episodes=1000):

        with self.distributed_execution_strategy.scope():
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
                        self.env.render()
                        logits, action = self.actor.get_action(state)

                        next_state, reward, dones, _ = self.env.step(action)
                        step_num += 1

                        print(
                            f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action} dones:{dones}",
                            end="\r",
                        )
                        done = np.all(dones)
                        if done:
                            next_state = prev_state
                        else:
                            prev_state = next_state

                        state_batch.append(state)
                        action_batch.append(action)
                        reward_batch.append((reward + 8) / 8)
                        old_policy_batch.append(logits)

                        if len(state_batch) >= args.update_freq or done:
                            states = np.array(
                                [state.squeeze() for state in state_batch]
                            )
                            actions = np.array(action_batch)
                            rewards = np.array(reward_batch)
                            old_policies = np.array(
                                [old_pi.squeeze() for old_pi in old_policy_batch]
                            )

                            v_values = self.critic.model.predict(states)
                            next_v_value = self.critic.model.predict(
                                np.expand_dims(next_state, 0)
                            )

                            gaes, td_targets = self.gae_target(
                                rewards, v_values, next_v_value, done
                            )
                            actor_losses, critic_losses = [], []
                            for epoch in range(args.epochs):
                                actor_loss = self.actor.train_distributed(
                                    old_policies, states, actions, gaes
                                )
                                actor_losses.append(actor_loss)
                                critic_loss = self.critic.train_distributed(
                                    states, td_targets
                                )
                                critic_losses.append(critic_loss)
                            # Plot mean actor & critic losses on every update
                            tf.summary.scalar(
                                "actor_loss", np.mean(actor_losses), step=ep
                            )
                            tf.summary.scalar(
                                "critic_loss", np.mean(critic_losses), step=ep
                            )

                            state_batch = []
                            action_batch = []
                            reward_batch = []
                            old_policy_batch = []

                        episode_reward += reward
                        state = next_state

                    print(f"\n Episode#{ep} Reward:{episode_reward}")
                    tf.summary.scalar("episode_reward", episode_reward, step=ep)


if __name__ == "__main__":
    env_name = args.env
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.Monitor(env=env, directory="./videos", force=True)
    agent = PPOAgent(env)
    agent.train()
