#!/usr/bin/env python
# Asynchronous, Advantage Actor-Critic (A3C) agent training script
# Chapter 3, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count
from threading import Thread

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch3-A3C")
parser.add_argument("--env", default="MountainCarContinuous-v0")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--actor-lr", type=float, default=0.001)
parser.add_argument("--critic-lr", type=float, default=0.002)
parser.add_argument("--update-interval", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

GLOBAL_EPISODE_NUM = 0


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def nn_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(32, activation="relu")(state_input)
        dense_2 = Dense(32, activation="relu")(dense_1)
        out_mu = Dense(self.action_dim, activation="tanh")(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation="softplus")(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.state_dim,)),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert (
            #    v_pred.shape == td_targets.shape
            # ), f"{v_pred.shape} not equal {td_targets.shape}"

            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name, num_workers=cpu_count()):
        env = gym.make(env_name)
        self.env_name = env_name
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.global_critic = Critic(self.state_dim)
        self.num_workers = num_workers

    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            workers.append(
                A3CWorker(env, self.global_actor, self.global_critic, max_episodes)
            )

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class A3CWorker(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def train(self):
        global GLOBAL_EPISODE_NUM
        while self.max_episodes >= GLOBAL_EPISODE_NUM:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = np.array([state.squeeze() for state in state_batch])
                    actions = np.array([action.squeeze() for action in action_batch])
                    rewards = np.array([reward.squeeze() for reward in reward_batch])
                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        (rewards + 8) / 8, next_v_value, done
                    )
                    advantages = td_targets - self.critic.model.predict(states)

                    actor_loss = self.global_actor.train(states, actions, advantages)
                    critic_loss = self.global_critic.train(states, td_targets)

                    self.actor.model.set_weights(self.global_actor.model.get_weights())
                    self.critic.model.set_weights(
                        self.global_critic.model.get_weights()
                    )

                    state_batch = []
                    action_batch = []
                    reward_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print(f"Episode#{GLOBAL_EPISODE_NUM} Reward:{episode_reward}")
            tf.summary.scalar("episode_reward", episode_reward, step=GLOBAL_EPISODE_NUM)
            GLOBAL_EPISODE_NUM += 1

    def run(self):
        self.train()


if __name__ == "__main__":
    env_name = "MountainCarContinuous-v0"
    agent = Agent(env_name, args.num_workers)
    agent.train()
