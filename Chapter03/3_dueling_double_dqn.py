#!/usr/bin/env python
# Dueling, Double DQN agent training script
# Chapter 3, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
from datetime import datetime
import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Input
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch3-DuelingDoubleDQN")
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.995)
parser.add_argument("--eps_min", type=float, default=0.01)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class DuelingDQN:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.model = self.nn_model()

    def nn_model(self):
        state_input = Input((self.state_dim,))
        fc1 = Dense(32, activation="relu")(state_input)
        fc2 = Dense(16, activation="relu")(fc1)
        value_output = Dense(1)(fc2)
        advantage_output = Dense(self.action_dim)(fc2)
        output = Add()([value_output, advantage_output])
        model = tf.keras.Model(state_input, output)
        model.compile(loss="mse", optimizer=Adam(args.lr))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = DuelingDQN(self.state_dim, self.action_dim)
        self.target_model = DuelingDQN(self.state_dim, self.action_dim)
        self.update_target()

        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[
                range(args.batch_size),
                np.argmax(self.model.predict(next_states), axis=1),
            ]
            targets[range(args.batch_size), actions] = (
                rewards + (1 - done) * next_q_values * args.gamma
            )
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                observation = self.env.reset()
                while not done:
                    action = self.model.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(action)
                    self.buffer.store(
                        observation, action, reward * 0.01, next_observation, done
                    )
                    episode_reward += reward
                    observation = next_observation

                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()
                self.update_target()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent(env)
    agent.train(max_episodes=2000)
