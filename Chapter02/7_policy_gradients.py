#!/usr/bin/env python
# Policy gradient algorithm and agent with neural network policy
# Chapter 2, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym


class PolicyNet(keras.Model):
    def __init__(self, action_dim=1):
        super(PolicyNet, self).__init__()
        self.fc1 = layers.Dense(24, activation="relu")
        self.fc2 = layers.Dense(36, activation="relu")
        self.fc3 = layers.Dense(action_dim, activation="softmax")

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def process(self, observations):
        # Process batch observations using `call(x)` behind-the-scenes
        action_probabilities = self.predict_on_batch(observations)
        return action_probabilities


class Agent(object):
    def __init__(self, action_dim=1):
        """Agent with a neural-network brain powered policy

        Args:
            action_dim (int): Action dimension
        """
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.gamma = 0.99

    def policy(self, observation):
        observation = observation.reshape(1, -1)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_logits = self.policy_net(observation)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        return action

    def get_action(self, observation):
        action = self.policy(observation).numpy()
        return action.squeeze()

    def learn(self, states, rewards, actions):
        discounted_reward = 0
        discounted_rewards = []
        rewards.reverse()
        for r in rewards:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
            discounted_rewards.reverse()

        for state, reward, action in zip(states, discounted_rewards, actions):
            with tf.GradientTape() as tape:
                action_probabilities = self.policy_net(np.array([state]), training=True)
                loss = self.loss(action_probabilities, action, reward)
            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.policy_net.trainable_variables)
            )

    def loss(self, action_probabilities, action, reward):
        dist = tfp.distributions.Categorical(
            probs=action_probabilities, dtype=tf.float32
        )
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss


def train(agent: Agent, env: gym.Env, episodes: int, render=True):
    """Train `agent` in `env` for `episodes`

    Args:
        agent (Agent): Agent to train
        env (gym.Env): Environment to train the agent
        episodes (int): Number of episodes to train
        render (bool): True=Enable/False=Disable rendering; Default=True
    """

    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0
        rewards = []
        states = []
        actions = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward
            if render:
                env.render()
            if done:
                agent.learn(states, rewards, actions)
                print("\n")
            print(f"Episode#:{episode} ep_reward:{total_reward}", end="\r")


if __name__ == "__main__":
    agent = Agent()
    episodes = 500
    env = gym.make("MountainCar-v0")
    train(agent, env, episodes)
    env.close()
