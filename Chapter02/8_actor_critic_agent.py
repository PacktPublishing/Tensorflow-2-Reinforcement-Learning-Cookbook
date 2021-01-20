#!/usr/bin/env python
# Neural Actor-Critic algorithm and agent
# Chapter N, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp


class ActorCritic(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.critic = tf.keras.layers.Dense(1, activation=None)
        self.actor = tf.keras.layers.Dense(action_dim, activation=None)

    def call(self, input_data):
        x = self.fc1(input_data)
        x1 = self.fc2(x)
        actor = self.actor(x1)
        critic = self.critic(x1)
        return critic, actor


class Agent:
    def __init__(self, action_dim=4, gamma=0.99):
        """Agent with a neural-network brain powered policy

        Args:
            action_dim (int): Action dimension
            gamma (float) : Discount factor. Default=0.99
        """

        self.gamma = gamma
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.actor_critic = ActorCritic(action_dim)

    def get_action(self, state):
        _, action_probabilities = self.actor_critic(np.array([state]))
        action_probabilities = tf.nn.softmax(action_probabilities)
        action_probabilities = action_probabilities.numpy()
        dist = tfp.distributions.Categorical(
            probs=action_probabilities, dtype=tf.float32
        )
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td):
        prob = tf.nn.softmax(prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])

        with tf.GradientTape() as tape:
            value, action_probabilities = self.actor_critic(state, training=True)
            value_next_st, _ = self.actor_critic(next_state, training=True)
            td = reward + self.gamma * value_next_st * (1 - int(done)) - value
            actor_loss = self.actor_loss(action_probabilities, action, td)
            critic_loss = td ** 2
            total_loss = actor_loss + critic_loss
        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
        return total_loss


def train(agent, env, episodes, render=True):
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
        all_loss = []

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            loss = agent.learn(state, action, reward, next_state, done)
            all_loss.append(loss)
            state = next_state
            total_reward += reward
            if render:
                env.render()
            if done:
                print("\n")
            print(f"Episode#:{episode} ep_reward:{total_reward}", end="\r")


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    agent = Agent(env.action_space.n)
    num_episodes = 20000
    train(agent, env, num_episodes)
