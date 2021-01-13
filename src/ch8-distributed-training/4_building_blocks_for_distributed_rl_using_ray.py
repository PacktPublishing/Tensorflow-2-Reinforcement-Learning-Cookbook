#!/usr/bin/env python
# Building-blocks for distributed deep RL agent training using Ray
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import pickle
import sys

import fire
import gym
import numpy as np
import ray

if "." not in sys.path:
    sys.path.insert(0, ".")
from sac_agent_base import SAC


@ray.remote
class ParameterServer(object):
    def __init__(self, weights):
        values = [value.copy() for value in weights]
        self.weights = values

    def push(self, weights):
        values = [value.copy() for value in weights]
        self.weights = values

    def pull(self):
        return self.weights

    def get_weights(self):
        return self.weights

    # save weights to disk
    def save_weights(self, name):
        with open(name + "weights.pkl", "wb") as pkl:
            pickle.dump(self.weights, pkl)
        print(f"Weights saved to {name + 'weights.pkl'}.")


@ray.remote
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for RL Agents
    """

    def __init__(self, obs_shape, action_shape, size):
        self.cur_states = np.zeros([size, obs_shape[0]], dtype=np.float32)
        self.actions = np.zeros([size, action_shape[0]], dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_states = np.zeros([size, obs_shape[0]], dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.idx, self.size, self.max_size = 0, 0, size
        self.rollout_steps = 0

    def store(self, obs, act, rew, next_obs, done):
        self.cur_states[self.idx] = np.squeeze(obs)
        self.actions[self.idx] = np.squeeze(act)
        self.rewards[self.idx] = np.squeeze(rew)
        self.next_states[self.idx] = np.squeeze(next_obs)
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.rollout_steps += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            cur_states=self.cur_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )

    def get_counts(self):
        return self.rollout_steps


@ray.remote
def rollout(ps, replay_buffer, config):
    """Collect experience using an exploration policy"""
    env = gym.make(config["env"])
    obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = config["steps_per_epoch"] * config["epochs"]

    agent = SAC(env.observation_space.shape, env.action_space)
    weights = ray.get(ps.pull.remote())
    target_weights = agent.actor.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i]
    agent.actor.set_weights(target_weights)

    for step in range(total_steps):
        if step > config["random_exploration_steps"]:
            # Use Agent's policy for exploration after `random_exploration_steps`
            a = agent.act(obs)
        else:  # Use a uniform random exploration policy
            a = env.action_space.sample()

        next_obs, reward, done, _ = env.step(a)
        print(f"Step#:{step} reward:{reward} done:{done}")
        ep_ret += reward
        ep_len += 1

        done = False if ep_len == config["max_ep_len"] else done

        # Store experience to replay buffer
        replay_buffer.store.remote(obs, a, reward, next_obs, done)

        obs = next_obs

        if done or (ep_len == config["max_ep_len"]):
            """
            Perform parameter sync at the end of the trajectory.
            """
            obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            weights = ray.get(ps.pull.remote())
            agent.actor.set_weights(weights)


@ray.remote(num_gpus=1, max_calls=1)
def train(ps, replay_buffer, config):
    agent = SAC(config["obs_shape"], config["action_space"])

    weights = ray.get(ps.pull.remote())
    agent.actor.set_weights(weights)

    train_step = 1
    while True:

        agent.train_with_distributed_replay_memory(
            ray.get(replay_buffer.sample_batch.remote())
        )

        if train_step % config["worker_update_freq"] == 0:
            weights = agent.actor.get_weights()
            ps.push.remote(weights)
        train_step += 1


def main(
    env="MountainCarContinuous-v0",
    epochs=1000,
    steps_per_epoch=5000,
    replay_size=100000,
    random_exploration_steps=1000,
    max_ep_len=1000,
    num_workers=4,
    num_learners=1,
    worker_update_freq=500,
):
    config = {
        "env": env,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "max_ep_len": max_ep_len,
        "replay_size": replay_size,
        "random_exploration_steps": random_exploration_steps,
        "num_workers": num_workers,
        "num_learners": num_learners,
        "worker_update_freq": worker_update_freq,
    }

    env = gym.make(config["env"])
    config["obs_shape"] = env.observation_space.shape
    config["action_space"] = env.action_space

    ray.init()

    agent = SAC(config["obs_shape"], config["action_space"])
    params_server = ParameterServer.remote(agent.actor.get_weights())

    replay_buffer = ReplayBuffer.remote(
        config["obs_shape"], config["action_space"].shape, config["replay_size"]
    )

    task_rollout = [
        rollout.remote(params_server, replay_buffer, config)
        for i in range(config["num_workers"])
    ]

    task_train = [
        train.remote(params_server, replay_buffer, config)
        for i in range(config["num_learners"])
    ]

    ray.wait(task_rollout)
    ray.wait(task_train)


if __name__ == "__main__":
    fire.Fire(main)
