#!/usr/bin/env python
# Training Deep RL agents using remote simulators
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import datetime
import os
import sys
import logging

import gym.spaces
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tradegym_http_client import Client
from sac_agent_base import SAC

# Create an App-level child logger
logger = logging.getLogger("TFRL-cookbook-ch7-training-with-sim-server")
# Set handler for this logger to handle messages
logger.addHandler(logging.StreamHandler())
# Set logging-level for this logger's handler
logger.setLevel(logging.DEBUG)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join("logs", "TFRL-Cookbook-Ch4-SAC", current_time)
summary_writer = tf.summary.create_file_writer(train_log_dir)


if __name__ == "__main__":

    # Set up client to connect to sim server
    sim_service_address = "http://127.0.0.1:6666"
    client = Client(sim_service_address)

    # Set up training environment
    env_id = "StockTradingContinuousEnv-v0"
    instance_id = client.env_create(env_id)

    # Set up agent
    observation_space_info = client.env_observation_space_info(instance_id)
    observation_shape = observation_space_info.get("shape")
    action_space_info = client.env_action_space_info(instance_id)
    action_space = gym.spaces.Box(
        np.array(action_space_info.get("low")),
        np.array(action_space_info.get("high")),
        action_space_info.get("shape"),
    )
    agent = SAC(observation_shape, action_space)

    # Configure training
    max_epochs = 30000
    random_epochs = 0.6 * max_epochs
    max_steps = 100
    save_freq = 500
    reward = 0
    done = False

    done, use_random, episode, steps, epoch, episode_reward = (
        False,
        True,
        0,
        0,
        0,
        0,
    )
    cur_state = client.env_reset(instance_id)

    # Start training
    while epoch < max_epochs:
        if steps > max_steps:
            done = True

        if done:
            episode += 1
            logger.info(
                f"episode:{episode} cumulative_reward:{episode_reward} steps:{steps} epochs:{epoch}"
            )
            with summary_writer.as_default():
                tf.summary.scalar("Main/episode_reward", episode_reward, step=episode)
                tf.summary.scalar("Main/episode_steps", steps, step=episode)
            summary_writer.flush()

            done, cur_state, steps, episode_reward = (
                False,
                client.env_reset(instance_id),
                0,
                0,
            )
            if episode % save_freq == 0:
                agent.save_model(
                    f"sac_actor_episode{episode}_{env_id}.h5",
                    f"sac_critic_episode{episode}_{env_id}.h5",
                )

        if epoch > random_epochs:
            use_random = False

        action = agent.act(np.array(cur_state), use_random=use_random)
        next_state, reward, done, _ = client.env_step(
            instance_id, action.numpy().tolist()
        )
        agent.train(np.array(cur_state), action, reward, np.array(next_state), done)

        cur_state = next_state
        episode_reward += reward
        steps += 1
        epoch += 1

        # Update Tensorboard with Agent's training status
        agent.log_status(summary_writer, epoch, reward)
        summary_writer.flush()

    agent.save_model(
        f"sac_actor_final_episode_{env_id}.h5", f"sac_critic_final_episode_{env_id}.h5"
    )
