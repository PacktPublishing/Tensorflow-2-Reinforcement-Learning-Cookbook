#!/usr/bin/env python
# Evaluating deep RL agents
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import os
import sys

from argparse import ArgumentParser
import imageio
import gym

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tradegym  # Register tradegym envs with OpenAI Gym registry
from sac_agent_runtime import SAC

parser = ArgumentParser(prog="TFRL-Cookbook-Ch7-Evaluating-RL-Agents")
parser.add_argument("--agent", default="SAC", help="Name of Agent. Default=SAC")
parser.add_argument(
    "--env",
    default="StockTradingContinuousEnv-v0",
    help="Name of Gym env. Default=StockTradingContinuousEnv-v0",
)
parser.add_argument(
    "--num-episodes",
    default=10,
    help="Number of episodes to evaluate the agent. Default=100",
)
parser.add_argument(
    "--trained-models-dir",
    default="trained_models",
    help="Directory contained trained models. Default=trained_models",
)
parser.add_argument(
    "--model-version",
    default="episode100",
    help="Trained model version. Default=episode100",
)
parser.add_argument(
    "--render",
    type=bool,
    help="Render environment and write to file? (True/False). Default=False",
)
args = parser.parse_args()


if __name__ == "__main__":
    # Create an instance of the evaluation environment
    env = gym.make(args.env)
    if args.agent != "SAC":
        print(f"Unsupported Agent: {args.agent}. Using SAC Agent")
        args.agent = "SAC"
    # Create an instance of the Soft Actor-Critic Agent
    agent = SAC(env.observation_space.shape, env.action_space)
    # Load trained Agent model/brain
    model_version = args.model_version
    agent.load_actor(
        os.path.join(args.trained_models_dir, f"sac_actor_{model_version}.h5")
    )
    agent.load_critic(
        os.path.join(args.trained_models_dir, f"sac_critic_{model_version}.h5")
    )
    print(f"Loaded {args.agent} agent with trained model version:{model_version}")
    render = args.render
    # Evaluate/Test/Rollout Agent with trained model/brain
    video = imageio.get_writer("agent_eval_video.mp4", fps=30)
    avg_reward = 0
    for i in range(args.num_episodes):
        cur_state, done, rewards = env.reset(), False, 0
        while not done:
            action = agent.act(cur_state, test=True)
            next_state, reward, done, _ = env.step(action[0])
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(env.render(mode="rgb_array"))
        print(f"Episode#:{i} cumulative_reward:{rewards}")
        avg_reward += rewards
    avg_reward /= args.num_episodes
    video.close()
    print(f"Average rewards over {args.num_episodes} episodes: {avg_reward}")