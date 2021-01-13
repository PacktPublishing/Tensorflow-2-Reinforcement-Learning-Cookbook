#!/usr/bin/env python
# Packaging deep RL agents for cloud deployments
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import os
import sys
from argparse import ArgumentParser

import gym.spaces
from flask import Flask, request
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sac_agent_runtime import SAC

parser = ArgumentParser(
    prog="TFRL-Cookbook-Ch7-Packaging-RL-Agents-For-Cloud-Deployments"
)

parser.add_argument("--agent", default="SAC", help="Name of Agent. Default=SAC")
parser.add_argument(
    "--host-ip",
    default="0.0.0.0",
    help="IP Address of the host server where Agent service is run. Default=127.0.0.1",
)
parser.add_argument(
    "--host-port",
    default="5555",
    help="Port on the host server to use for Agent service. Default=5555",
)
parser.add_argument(
    "--trained-models-dir",
    default="trained_models",
    help="Directory contained trained models. Default=trained_models",
)
parser.add_argument(
    "--config",
    default="runtime_config.json",
    help="Runtime config parameters for the Agent. Default=runtime_config.json",
)
parser.add_argument(
    "--observation-shape",
    default=(6, 31),
    help="Shape of observations. Default=(6, 31)",
)
parser.add_argument(
    "--action-space-low", default=[-1], help="Low value of action space. Default=[-1]"
)
parser.add_argument(
    "--action-space-high", default=[1], help="High value of action space. Default=[1]"
)
parser.add_argument(
    "--action-shape", default=(1,), help="Shape of actions. Default=(1,)"
)
parser.add_argument(
    "--model-version",
    default="episode100",
    help="Trained model version. Default=episode100",
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.agent != "SAC":
        print(f"Unsupported Agent: {args.agent}. Using SAC Agent")
        args.agent = "SAC"
    # Set Agent's runtime configs
    observation_shape = args.observation_shape
    action_space = gym.spaces.Box(
        np.array(args.action_space_low),
        np.array(args.action_space_high),
        args.action_shape,
    )

    # Create an instance of the Agent
    agent = SAC(observation_shape, action_space)
    # Load trained Agent model/brain
    model_version = args.model_version
    agent.load_actor(
        os.path.join(args.trained_models_dir, f"sac_actor_{model_version}.h5")
    )
    agent.load_critic(
        os.path.join(args.trained_models_dir, f"sac_critic_{model_version}.h5")
    )
    print(f"Loaded {args.agent} agent with trained model version:{model_version}")

    # Setup Agent (http) service
    app = Flask(__name__)

    @app.route("/v1/act", methods=["POST"])
    def get_action():
        data = request.get_json()
        action = agent.act(np.array(data.get("observation")), test=True)
        return {"action": action.numpy().tolist()}

    # Launch/Run the Agent (http) service
    app.run(host=args.host_ip, port=args.host_port, debug=True)
