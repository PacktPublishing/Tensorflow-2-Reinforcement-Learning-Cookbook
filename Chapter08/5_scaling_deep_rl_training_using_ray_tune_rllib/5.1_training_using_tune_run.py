#!/usr/bin/env python
# Large-scale training of PPO agent using Ray Tune
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import ray
import sys
from ray import tune
from ray.rllib.models import ModelCatalog

if not "." in sys.path:
    sys.path.insert(0, ".")
from custom_model import CustomModel

# Register custom-model in ModelCatalog
ModelCatalog.register_custom_model("CustomCNN", CustomModel)

ray.init()
experiment_analysis = tune.run(
    "PPO",
    config={
        "env": "procgen:procgen-coinrun-v0",
        "num_gpus": 0,
        "num_workers": 2,
        "model": {"custom_model": "CustomCNN"},
        "framework": "tf2",
        "log_level": "INFO",
    },
    local_dir="ray_results",  # store experiment results in this dir
)
ray.shutdown()