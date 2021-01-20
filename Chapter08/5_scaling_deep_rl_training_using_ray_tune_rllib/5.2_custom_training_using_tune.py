#!/usr/bin/env python
# Training Deep RL agents with custom models using Ray Tune
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import sys

import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

if not "." in sys.path:
    sys.path.insert(0, ".")
from custom_model import CustomModel

# Register custom-model in ModelCatalog
ModelCatalog.register_custom_model("CustomCNN", CustomModel)

ray.init()
config = impala.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["model"]["custom_model"] = "CustomCNN"
config["log_level"] = "INFO"
config["framework"] = "tf2"
trainer = impala.ImpalaTrainer(config=config, env="procgen:procgen-coinrun-v0")

for step in range(1000):
    # Custom training loop
    result = trainer.train()
    print(pretty_print(result))

    if step % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

# Restore agent from a checkpoint and start a new training run with a different config
# config["lr"] =  ray.tune.grid_search([0.01, 0.001])"]
# ray.tune.run(trainer, config=config, restore=checkpoint)

ray.shutdown()