#!/usr/bin/env python
# TradeGym env registration script
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import sys
import os

from gym.envs.registration import register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


_AVAILABLE_ENVS = {
    "CryptoTradingEnv-v0": {
        "entry_point": "tradegym.crypto_trading_env:CryptoTradingEnv",
        "description": "Crypto Trading RL environment",
    },
    "StockTradingContinuousEnv-v0": {
        "entry_point": "tradegym.stock_trading_continuous_env:StockTradingContinuousEnv",
        "description": "Stock Trading  RL environment with continous action space",
    },
}


for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))
