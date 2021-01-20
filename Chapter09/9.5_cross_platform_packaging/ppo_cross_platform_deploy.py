#!/usr/bin/env python
# Train & export PPO Deep RL agent in various model formats for building cross platform apps
# Supported formats include: TF SavedModel, TFLite, TF.js Layers, ONNX
# Chapter 9, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
import os
import sys
from datetime import datetime

import gym
import keras2onnx
import numpy as np
import procgen  # Used to register procgen envs with Gym registry
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D

tf.keras.backend.set_floatx("float32")

parser = argparse.ArgumentParser(
    prog="TFRL-Cookbook-Ch9-PPO-trainer-exporter-xplatform"
)
parser.add_argument(
    "--env",
    default="procgen:procgen-coinrun-v0",
    choices=[
        "procgen:procgen-bigfish-v0",
        "procgen:procgen-bossfight-v0",
        "procgen:procgen-caveflyer-v0",
        "procgen:procgen-chaser-v0",
        "procgen:procgen-climber-v0",
        "procgen:procgen-coinrun-v0",
        "procgen:procgen-dodgeball-v0",
        "procgen:procgen-fruitbot-v0",
        "procgen:procgen-heist-v0",
        "procgen:procgen-jumper-v0",
        "procgen:procgen-leaper-v0",
        "procgen:procgen-maze-v0",
        "procgen:procgen-miner-v0",
        "procgen:procgen-ninja-v0",
        "procgen:procgen-plunder-v0",
        "procgen:procgen-starpilot-v0",
        "Pong-v4",
    ],
)
parser.add_argument("--update-freq", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--actor-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=1e-4)
parser.add_argument("--clip-ratio", type=float, default=0.1)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.eps = 1e-5
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Actor model
        self.opt = tf.keras.optimizers.Nadam(args.actor_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim, name="im_obs")
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
            name="img_obs",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=1)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=1)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        # action_dim[0] = 2
        output_discrete_action = Dense(
            self.action_dim,
            activation="softmax",
            kernel_initializer=self.weight_initializer,
            name="action",
        )(dropout2)
        return tf.keras.models.Model(
            inputs=obs_input, outputs=output_discrete_action, name="Actor"
        )

    def get_action(self, state):
        # Convert [Image] to np.array(np.adarray)
        state_np = np.array([np.array(s) for s in state])
        if len(state_np.shape) == 3:
            # Convert (w, h, c) to (1, w, h, c)
            state_np = np.expand_dims(state_np, 0)
        logits = self.model.predict(state_np)  # shape: (batch_size, self.action_dim)
        action = np.random.choice(self.action_dim, p=logits[0])
        # 1 Action per instance of env; Env expects: (num_instances, actions)
        # action = (action,)
        return logits, action

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        log_old_policy = tf.math.log(tf.reduce_sum(old_policy * actions))
        log_old_policy = tf.stop_gradient(log_old_policy)
        log_new_policy = tf.math.log(tf.reduce_sum(new_policy * actions))
        # Avoid INF in exp by setting 80 as the upper bound since,
        # tf.exp(x) for x>88 yeilds NaN (float32)
        ratio = tf.exp(
            tf.minimum(log_new_policy - tf.stop_gradient(log_old_policy), 80)
        )
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
        )
        gaes = tf.stop_gradient(gaes)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)  # One-hot encoding
        actions = tf.reshape(actions, [-1, self.action_dim])  # Add batch dimension
        actions = tf.cast(actions, tf.float32)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(old_policy, logits, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        actor_model_save_dir = os.path.join(
            model_dir, "actor", str(version), "model.savedmodel"
        )
        self.model.save(actor_model_save_dir, save_format="tf")
        print(f"Actor model saved at:{actor_model_save_dir}")

    def save_tflite(self, model_dir: str, version: int = 1):
        """Save/Export Actor model in TensorFlow Lite format"""
        actor_model_save_dir = os.path.join(model_dir, "actor", str(version))
        model_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # Convert model to TFLite Flatbuffer
        tflite_model = model_converter.convert()
        # Save the model to disk/persistent-storage
        if not os.path.exists(actor_model_save_dir):
            os.makedirs(actor_model_save_dir)
        actor_model_file_name = os.path.join(actor_model_save_dir, "model.tflite")
        with open(actor_model_file_name, "wb") as model_file:
            model_file.write(tflite_model)
        print(f"Actor model saved in TFLite format at:{actor_model_file_name}")

    def save_h5(self, model_dir: str, version: int = 1):
        actor_model_save_path = os.path.join(
            model_dir, "actor", str(version), "model.h5"
        )
        self.model.save(actor_model_save_path, save_format="h5")
        print(f"Actor model saved at:{actor_model_save_path}")

    def save_tfjs(self, model_dir: str, version: int = 1):
        """Save/Export Actor model in TensorFlow.js supported format"""
        actor_model_save_dir = os.path.join(
            model_dir, "actor", str(version), "model.tfjs"
        )
        tfjs.converters.save_keras_model(self.model, actor_model_save_dir)
        print(f"Actor model saved in TF.js format at:{actor_model_save_dir}")

    def save_onnx(self, model_dir: str, version: int = 1):
        """Save/Export Actor model in ONNX format"""
        actor_model_save_path = os.path.join(
            model_dir, "actor", str(version), "model.onnx"
        )
        onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
        keras2onnx.save_model(onnx_model, actor_model_save_path)
        print(f"Actor model saved in ONNX format at:{actor_model_save_path}")


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.model = self.nn_model()
        self.model.summary()  # Print a summary of the Critic model
        self.opt = tf.keras.optimizers.Nadam(args.critic_lr)

    def nn_model(self):
        obs_input = Input(self.state_dim)
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            input_shape=self.state_dim,
            data_format="channels_last",
            activation="relu",
            name="image_obs",
        )(obs_input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=2)(conv1)
        conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=2)(conv2)
        conv3 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool2)
        pool3 = MaxPool2D(pool_size=(3, 3), strides=1)(conv3)
        conv4 = Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )(pool3)
        pool4 = MaxPool2D(pool_size=(3, 3), strides=1)(conv4)
        flat = Flatten()(pool4)
        dense1 = Dense(
            16, activation="relu", kernel_initializer=self.weight_initializer
        )(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(
            8, activation="relu", kernel_initializer=self.weight_initializer
        )(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        value = Dense(
            1,
            activation="linear",
            kernel_initializer=self.weight_initializer,
            name="value",
        )(dropout2)

        return tf.keras.models.Model(inputs=obs_input, outputs=value, name="Critic")

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save(self, model_dir: str, version: int = 1):
        critic_model_save_dir = os.path.join(
            model_dir, "critic", str(version), "model.savedmodel"
        )
        self.model.save(critic_model_save_dir, save_format="tf")
        print(f"Critic model saved at:{critic_model_save_dir}")

    def save_tflite(self, model_dir: str, version: int = 1):
        """Save/Export Critic model in TensorFlow Lite format"""
        critic_model_save_dir = os.path.join(model_dir, "critic", str(version))
        model_converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # Convert model to TFLite Flatbuffer
        tflite_model = model_converter.convert()
        # Save the model to disk/persistent-storage
        if not os.path.exists(critic_model_save_dir):
            os.makedirs(critic_model_save_dir)
        critic_model_file_name = os.path.join(critic_model_save_dir, "model.tflite")
        with open(critic_model_file_name, "wb") as model_file:
            model_file.write(tflite_model)
        print(f"Critic model saved in TFLite format at:{critic_model_file_name}")

    def save_h5(self, model_dir: str, version: int = 1):
        critic_model_save_dir = os.path.join(
            model_dir, "critic", str(version), "model.h5"
        )
        self.model.save(critic_model_save_dir, save_format="h5")
        print(f"Critic model saved at:{critic_model_save_dir}")

    def save_tfjs(self, model_dir: str, version: int = 1):
        """Save/Export Critic model in TensorFlow.js supported format"""
        critic_model_save_dir = os.path.join(
            model_dir, "critic", str(version), "model.tfjs"
        )
        tfjs.converters.save_keras_model(self.model, critic_model_save_dir)
        print(f"Critic model saved TF.js format at:{critic_model_save_dir}")

    def save_onnx(self, model_dir: str, version: int = 1):
        """Save/Export Critic model in ONNX format"""
        critic_model_save_path = os.path.join(
            model_dir, "critic", str(version), "model.onnx"
        )
        onnx_model = keras2onnx.convert_keras(self.model, self.model.name)
        keras2onnx.save_model(onnx_model, critic_model_save_path)
        print(f"Critic model saved in ONNX format at:{critic_model_save_path}")


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                state_batch = []
                action_batch = []
                reward_batch = []
                old_policy_batch = []

                episode_reward, done = 0, False

                state = self.env.reset()
                prev_state = state
                step_num = 0

                while not done:
                    # self.env.render()
                    logits, action = self.actor.get_action(state)

                    next_state, reward, dones, _ = self.env.step(action)
                    step_num += 1
                    print(
                        f"ep#:{ep} step#:{step_num} step_rew:{reward} action:{action} dones:{dones}",
                        end="\r",
                    )
                    done = np.all(dones)
                    if done:
                        next_state = prev_state
                    else:
                        prev_state = next_state

                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append((reward + 8) / 8)
                    old_policy_batch.append(logits)

                    if len(state_batch) >= args.update_freq or done:
                        states = np.array([state.squeeze() for state in state_batch])
                        actions = np.array(action_batch).astype("float32")
                        rewards = np.array(reward_batch).astype("float32")
                        old_policies = np.array(
                            [old_pi.squeeze() for old_pi in old_policy_batch]
                        )
                        v_values = self.critic.model.predict(states)
                        next_v_value = self.critic.model.predict(
                            np.expand_dims(next_state, 0)
                        )

                        gaes, td_targets = self.gae_target(
                            rewards, v_values, next_v_value, done
                        )
                        actor_losses, critic_losses = [], []
                        for epoch in range(args.epochs):
                            actor_loss = self.actor.train(
                                old_policies, states, actions, gaes
                            )
                            actor_losses.append(actor_loss)
                            critic_loss = self.critic.train(states, td_targets)
                            critic_losses.append(critic_loss)
                        # Plot mean actor & critic losses on every update
                        tf.summary.scalar("actor_loss", np.mean(actor_losses), step=ep)
                        tf.summary.scalar(
                            "critic_loss", np.mean(critic_losses), step=ep
                        )

                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        old_policy_batch = []

                    episode_reward += reward
                    state = next_state

                print(f"\n Episode#{ep} Reward:{episode_reward} Actions:{action_batch}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

    def save(self, model_dir: str, version: int = 1):
        self.actor.save(model_dir, version)
        self.critic.save(model_dir, version)

    def save_tflite(self, model_dir: str, version: int = 1):
        # Make sure `toco_from_protos binary` is on system's PATH to avoid TFLite ConverterError
        toco_bin_dir = os.path.dirname(sys.executable)
        if not toco_bin_dir in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + toco_bin_dir
        print(f"Saving Agent model (TFLite) to:{model_dir}\n")
        self.actor.save_tflite(model_dir, version)
        self.critic.save_tflite(model_dir, version)

    def save_h5(self, model_dir: str, version: int = 1):
        print(f"Saving Agent model (HDF5) to:{model_dir}\n")
        self.actor.save_h5(model_dir, version)
        self.critic.save_h5(model_dir, version)

    def save_tfjs(self, model_dir: str, version: int = 1):
        print(f"Saving Agent model (TF.js) to:{model_dir}\n")
        self.actor.save_tfjs(model_dir, version)
        self.critic.save_tfjs(model_dir, version)

    def save_onnx(self, model_dir: str, version: int = 1):
        print(f"Saving Agent model (ONNX) to:{model_dir}\n")
        self.actor.save_onnx(model_dir, version)
        self.critic.save_onnx(model_dir, version)


if __name__ == "__main__":
    env_name = args.env
    env = gym.make(env_name)
    agent = PPOAgent(env)
    agent.train(max_episodes=1)
    # Model saving
    model_dir = "trained_models"
    agent_name = f"PPO_{env_name}"
    agent_version = 1
    agent_model_path = os.path.join(model_dir, agent_name)
    agent.save(agent_model_path, agent_version)
    agent.save_onnx(agent_model_path, agent_version)
    agent.save_h5(agent_model_path, agent_version)
    agent.save_tfjs(agent_model_path, agent_version)
    agent.save_tflite(agent_model_path, agent_version)
