#!/usr/bin/env python
# SAC RL agent runtime componenets for deploying as a service
# Chapter 7, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import functools
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx("float64")


def actor(state_shape, action_shape, units=(512, 256, 64)):
    state_shape_flattened = functools.reduce(lambda x, y: x * y, state_shape)
    state = Input(shape=state_shape_flattened)
    x = Dense(units[0], name="L0", activation="relu")(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

    actions_mean = Dense(action_shape[0], name="Out_mean")(x)
    actions_std = Dense(action_shape[0], name="Out_std")(x)

    model = Model(inputs=state, outputs=[actions_mean, actions_std])

    return model


def critic(state_shape, action_shape, units=(512, 256, 64)):
    state_shape_flattened = functools.reduce(lambda x, y: x * y, state_shape)
    inputs = [Input(shape=state_shape_flattened), Input(shape=action_shape)]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="Hidden0", activation="relu")(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

    output = Dense(1, name="Out_QVal")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


class SAC(object):
    def __init__(
        self,
        observation_shape,
        action_space,
        lr_actor=3e-5,
        lr_critic=3e-4,
        actor_units=(64, 64),
        critic_units=(64, 64),
        auto_alpha=True,
        alpha=0.2,
        tau=0.005,
        gamma=0.99,
        batch_size=128,
        memory_cap=100000,
    ):
        self.state_shape = observation_shape  # shape of observations
        self.action_shape = action_space.shape  # number of actions
        self.action_bound = (action_space.high - action_space.low) / 2
        self.action_shift = (action_space.high + action_space.low) / 2
        self.memory = deque(maxlen=int(memory_cap))

        # Define and initialize actor network
        self.actor = actor(self.state_shape, self.action_shape, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.log_std_min = -20
        self.log_std_max = 2
        print(self.actor.summary())

        # Define and initialize critic networks
        self.critic_1 = critic(self.state_shape, self.action_shape, critic_units)
        self.critic_target_1 = critic(self.state_shape, self.action_shape, critic_units)
        self.critic_optimizer_1 = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_1, self.critic_target_1, tau=1.0)

        self.critic_2 = critic(self.state_shape, self.action_shape, critic_units)
        self.critic_target_2 = critic(self.state_shape, self.action_shape, critic_units)
        self.critic_optimizer_2 = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_2, self.critic_target_2, tau=1.0)

        print(self.critic_1.summary())

        # Define and initialize temperature alpha and target entropy
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.prod(self.action_shape)
            self.log_alpha = tf.Variable(0.0, dtype=tf.float64)
            self.alpha = tf.Variable(0.0, dtype=tf.float64)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_optimizer = Adam(learning_rate=lr_actor)
        else:
            self.alpha = tf.Variable(alpha, dtype=tf.float64)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # Tensorboard
        self.summaries = {}

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.math.exp(log_std)
        raw_actions = mean

        if not test:
            raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float64) * std

        log_prob_u = tfp.distributions.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))

        actions = actions * self.action_bound + self.action_shift

        return actions, log_prob

    def act(self, state, test=False, use_random=False):
        state = state.reshape(-1)  # Flatten state
        state = np.expand_dims(state, axis=0).astype(np.float64)

        if use_random:
            a = tf.random.uniform(
                shape=(1, self.action_shape[0]), minval=-1, maxval=1, dtype=tf.float64
            )
        else:
            means, log_stds = self.actor.predict(state)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)

            a, log_prob = self.process_actions(means, log_stds, test=test)

        q1 = self.critic_1.predict([state, a])[0][0]
        q2 = self.critic_2.predict([state, a])[0][0]
        self.summaries["q_min"] = tf.math.minimum(q1, q2)
        self.summaries["q_mean"] = np.mean([q1, q2])

        return a

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic_1.load_weights(c_fn)
        self.critic_target_1.load_weights(c_fn)
        self.critic_2.load_weights(c_fn)
        self.critic_target_2.load_weights(c_fn)
        print(self.critic_1.summary())
