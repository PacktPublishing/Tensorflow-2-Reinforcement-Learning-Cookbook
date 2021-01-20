#!/usr/bin/env python
# Recipe for accelerating custom model training using multi-GPU distributed training including model saving & loading
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy
# Based on official TensorFlow docs/tutorial

import sys

import tensorflow as tf
import tensorflow_datasets as tfds

if "." not in sys.path:
    sys.path.insert(0, ".")

import os

import resnet

dataset_name = "dmlab"  # "cifar10" or "cifar100"; See tensrflow.org/datasets/catalog for complete list
# NOTE: dmlab is large in size; Download bandwidth and GPU memory to be considered
datasets, info = tfds.load(name="dmlab", with_info=True, as_supervised=True)
dataset_train, dataset_test = datasets["train"], datasets["test"]
input_shape = info.features["image"].shape
num_classes = info.features["label"].num_classes

strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

num_train_examples = info.splits["train"].num_examples
num_test_examples = info.splits["test"].num_examples

BUFFER_SIZE = 1000  # Increase as per available memory
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = (
    dataset_train.map(preprocess).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)
eval_dataset = dataset_test.map(preprocess).batch(BATCH_SIZE)


def create_model(model_name=None):
    if model_name == "resnet_mini":
        model = resnet.resnet_mini(img_input=img_input, classes=num_classes)
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(6),
            ]
        )
    print(model.summary())
    return model


img_input = tf.keras.layers.Input(shape=input_shape)

with strategy.scope():
    # model = create_model()
    model = create_model("resnet_mini")
    tf.keras.utils.plot_model(model, to_file="./slim_resnet.png", show_shapes=True)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir="./logs", write_images=True, update_freq="batch"
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    ),
]

model.fit(train_dataset, epochs=12, callbacks=callbacks)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

path = "saved_model/"

model.save(path, save_format="tf")

unreplicated_model = tf.keras.models.load_model(path)

unreplicated_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))

with strategy.scope():
    replicated_model = tf.keras.models.load_model(path)
    replicated_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))
