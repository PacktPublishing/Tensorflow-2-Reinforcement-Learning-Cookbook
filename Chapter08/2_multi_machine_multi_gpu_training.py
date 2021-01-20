#!/usr/bin/env python
# Recipe to accelerate custom model training with multi-machine, multi-GPU training
# Chapter 8, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import sys

import tensorflow as tf
from tensorflow import keras

if "." not in sys.path:
    sys.path.insert(0, ".")
import resnet

## Uncomment the following lines and fill worker details based on your cluster configuration
# tf_config = {
#    "cluster": {"worker": ["1.2.3.4:1111", "localhost:2222"]},
#    "task": {"index": 0, "type": "worker"},
# }
# os.environ["TF_CONFIG"] = json.dumps(tf_config)
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1


def normalize(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = (
    train_dataset.map(augmentation)
    .map(normalize)
    .shuffle(NUM_TRAIN_SAMPLES)
    .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
)
test_dataset = test_dataset.map(normalize).batch(
    BS_PER_GPU * NUM_GPUS, drop_remainder=True
)


input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


with strategy.scope():
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
model.fit(train_dataset, epochs=NUM_EPOCHS)
