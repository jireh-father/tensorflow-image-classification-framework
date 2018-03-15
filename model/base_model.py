import tensorflow as tf


def build_model(inputs, config):
    return tf.layers.conv2d(inputs, 3, 3)

# def build_cost(config):
#     pass
#
#
# def build_train(config):
#     pass
