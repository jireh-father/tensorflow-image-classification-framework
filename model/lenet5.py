import tensorflow as tf

default_image_size = 32
default_last_conv_name = 'conv5'


def build_model(inputs, config, is_training):
    end_points = {}
    print(inputs)
    net = tf.layers.conv2d(inputs, 6, 5, padding="VALID", name="conv1")
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    net = tf.layers.conv2d(net, 6, 2, 2, padding="VALID", name="pool2")
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    # todo: sparse channel
    net = tf.layers.conv2d(net, 16, 5, padding="VALID", name="conv3")
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    net = tf.layers.conv2d(net, 6, 2, 2, padding="VALID", name="pool4")
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    net = tf.layers.conv2d(net, 120, 5, padding="VALID", name="conv5")
    end_points['conv5'] = net
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    net = tf.reshape(net, [-1, 120], name="reshape")
    net = tf.layers.dense(net, 84, name="fc6")
    # sigmoid or tanh?
    # net = tf.nn.sigmoid(net)
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    # todo: RBF output layer
    # weights = tf.Variable(tf.random_uniform([config.num_class, 84], -2.4 / 84, 2.4 / 84), name="output_weights")
    # net = tf.reduce_sum(net - weights, axis=-1)
    net = tf.layers.dense(net, config.num_class, name="fc7")
    # net = 1.7159 * tf.nn.tanh((2 / 3) * net)

    return net, end_points
