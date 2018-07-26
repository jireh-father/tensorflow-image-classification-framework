import tensorflow as tf

default_image_size = 32
default_last_conv_name = 'conv5'


def build_model(inputs, config, is_training):
    end_points = {}
    print(inputs)
    net = tf.layers.conv2d(inputs, 6, 5, padding="VALID", name="conv1",
                           kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 1024,
                                                                            maxval=2.4 / 1024),
                           bias_initializer=tf.random_uniform_initializer(
                               minval=-2.4 / 1024,
                               maxval=2.4 / 1024))
    net = tf.layers.conv2d(net, 6, 2, 2, padding="VALID", name="pool2", activation=tf.nn.sigmoid,
                           kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 784,
                                                                            maxval=2.4 / 784),
                           bias_initializer=tf.random_uniform_initializer(
                               minval=-2.4 / 1024,
                               maxval=2.4 / 1024))
    # todo: sparse channel
    net = tf.layers.conv2d(net, 16, 5, padding="VALID", name="conv3",
                           kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 196,
                                                                            maxval=2.4 / 196),
                           bias_initializer=tf.random_uniform_initializer(
                               minval=-2.4 / 1024,
                               maxval=2.4 / 1024))
    net = tf.layers.conv2d(net, 6, 2, 2, padding="VALID", name="pool4", activation=tf.nn.sigmoid,
                           kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 100,
                                                                            maxval=2.4 / 100),
                           bias_initializer=tf.random_uniform_initializer(
                               minval=-2.4 / 1024,
                               maxval=2.4 / 1024))
    net = tf.layers.conv2d(net, 120, 5, padding="VALID", name="conv5",
                           kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 25,
                                                                            maxval=2.4 / 25),
                           bias_initializer=tf.random_uniform_initializer(
                               minval=-2.4 / 1024,
                               maxval=2.4 / 1024))
    end_points['conv5'] = net
    net = tf.reshape(net, [-1, 120], name="reshape")
    net = tf.layers.dense(net, 84, name="fc6",
                          kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 120,
                                                                           maxval=2.4 / 120),
                          bias_initializer=tf.random_uniform_initializer(
                              minval=-2.4 / 1024,
                              maxval=2.4 / 1024))
    # sigmoid or tanh?
    # net = tf.nn.sigmoid(net)
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)
    # todo: RBF output layer
    # weights = tf.Variable(tf.random_uniform([config.num_class, 84], -2.4 / 84, 2.4 / 84), name="output_weights")
    # net = tf.reduce_sum(net - weights, axis=-1)
    net = tf.layers.dense(net, config.num_class, name="fc7",
                          kernel_initializer=tf.random_uniform_initializer(minval=-2.4 / 120,
                                                                           maxval=2.4 / 120),
                          bias_initializer=tf.random_uniform_initializer(
                              minval=-2.4 / 1024,
                              maxval=2.4 / 1024))
    net = 1.7159 * tf.nn.tanh((2 / 3) * net)

    return net, end_points
