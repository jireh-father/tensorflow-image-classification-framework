import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def build_model(inputs, config, is_training):
    end_points = {}
    net = tf.layers.conv2d(inputs, 64, 11, 4, padding="VALID", activation=tf.nn.relu, name="conv1")
    net = tf.layers.max_pooling2d(net, 3, 2, name="max_pool1")
    net = tf.layers.conv2d(net, 128, 5, activation=tf.nn.relu, name="conv2")
    net = tf.layers.max_pooling2d(net, 3, 2, name="max_pool2")
    net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu, name="conv3")
    net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu, name="conv4")
    net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu, name="conv5")
    end_points["conv5"] = net
    net = tf.layers.max_pooling2d(net, 3, 2, name="max_pool3")
    net = tf.reshape(net, [-1, int(net.get_shape()[1]) * int(net.get_shape()[2]) * int(net.get_shape()[3])])
    net = tf.layers.dense(net, 512, activation=tf.nn.relu, name="fc1")
    net = tf.layers.dense(net, 256, activation=tf.nn.relu, name="fc2")
    net = tf.layers.dense(net, config.num_class, name="fc3")
    return net, end_points
