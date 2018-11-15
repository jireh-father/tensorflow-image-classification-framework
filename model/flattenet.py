import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def flattend_layer(inputs, filters, kernel_size, strides, padding, name):
    net = tf.layers.conv2d(inputs, filters, 1, strides=strides, padding=padding, name=name + "_lateral1")
    channels = tf.split(net, num_or_size_splits=filters, axis=3)
    print(channels)
    concatted = None
    for i, channel in enumerate(channels):
        channel = tf.layers.conv2d(channel, 1, [kernel_size, 1], padding="same",
                                   name=name + "_vertical1_%d" % i)
        print("v", channel)
        channel = tf.layers.conv2d(channel, 1, [1, kernel_size], padding="same",
                                   name=name + "_horizontal1_%d" % i)
        print("h", channel)
        if concatted is None:
            concatted = channel
        else:
            concatted = tf.concat([concatted, channel], axis=3)
    print(concatted)
    net = tf.layers.conv2d(concatted, filters, 1, padding="same", name=name + "_lateral2")
    channels = tf.split(net, num_or_size_splits=filters, axis=3)
    concatted = None
    print(channels)
    for i, channel in enumerate(channels):
        print(channel)
        channel = tf.layers.conv2d(channel, 1, [kernel_size, 1], padding="same",
                                   name=name + "_vertical2_%d" % i)
        channel = tf.layers.conv2d(channel, 1, [1, kernel_size], padding="same",
                                   name=name + "_horizontal2_%d" % i)
        if concatted is None:
            concatted = channel
        else:
            concatted = tf.concat([concatted, channel], axis=3)

    return tf.nn.relu(concatted)


def build_model(inputs, config, is_training):
    end_points = {}
    net = flattend_layer(inputs, 96, 11, 4, "VALID", name="conv1")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool1")

    net = flattend_layer(net, 256, 5, strides=1, padding="VALID", name="conv2")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool2")

    net = flattend_layer(net, 384, 3, strides=1, padding="SAME", name="conv3")
    net = flattend_layer(net, 384, 3, strides=1, padding="SAME", name="conv4")
    net = flattend_layer(net, 256, 3, strides=1, padding="SAME", name="conv5")
    end_points['conv5'] = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool5")
    print(net)

    net = tf.reshape(net, [-1, int(net.get_shape()[1]) * int(net.get_shape()[2]) * int(net.get_shape()[3])],
                     name="reshape")
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name="fc6")
    net = tf.layers.dropout(net, training=is_training)
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name="fc7")
    net = tf.layers.dropout(net, training=is_training)
    net = tf.layers.dense(net, config.num_class, activation=tf.nn.relu, name="fc8")

    return net, end_points
