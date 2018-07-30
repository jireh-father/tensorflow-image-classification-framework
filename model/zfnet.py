import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def build_model(inputs, config, is_training):
    end_points = {}
    net = tf.layers.conv2d(inputs, 96, 7, strides=2, padding="VALID", name="conv1", activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool1")
    net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=0.0001, beta=0.75, name="lrn1")
    net = tf.layers.conv2d(net, 256, 5, strides=2, padding="VALID", name="conv2", activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool2")
    net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=0.0001, beta=0.75, name="lrn2")
    net = tf.layers.conv2d(net, 384, 3, padding="SAME", name="conv3", activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 384, 3, padding="SAME", name="conv4", activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 256, 3, padding="SAME", name="conv5", activation=tf.nn.relu)
    end_points['conv5'] = net
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool5")
    net = tf.reshape(net, [-1, int(net.get_shape()[1]) * int(net.get_shape()[2]) * int(net.get_shape()[3])],
                     name="reshape")
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name="fc6")
    net = tf.layers.dropout(net, training=is_training)
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name="fc7")
    net = tf.layers.dropout(net, training=is_training)
    net = tf.layers.dense(net, config.num_class, activation=tf.nn.relu, name="fc8")

    return net, end_points
