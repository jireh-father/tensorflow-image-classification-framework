import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def build_model(inputs, config, is_training):
    end_points = {}
    net = tf.layers.conv2d(inputs, 64, 11, 4, padding="VALID", activation=tf.nn.relu, name="conv1")
    net = tf.layers.max_pooling2d(net, 3, 2, name="max_pool1")

    attention_mask = tf.layers.conv2d(net, 1, 1, padding="SAME")
    shape = attention_mask.get_shape()
    print("shape", shape)
    # features = tf.reshape(tf.transpose(attention_mask, [0, 3, 1, 2]), [None, int(shape[1]) * int(shape[2])])
    features = tf.reshape(tf.transpose(attention_mask, [0, 3, 1, 2]),
                          [config.batch_size * int(shape[3]), int(shape[1]) * int(
                              shape[2])])
    spatial_softmax = tf.nn.softmax(features)
    # Reshape and transpose back to original format.
    spatial_softmax = tf.transpose(tf.reshape(spatial_softmax, [config.batch_size, int(shape[3]), int(shape[1]),
                                                                int(shape[2])]), [0, 2, 3, 1])

    # spatial_softmax = tf.transpose(tf.reshape(spatial_softmax, [None, int(shape[3]), int(shape[1]), int(shape[2])]),
    #                                [0, 2, 3, 1])

    attention_head = tf.multiply(net, spatial_softmax)
    print(attention_head)
    ouput_head = tf.layers.average_pooling2d(attention_head, int(attention_head.get_shape()[1]), strides=1)
    print(ouput_head)
    ouput_head = tf.squeeze(ouput_head)
    attention_prediction = tf.layers.dense(ouput_head, config.num_class)
    # ouput_head = tf.reshape(ouput_head, [None, 1, int(ouput_head.get_shape()[3])])
    # w = tf.get_variable(shape=[config.batch_size, 1, int(attention_head.get_shape()[3])], dtype=tf.float32,
    #                     initializer=tf.initializers.truncated_normal, name="head_weights")
    # print(w)
    # ouput_head = tf.multiply(attention_head, w)
    confidence = tf.nn.tanh(tf.layers.dense(attention_prediction, config.num_class))
    gate_weights = tf.nn.softmax(confidence)
    attention_output = tf.multiply(attention_prediction, gate_weights)
    # print(attention_output)
    # sys.exit()

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
    net_gate_weights = tf.nn.softmax(tf.nn.tanh(tf.layers.dense(net, config.num_class)))
    net_output = tf.multiply(net, net_gate_weights)
    net = net_output + attention_output
    return net, end_points
