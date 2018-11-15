import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def fire_module(inputs, s_filters, e_1x1_filters, e_3x3_filters, name):
    squeeze = tf.layers.conv2d(inputs, s_filters, 1, padding="SAME", name=name + "_squeeze", activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    squeeze = tf.layers.batch_normalization(squeeze)
    e1x1 = tf.layers.conv2d(squeeze, e_1x1_filters, 1, padding="SAME", name=name + "_expend1x1", activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    e1x1 = tf.layers.batch_normalization(e1x1)

    e3x3 = tf.layers.conv2d(squeeze, e_3x3_filters, 3, padding="SAME", name=name + "_expend3x3", activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    e3x3 = tf.layers.batch_normalization(e3x3)
    return tf.concat([e1x1, e3x3], axis=3)


def build_model(inputs, config, is_training):
    end_points = {}
    net = tf.layers.conv2d(inputs, 96, 7, strides=2, padding="SAME", name="conv1", activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.batch_normalization(net)
    print(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool1")
    print(net)
    net = fire_module(net, 16, 64, 64, "fire2")
    print(net)
    net = fire_module(net, 16, 64, 64, "fire3")
    print(net)
    net = fire_module(net, 32, 128, 128, "fire4")
    print(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool2")
    net = fire_module(net, 32, 128, 128, "fire5")
    print(net)
    net = fire_module(net, 48, 192, 192, "fire6")
    print(net)
    net = fire_module(net, 48, 192, 192, "fire7")
    print(net)
    net = fire_module(net, 64, 256, 256, "fire8")
    print(net)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool3")
    net = fire_module(net, 64, 256, 256, "fire9")
    print(net)
    net = tf.layers.dropout(net, 0.5, training=is_training)
    net = tf.layers.conv2d(net, config.num_class, 1, padding="VALID", name="conv10", activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.batch_normalization(net)
    print(net)
    end_points['conv5'] = net
    # net = tf.reduce_mean(net, [1, 2], name='avg_pooling')
    net = tf.layers.average_pooling2d(net, [13, 13], strides=1, name='avgpool10')
    print(net)
    net = tf.layers.flatten(net, name='logits')
    print(net)
    total_parameters = 0

    # for variable in tf.trainable_variables():
    #     # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #     print(shape)
    #     print(len(shape))
    #     variable_parameters = 1
    #     for dim in shape:
    #         print(dim)
    #         variable_parameters *= dim.value
    #     print(variable_parameters)
    #     total_parameters += variable_parameters
    # print("total parameters", total_parameters)

    return net, end_points
