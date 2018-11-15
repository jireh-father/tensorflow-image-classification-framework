import tensorflow as tf

default_image_size = 224
default_last_conv_name = 'conv5'


def fire_module(inputs, s_filters, e_1x1_filters, e_3x3_filters, name):
    s_filters *= 4
    squeeze = tf.layers.conv2d(inputs, s_filters, 1, padding="SAME", name=name + "_squeeze", activation=tf.nn.relu)
    squeeze = tf.layers.batch_normalization(squeeze)
    e1x1 = tf.layers.conv2d(squeeze, e_1x1_filters, 1, padding="SAME", name=name + "_expend1x1", activation=tf.nn.relu)
    e1x1 = tf.layers.batch_normalization(e1x1)
    e3x3 = tf.layers.conv2d(squeeze, e_3x3_filters, 3, padding="SAME", name=name + "_expend3x3", activation=tf.nn.relu)
    e3x3 = tf.layers.batch_normalization(e3x3)
    return tf.concat([e1x1, e3x3], axis=3)


def build_model(inputs, config, is_training):
    end_points = {}
    net = tf.layers.conv2d(inputs, 96, 7, strides=2, padding="SAME", name="conv1", activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    print(net)
    sc1 = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool1")

    print(net)
    net = fire_module(sc1, 16, 64, 64, "fire2")

    sc2 = tf.layers.conv2d(inputs=sc1, filters=int(net.get_shape()[3]), kernel_size=1, padding="SAME") + net
    print(net)
    net = fire_module(sc2, 16, 64, 64, "fire3")
    print(net)
    sc3 = net + sc2

    net = fire_module(sc3, 32, 128, 128, "fire4")

    net = tf.layers.conv2d(inputs=sc3, filters=int(net.get_shape()[3]), kernel_size=1, padding="SAME") + net
    print(net)

    sc4 = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool2")

    net = fire_module(sc4, 32, 128, 128, "fire5")
    sc5 = sc4 + net
    print(net)

    net = fire_module(sc5, 48, 192, 192, "fire6")
    sc6 = tf.layers.conv2d(inputs=sc5, filters=int(net.get_shape()[3]), kernel_size=1, padding="SAME") + net
    print(sc6)
    net = fire_module(sc6, 48, 192, 192, "fire7")
    sc7 = sc6 + net
    print(net)
    net = fire_module(sc7, 64, 256, 256, "fire8")
    net = tf.layers.conv2d(inputs=sc7, filters=int(net.get_shape()[3]), kernel_size=1, padding="SAME") + net
    print(net)
    sc8 = tf.layers.max_pooling2d(inputs=net, pool_size=3, strides=2, name="pool3")
    net = fire_module(sc8, 64, 256, 256, "fire9")
    print(net)
    net += sc8
    net = tf.layers.dropout(net, 0.5, training=is_training)
    net = tf.layers.conv2d(net, config.num_class, 1, padding="VALID", name="conv10", activation=tf.nn.relu)
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
