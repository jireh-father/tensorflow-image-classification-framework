import tensorflow as tf


def residual_block(net, block, repeat, name, use_stride=True, is_training=None):
    print("block_%s" % name)
    for i in range(repeat):
        short_cut = net
        for j, filter in enumerate(block):
            stride = 1
            if i == 0 and j == 0 and use_stride:
                stride = 2
            net = tf.layers.conv2d(net, filter[1], filter[0], stride, 'same', name="%s_%d_%d" % (name, i, j),
                                   use_bias=False)
            net = tf.layers.batch_normalization(net, training=is_training)
            print(net)
            if j > len(block) - 1:
                net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
        short_cut_channel = short_cut.get_shape()[3]
        last_layer_channel = net.get_shape()[3]

        stride = 1
        if i == 0 and use_stride:
            stride = 2

        if short_cut_channel == last_layer_channel:
            if stride > 1:
                short_cut = tf.layers.max_pooling2d(short_cut, 1, strides=stride)
        else:
            short_cut = tf.layers.conv2d(short_cut, int(net.get_shape()[3]), 1, stride, 'same',
                                         name="%s_projection_%d_%d" % (name, i, j))
        net += short_cut
        net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
    return net


def resnet_18(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[3, 64], [3, 64]], 2, "conv2", False, is_training=is_training)
    net = residual_block(net, [[3, 128], [3, 128]], 2, "conv3", True, is_training)
    net = residual_block(net, [[3, 256], [3, 256]], 2, "conv4", True, is_training)
    net = residual_block(net, [[3, 512], [3, 512]], 2, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def random_resnet_18(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[3, 64], [3, 64]], 2, "conv2", False, is_training=is_training)
    net = residual_block(net, [[3, 128], [3, 128]], 2, "conv3", True, is_training)
    net = residual_block(net, [[3, 256], [3, 256]], 2, "conv4", True, is_training)
    net = residual_block(net, [[3, 512], [3, 512]], 2, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_18_mar(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")

    conv2_ar1 = residual_block(net, [[3, 32], [3, 32]], 2, "conv2_ar1", False, is_training=is_training)
    conv3_ar1 = residual_block(conv2_ar1, [[3, 64], [3, 64]], 2, "conv3_ar1", True, is_training)
    conv4_ar1 = residual_block(conv3_ar1, [[3, 128], [3, 128]], 2, "conv4_ar1", True, is_training)
    conv5_ar1 = residual_block(conv4_ar1, [[3, 256], [3, 256]], 2, "conv5_ar1", True, is_training)

    conv2_ar3 = residual_block(net, [[[3, 1], 32], [[3, 1], 32]], 2, "conv2_ar3", False, is_training=is_training)
    conv3_ar3 = residual_block(conv2_ar3, [[[3, 1], 64], [[3, 1], 64]], 2, "conv3_ar3", True, is_training)
    conv4_ar3 = residual_block(conv3_ar3, [[[3, 1], 128], [[3, 1], 128]], 2, "conv4_ar3", True, is_training)
    conv5_ar3 = residual_block(conv4_ar3, [[[3, 1], 256], [[3, 1], 256]], 2, "conv5_ar3", True, is_training)

    conv2_ar03 = residual_block(net, [[[1, 3], 32], [[1, 3], 32]], 2, "conv2_ar03", False, is_training=is_training)
    conv3_ar03 = residual_block(conv2_ar03, [[[1, 3], 64], [[1, 3], 64]], 2, "conv3_ar03", True, is_training)
    conv4_ar03 = residual_block(conv3_ar03, [[[1, 3], 128], [[1, 3], 128]], 2, "conv4_ar03", True, is_training)
    conv5_ar03 = residual_block(conv4_ar03, [[[1, 3], 256], [[1, 3], 256]], 2, "conv5_ar03", True, is_training)

    conv5_merge = tf.concat([conv5_ar1, conv5_ar3, conv5_ar03], axis=3)
    print(conv5_merge)
    net = tf.reduce_mean(conv5_merge, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_18_mar_ms(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")

    conv2_ar1 = residual_block(net, [[3, 32], [3, 32]], 2, "conv2_ar1", False, is_training=is_training)
    conv3_ar1 = residual_block(conv2_ar1, [[3, 64], [3, 64]], 2, "conv3_ar1", True, is_training)
    conv4_ar1 = residual_block(conv3_ar1, [[3, 128], [3, 128]], 2, "conv4_ar1", True, is_training)
    conv5_ar1 = residual_block(conv4_ar1, [[3, 256], [3, 256]], 2, "conv5_ar1", True, is_training)

    conv2_ar3 = residual_block(net, [[[3, 1], 32], [[3, 1], 32]], 2, "conv2_ar3", False, is_training=is_training)
    conv3_ar3 = residual_block(conv2_ar3, [[[3, 1], 64], [[3, 1], 64]], 2, "conv3_ar3", True, is_training)
    conv4_ar3 = residual_block(conv3_ar3, [[[3, 1], 128], [[3, 1], 128]], 2, "conv4_ar3", True, is_training)
    conv5_ar3 = residual_block(conv4_ar3, [[[3, 1], 256], [[3, 1], 256]], 2, "conv5_ar3", True, is_training)

    conv2_ar03 = residual_block(net, [[[1, 3], 32], [[1, 3], 32]], 2, "conv2_ar03", False, is_training=is_training)
    conv3_ar03 = residual_block(conv2_ar03, [[[1, 3], 64], [[1, 3], 64]], 2, "conv3_ar03", True, is_training)
    conv4_ar03 = residual_block(conv3_ar03, [[[1, 3], 128], [[1, 3], 128]], 2, "conv4_ar03", True, is_training)
    conv5_ar03 = residual_block(conv4_ar03, [[[1, 3], 256], [[1, 3], 256]], 2, "conv5_ar03", True, is_training)

    conv5_merge = tf.concat([conv5_ar1, conv5_ar3, conv5_ar03], axis=3)
    print(conv5_merge)
    net = tf.reduce_mean(conv5_merge, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net

def resnet_34(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[3, 64], [3, 64]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[3, 128], [3, 128]], 4, "conv3", True, is_training)
    net = residual_block(net, [[3, 256], [3, 256]], 6, "conv4", True, is_training)
    net = residual_block(net, [[3, 512], [3, 512]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_50(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_50_fpn(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(conv2, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(conv3, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    conv5 = residual_block(conv4, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_class, 1, padding='same', name="classifier_output")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)

    p5_logits = tf.reduce_mean(p5_conv2, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv2, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv2, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]


def resnet_50_augnet(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2_ar1 = residual_block(net, [[1, 32], [3, 32], [1, 128]], 3, "conv2_ar1", False, is_training=is_training)
    conv3_ar1 = residual_block(conv2_ar1, [[1, 64], [3, 64], [1, 256]], 4, "conv3_ar1", True, is_training)
    conv4_ar1 = residual_block(conv3_ar1, [[1, 128], [3, 128], [1, 512]], 6, "conv4_ar1", True, is_training)
    conv5_ar1 = residual_block(conv4_ar1, [[1, 256], [3, 256], [1, 1024]], 3, "conv5_ar1", True, is_training)

    conv2_ar15 = residual_block(net, [[1, 32], [[3, 2], 32], [1, 128]], 3, "conv2_ar15", False, is_training=is_training)
    conv3_ar15 = residual_block(conv2_ar15, [[1, 64], [[3, 2], 64], [1, 256]], 4, "conv3_ar15", True, is_training)
    conv4_ar15 = residual_block(conv3_ar15, [[1, 128], [[3, 2], 128], [1, 512]], 6, "conv4_ar15", True, is_training)
    conv5_ar15 = residual_block(conv4_ar15, [[1, 256], [[3, 2], 256], [1, 1024]], 3, "conv5_ar15", True, is_training)

    conv2_ar07 = residual_block(net, [[1, 32], [[2, 3], 32], [1, 128]], 3, "conv2_ar07", False, is_training=is_training)
    conv3_ar07 = residual_block(conv2_ar07, [[1, 64], [[2, 3], 64], [1, 256]], 4, "conv3_ar07", True, is_training)
    conv4_ar07 = residual_block(conv3_ar07, [[1, 128], [[2, 3], 128], [1, 512]], 6, "conv4_ar07", True, is_training)
    conv5_ar07 = residual_block(conv4_ar07, [[1, 256], [[2, 3], 256], [1, 1024]], 3, "conv5_ar07", True, is_training)

    num_output_channel = 256
    m5_ar1 = tf.layers.conv2d(conv5_ar1, num_output_channel, 1, padding='same')
    m5_ar15 = tf.layers.conv2d(conv5_ar15, num_output_channel, 1, padding='same')
    m5_ar07 = tf.layers.conv2d(conv5_ar07, num_output_channel, 1, padding='same')
    m5 = tf.concat([m5_ar1, m5_ar15, m5_ar07], axis=3)
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4_ar1 = tf.layers.conv2d(conv4_ar1, num_output_channel, 1, padding='same')
    m4_ar15 = tf.layers.conv2d(conv4_ar15, num_output_channel, 1, padding='same')
    m4_ar07 = tf.layers.conv2d(conv4_ar07, num_output_channel, 1, padding='same')
    m4 = tf.concat([m4_ar1, m4_ar15, m4_ar07], axis=3)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3_ar1 = tf.layers.conv2d(conv3_ar1, num_output_channel, 1, padding='same')
    m3_ar15 = tf.layers.conv2d(conv3_ar15, num_output_channel, 1, padding='same')
    m3_ar07 = tf.layers.conv2d(conv3_ar07, num_output_channel, 1, padding='same')
    m3 = tf.concat([m3_ar1, m3_ar15, m3_ar07], axis=3)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2_ar1 = tf.layers.conv2d(conv2_ar1, num_output_channel, 1, padding='same')
    m2_ar15 = tf.layers.conv2d(conv2_ar15, num_output_channel, 1, padding='same')
    m2_ar07 = tf.layers.conv2d(conv2_ar07, num_output_channel, 1, padding='same')
    m2 = tf.concat([m2_ar1, m2_ar15, m2_ar07], axis=3)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_class, 1, padding='same', name="classifier_output")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)

    p5_logits = tf.reduce_mean(p5_conv2, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv2, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv2, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]


def resnet_50_fpn_merge(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(conv2, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(conv3, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    conv5 = residual_block(conv4, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1")

    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output")

    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return p2_logits


def resnet_50_fpn_deep_classifier(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(conv2, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(conv3, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    conv5 = residual_block(conv4, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_output_channel, 1, padding='same', name="classifier_conv2")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_output_channel, 1, padding='same', name="classifier_conv2", reuse=True)
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_output_channel, 1, padding='same', name="classifier_conv2", reuse=True)
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_output_channel, 1, padding='same', name="classifier_conv2", reuse=True)

    p5_conv3 = tf.layers.conv2d(p5_conv2, num_output_channel, 1, padding='same', name="classifier_conv3")
    p4_conv3 = tf.layers.conv2d(p4_conv2, num_output_channel, 1, padding='same', name="classifier_conv3", reuse=True)
    p3_conv3 = tf.layers.conv2d(p3_conv2, num_output_channel, 1, padding='same', name="classifier_conv3", reuse=True)
    p2_conv3 = tf.layers.conv2d(p2_conv2, num_output_channel, 1, padding='same', name="classifier_conv3", reuse=True)

    p5_conv4 = tf.layers.conv2d(p5_conv3, num_class, 1, padding='same', name="classifier_conv4")
    p4_conv4 = tf.layers.conv2d(p4_conv3, num_class, 1, padding='same', name="classifier_conv4", reuse=True)
    p3_conv4 = tf.layers.conv2d(p3_conv3, num_class, 1, padding='same', name="classifier_conv4", reuse=True)
    p2_conv4 = tf.layers.conv2d(p2_conv3, num_class, 1, padding='same', name="classifier_conv4", reuse=True)

    p5_logits = tf.reduce_mean(p5_conv4, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv4, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv4, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv4, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]


def resnet_50_fpn_classifier_no_sharing(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(conv2, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(conv3, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    conv5 = residual_block(conv4, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1_1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1_2")
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1_3")
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1_4")

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_class, 1, padding='same', name="classifier_output1")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_class, 1, padding='same', name="classifier_output2")
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_class, 1, padding='same', name="classifier_output3")
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output4")

    p5_logits = tf.reduce_mean(p5_conv2, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv2, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv2, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]


def resnet_101(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 23, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_101_fpn(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 23, "conv4", True, is_training)
    conv5 = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_class, 1, padding='same', name="classifier_output")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)

    p5_logits = tf.reduce_mean(p5_conv2, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv2, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv2, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]


def resnet_152(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 36, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_101_fpn_merge(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    conv2 = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    conv3 = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    conv4 = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 23, "conv4", True, is_training)
    conv5 = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)

    num_output_channel = 256
    m5 = tf.layers.conv2d(conv5, num_output_channel, 1, padding='same')

    m4 = tf.layers.conv2d(conv4, num_output_channel, 1, padding='same') + tf.image.resize_images(m5, (int(
        conv4.get_shape()[1]), int(conv4.get_shape()[2])), 1)

    m3 = tf.layers.conv2d(conv3, num_output_channel, 1, padding='same') + tf.image.resize_images(m4, (int(
        conv3.get_shape()[1]), int(conv3.get_shape()[2])), 1)

    m2 = tf.layers.conv2d(conv2, num_output_channel, 1, padding='same') + tf.image.resize_images(m3, (int(
        conv2.get_shape()[1]), int(conv2.get_shape()[2])), 1)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1")

    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output")

    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return p2_logits


def resnet_101_augnet(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")

    conv2_ar1 = residual_block(net, [[1, 32], [3, 32], [1, 128]], 3, "conv2_ar1", False, is_training=is_training)
    conv3_ar1 = residual_block(conv2_ar1, [[1, 64], [3, 64], [1, 256]], 4, "conv3_ar1", True, is_training)
    conv4_ar1 = residual_block(conv3_ar1, [[1, 128], [3, 128], [1, 512]], 23, "conv4_ar1", True, is_training)
    conv5_ar1 = residual_block(conv4_ar1, [[1, 256], [3, 256], [1, 1024]], 3, "conv5_ar1", True, is_training)

    conv2_ar15 = residual_block(net, [[1, 32], [[3, 2], 32], [1, 128]], 3, "conv2_ar15", False, is_training=is_training)
    conv3_ar15 = residual_block(conv2_ar15, [[1, 64], [[3, 2], 64], [1, 256]], 4, "conv3_ar15", True, is_training)
    conv4_ar15 = residual_block(conv3_ar15, [[1, 128], [[3, 2], 128], [1, 512]], 23, "conv4_ar15", True, is_training)
    conv5_ar15 = residual_block(conv4_ar15, [[1, 256], [[3, 2], 256], [1, 1024]], 3, "conv5_ar15", True, is_training)

    conv2_ar07 = residual_block(net, [[1, 32], [[2, 3], 32], [1, 128]], 3, "conv2_ar07", False, is_training=is_training)
    conv3_ar07 = residual_block(conv2_ar07, [[1, 64], [[2, 3], 64], [1, 256]], 4, "conv3_ar07", True, is_training)
    conv4_ar07 = residual_block(conv3_ar07, [[1, 128], [[2, 3], 128], [1, 512]], 23, "conv4_ar07", True, is_training)
    conv5_ar07 = residual_block(conv4_ar07, [[1, 256], [[2, 3], 256], [1, 1024]], 3, "conv5_ar07", True, is_training)

    num_output_channel = 256
    m5_ar1 = tf.layers.conv2d(conv5_ar1, num_output_channel, 1, padding='same')
    m5_ar15 = tf.layers.conv2d(conv5_ar15, num_output_channel, 1, padding='same')
    m5_ar07 = tf.layers.conv2d(conv5_ar07, num_output_channel, 1, padding='same')
    m5 = tf.concat([m5_ar1, m5_ar15, m5_ar07], axis=3)
    p5 = tf.layers.conv2d(m5, num_output_channel, 3, padding='same')

    m4_ar1 = tf.layers.conv2d(conv4_ar1, num_output_channel, 1, padding='same')
    m4_ar15 = tf.layers.conv2d(conv4_ar15, num_output_channel, 1, padding='same')
    m4_ar07 = tf.layers.conv2d(conv4_ar07, num_output_channel, 1, padding='same')
    m4 = tf.concat([m4_ar1, m4_ar15, m4_ar07], axis=3)
    p4 = tf.layers.conv2d(m4, num_output_channel, 3, padding='same')

    m3_ar1 = tf.layers.conv2d(conv3_ar1, num_output_channel, 1, padding='same')
    m3_ar15 = tf.layers.conv2d(conv3_ar15, num_output_channel, 1, padding='same')
    m3_ar07 = tf.layers.conv2d(conv3_ar07, num_output_channel, 1, padding='same')
    m3 = tf.concat([m3_ar1, m3_ar15, m3_ar07], axis=3)
    p3 = tf.layers.conv2d(m3, num_output_channel, 3, padding='same')

    m2_ar1 = tf.layers.conv2d(conv2_ar1, num_output_channel, 1, padding='same')
    m2_ar15 = tf.layers.conv2d(conv2_ar15, num_output_channel, 1, padding='same')
    m2_ar07 = tf.layers.conv2d(conv2_ar07, num_output_channel, 1, padding='same')
    m2 = tf.concat([m2_ar1, m2_ar15, m2_ar07], axis=3)
    p2 = tf.layers.conv2d(m2, num_output_channel, 3, padding='same')

    p5_conv1 = tf.layers.conv2d(p5, num_output_channel, 3, padding='same', name="classifier_conv1")
    p4_conv1 = tf.layers.conv2d(p4, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p3_conv1 = tf.layers.conv2d(p3, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)
    p2_conv1 = tf.layers.conv2d(p2, num_output_channel, 3, padding='same', name="classifier_conv1", reuse=True)

    p5_conv2 = tf.layers.conv2d(p5_conv1, num_class, 1, padding='same', name="classifier_output")
    p4_conv2 = tf.layers.conv2d(p4_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p3_conv2 = tf.layers.conv2d(p3_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)
    p2_conv2 = tf.layers.conv2d(p2_conv1, num_class, 1, padding='same', name="classifier_output", reuse=True)

    p5_logits = tf.reduce_mean(p5_conv2, [1, 2])
    p4_logits = tf.reduce_mean(p4_conv2, [1, 2])
    p3_logits = tf.reduce_mean(p3_conv2, [1, 2])
    p2_logits = tf.reduce_mean(p2_conv2, [1, 2])

    # net = tf.reduce_mean(conv5, [1, 2], name='last_pool')
    # net = tf.layers.dense(net, num_class, name='logits')
    return [p5_logits, p4_logits, p3_logits, p2_logits]
