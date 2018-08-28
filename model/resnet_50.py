import tensorflow as tf


def residual_block(net, block, repeat, name, use_stride=True, use_projection=True, is_training=None):
    for i in range(repeat):
        short_cut = net
        for j, filter in enumerate(block):
            stride = 1
            if i == 0 and j == 0 and use_stride:
                stride = 2
                print("stride")
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
            if use_projection:
                short_cut = tf.layers.conv2d(short_cut, int(net.get_shape()[3]), 1, stride, 'same',
                                             name="%s_projection_%d_%d" % (name, i, j))
            else:
                tf.pad(net, tf.constant([[[[]]]]))
                pass
        # print("short_cut", short_cut)
        net += short_cut
        net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
        # print(net)
    return net


default_image_size = 224
default_last_conv_name = 'conv5'


def build_model(input, config, is_training):
    end_points = {}
    use_projection = True
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, use_projection, is_training=is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, use_projection,
                         is_training=is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, use_projection,
                         is_training=is_training)
    end_points['conv5'] = net
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    print(net)
    net = tf.layers.dense(net, config.num_class, name='logits')
    print("last", net)
    return net, end_points
