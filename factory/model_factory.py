from slim.model import model_factory
from core import util
import tensorflow as tf


def build_model(config):
    labels = tf.placeholder(tf.float32, shape=[None, config.num_class], name="labels")
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
    default_last_conv_name = None
    if config.model_name in model_factory.networks_map.keys():
        if config.model_name[:6] == "nasnet":
            is_training = config.train
        model_f = model_factory.get_network_fn(config.model_name, config.num_class, weight_decay=config.weight_decay,
                                               is_training=is_training)

        if hasattr(model_f, 'default_last_conv_name'):
            default_last_conv_name = model_f.default_last_conv_name

        if not config.input_size:
            config.input_size = model_f.default_image_size
        inputs = tf.placeholder(tf.float32, shape=[None, config.input_size, config.input_size, config.num_channel],
                                name="inputs")
        tf.summary.image('input', inputs, config.num_input_summary)
        logits, end_points = model_f(inputs)

        if config.model_name[:6] == "resnet":
            logits = tf.reshape(logits, [-1, config.num_class])

    else:
        build_model_f = util.get_func('model.%s' % config.model_name, "build_model")
        if not build_model_f:
            return None
        model_module = util.get_module('model.%s' % config.model_name)
        if not config.input_size:
            config.input_size = getattr(model_module, "input_size", 224)
        if hasattr(model_module, "default_last_conv_name"):
            default_last_conv_name = getattr(model_module, "default_last_conv_name")

        inputs = tf.placeholder(tf.float32, shape=[None, config.input_size, config.input_size, config.num_channel],
                                name="inputs")
        tf.summary.image('input', inputs, config.num_input_summary)
        model_result = build_model_f(inputs, config, is_training=is_training)
        if isinstance(model_result, tuple):
            logits = model_result[0]
            end_points = model_result[1]
        else:
            logits = model_result
            end_points = None

    if hasattr(config, 'loss_file'):
        loss_file = config.loss_file
    else:
        loss_file = "softmax_cross_entropy"

    loss_f = util.get_func('model.loss.%s' % loss_file, "build_loss")
    ops = None
    if loss_f:
        ops = loss_f(logits, labels, global_step, config)

    return inputs, labels, logits, end_points, is_training, global_step, default_last_conv_name, ops
