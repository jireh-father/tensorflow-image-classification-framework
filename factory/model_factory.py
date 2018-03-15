from defined.model import model_factory
from core import util
import tensorflow as tf


def build_model(config):
    inputs = tf.placeholder(tf.float32, shape=[None, config.input_size, config.input_size, config.num_channel],
                            name="inputs")
    labels = tf.placeholder(tf.float32, shape=[None, config.num_classes], name="labels")
    if config.model_name in model_factory.networks_map.keys():
        if config.model_name[:6] == "nasnet":
            is_training = config.train
        else:
            is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        model_f = model_factory.get_network_fn(config.model_name, config.num_classes, weight_decay=config.weight_decay,
                                               is_training=is_training)

        logits, end_points = model_f(inputs)

        if config.model_name[:6] == "resnet":
            logits = tf.reshape(logits, [-1, config.num_classes])

    else:
        build_model_f = util.get_func('model.%s' % config.model_name, "build_model")
        if not build_model_f:
            return None
        model_result = build_model_f(inputs, config)
        if isinstance(model_result, list):
            logits = model_result[0]
            end_points = model_result[1]
        else:
            logits = model_result
            end_points = None

    if hasattr(config, 'cost_func'):
        cost_func = config.cost_func
    else:
        cost_func = "softmax_cross_entropy"
    cost_f = util.get_func('cost.%s' % cost_func, "build_cost")
    ops = None
    if cost_f:
        ops = cost_f(logits, labels, config)

    return inputs, labels, logits, end_points, ops
