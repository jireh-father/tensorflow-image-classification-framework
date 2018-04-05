from slim.model import model_factory
from core import util
import tensorflow as tf
from core import optimizer


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
        build_model_f = util.get_attr('model.%s' % config.model_name, "build_model")
        if not build_model_f:
            return None
        if not config.input_size:
            config.input_size = util.get_attr('model.%s' % config.model_name, "input_size")
            if not config.input_size:
                config.input_size = 224
        default_last_conv_name = util.get_attr('model.%s' % config.model_name, "default_last_conv_name")

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

    loss_f = util.get_attr('model.loss.%s' % loss_file, "build_loss")
    ops = None
    if loss_f:
        ops = loss_f(logits, labels, global_step, config)

    return inputs, labels, logits, end_points, is_training, global_step, default_last_conv_name, ops


def build_model_multiple(config, dataset):
    # labels = tf.placeholder(tf.float32, shape=[None, config.num_class], name="labels")
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
    default_last_conv_name = None
    learning_rate = optimizer.configure_learning_rate(global_step, config)
    opt = optimizer.configure_optimizer(learning_rate, config)
    tower_grads = []
    if config.gpu_list:
        gpu_list = [int(i) for i in config.gpu_list.split(",")]
    else:
        gpu_list = range(config.num_gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in gpu_list:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("tower", i)) as scope:

                    if config.model_name in model_factory.networks_map.keys():
                        if config.model_name[:6] == "nasnet":
                            is_training = config.train
                        model_f = model_factory.get_network_fn(config.model_name, config.num_class,
                                                               weight_decay=config.weight_decay,
                                                               is_training=is_training)

                        if hasattr(model_f, 'default_last_conv_name'):
                            default_last_conv_name = model_f.default_last_conv_name

                        if not config.input_size:
                            config.input_size = model_f.default_image_size
                        # inputs = tf.placeholder(tf.float32, shape=[None, config.input_size, config.input_size, config.num_channel],
                        #                         name="inputs")
                        # tf.summary.image('input', inputs, config.num_input_summary)
                        logits, end_points = model_f(dataset.images[i])

                        if config.model_name[:6] == "resnet":
                            logits = tf.reshape(logits, [-1, config.num_class])

                    else:
                        build_model_f = util.get_attr('model.%s' % config.model_name, "build_model")
                        if not build_model_f:
                            return None
                        if not config.input_size:
                            config.input_size = util.get_attr('model.%s' % config.model_name, "input_size")
                            if not config.input_size:
                                config.input_size = 224
                        default_last_conv_name = util.get_attr('model.%s' % config.model_name, "default_last_conv_name")

                        # inputs = tf.placeholder(tf.float32, shape=[None, config.input_size, config.input_size, config.num_channel],
                        #                         name="inputs")
                        tf.summary.image('input', dataset.images[i], config.num_input_summary)
                        model_result = build_model_f(dataset.images[i], config, is_training=is_training)
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

                    loss_f = util.get_attr('model.loss.%s' % loss_file, "build_loss_multiple")
                    ops = None
                    if loss_f:
                        loss = loss_f(logits, dataset.labels[i], global_step, config, scope, opt)

                        tower_grads.append(loss)
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return train_op, is_training, global_step, default_last_conv_name, grads


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
