import tensorflow as tf
from core import optimizer
import re


def build_loss(logits, labels, global_step, class_weights_ph, config):
    if config.use_weighted_loss:
        logits_class_weights = tf.reduce_sum(labels * class_weights_ph, axis=1)
        loss_op = tf.losses.softmax_cross_entropy(labels, logits, weights=logits_class_weights)
    else:
        if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits),
                                     name="softmax_cross_entropy")
        else:
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits),
                                     name="softmax_cross_entropy")
    # todo : tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.add(loss, tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    if config.use_regularizer:
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = 0
        for weight in weights:
            regularizer += tf.nn.l2_loss(weight)
        regularizer *= config.weight_decay
        loss_op += regularizer
    result = {"loss": loss_op}
    if config.train:
        train_op, learning_rate = train_op_fun(loss_op, global_step, config)
        # learning_rate = optimizer.configure_learning_rate(global_step, config)
        # opt = optimizer.configure_optimizer(learning_rate, config)
        # train_op = opt.minimize(loss_op, global_step=global_step)
        result["learning_rate"] = learning_rate
        result["train"] = train_op
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    pred_idx_op = tf.argmax(logits, 1)
    result["accuracy"] = accuracy_op
    result["pred_idx"] = pred_idx_op
    return result

def _get_variables_to_train(cf):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if cf.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in cf.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def train_op_fun(total_loss, global_step, cf):
    """Train model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    lr = optimizer.configure_learning_rate(global_step, cf)
    tf.summary.scalar('learning_rate', lr)
    opt = optimizer.configure_optimizer(lr, cf)
    variables_to_train = _get_variables_to_train(cf)
    grads = opt.compute_gradients(total_loss, variables_to_train)
    grad_updates = opt.apply_gradients(grads, global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_op = tf.identity(total_loss, name='train_op')

    return train_op, lr

def build_loss_multiple(logits, labels, global_step, config, scope, opt):
    if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits),
                                 name="softmax_cross_entropy")
    else:
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits),
                                 name="softmax_cross_entropy")
    # todo : tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.add(loss, tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    if config.use_regularizer:
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = 0
        for weight in weights:
            regularizer += tf.nn.l2_loss(weight)
        regularizer *= config.weight_decay
        loss_op += regularizer

    tf.add_to_collection('losses', loss_op)
    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % "tower", '', l.op.name)
        tf.summary.scalar(loss_name, l)

    tf.get_variable_scope().reuse_variables()
    return opt.compute_gradients(total_loss)

    # train_op = opt.minimize(loss_op, global_step=global_step)
    # accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    # pred_idx_op = tf.argmax(logits, 1)
    #
    # return {"loss": loss_op, "train": train_op, "accuracy": accuracy_op, "learning_rate": learning_rate,
    #         "pred_idx": pred_idx_op}
