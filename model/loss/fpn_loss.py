import tensorflow as tf
from core import optimizer
import re


def build_loss(logits_list, labels, global_step, class_weights_ph, config):
    loss_op = None
    probs = None
    for logits in logits_list:
        if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        if loss_op is None:
            loss_op = loss
        else:
            loss_op += loss
        prob = tf.nn.softmax(logits)
        if probs is None:
            probs = prob
        else:
            probs += prob
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
        learning_rate = optimizer.configure_learning_rate(global_step, config)
        opt = optimizer.configure_optimizer(learning_rate, config)
        train_op = opt.minimize(loss_op, global_step=global_step)
        result["learning_rate"] = learning_rate
        result["train"] = train_op
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1)), tf.float32))
    pred_idx_op = tf.argmax(probs, 1)
    result["accuracy"] = accuracy_op
    result["pred_idx"] = pred_idx_op
    return result


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
