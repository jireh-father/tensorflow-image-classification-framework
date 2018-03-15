import tensorflow as tf
from core import optimizer


def build_cost(logits, labels, config):
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    if config.use_regularizer:
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularizer = 0
        for weight in weights:
            regularizer += tf.nn.l2_loss(weight)
        regularizer *= config.weight_decay
        cost_op += regularizer
    tf.summary.scalar('cost', cost_op)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = optimizer.configure_learning_rate(global_step, config)
    opt = optimizer.configure_optimizer(learning_rate, config)
    train_op = opt.minimize(cost_op, global_step=global_step)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    pred_idx_op = tf.argmax(logits, 1)

    return {"cost": cost_op, "train": train_op, "accuracy": accuracy_op, "learning_rate": learning_rate,
            "pred_idx": pred_idx_op}
