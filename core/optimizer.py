import tensorflow as tf


def configure_optimizer(learning_rate, conf):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if conf.optimizer is not recognized.
    """
    if conf.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=conf.adadelta_rho,
            epsilon=conf.opt_epsilon)
    elif conf.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=conf.adagrad_initial_accumulator_value)
    elif conf.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=conf.adam_beta1,
            beta2=conf.adam_beta2,
            epsilon=conf.opt_epsilon)
    elif conf.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=conf.ftrl_learning_rate_power,
            initial_accumulator_value=conf.ftrl_initial_accumulator_value,
            l1_regularization_strength=conf.ftrl_l1,
            l2_regularization_strength=conf.ftrl_l2)
    elif conf.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=conf.momentum,
            name='Momentum')
    elif conf.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=conf.rmsprop_decay,
            momentum=conf.rmsprop_momentum,
            epsilon=conf.opt_epsilon)
    elif conf.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', conf.optimizer)
    return optimizer


def configure_learning_rate(global_step, config):
    decay_steps = int(config.num_train_sample / config.batch_size *
                      config.num_epochs_per_decay)

    if config.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(config.learning_rate,
                                          global_step,
                                          decay_steps,
                                          config.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif config.learning_rate_decay_type == 'fixed':
        return tf.constant(config.learning_rate, name='fixed_learning_rate')
    elif config.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(config.learning_rate,
                                         global_step,
                                         decay_steps,
                                         config.end_learning_rate,
                                         power=1.0,
                                         cycle=config.polynomial_learning_rate_cycle,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         config.learning_rate_decay_type)
