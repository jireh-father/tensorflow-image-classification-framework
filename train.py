import tensorflow as tf
from core import util, init
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name', "mnist", "dataset name")
tf.app.flags.DEFINE_string('dataset_dir', "F:\data\mnist", "dataset_dir")
tf.app.flags.DEFINE_string('model_name', "alexnet_v2", "model name")
tf.app.flags.DEFINE_integer('batch_size', 32, "batch_size")
tf.app.flags.DEFINE_integer('epoch', 10, "epoch")
tf.app.flags.DEFINE_string('train_name', "train", "train dataset file name")
tf.app.flags.DEFINE_string('validation_name', "validation", "validation dataset file name")
tf.app.flags.DEFINE_boolean('train', True, "trains")
tf.app.flags.DEFINE_boolean('eval', True, "eval")
tf.app.flags.DEFINE_float('train_fraction', 0.9, "train_fraction")
tf.app.flags.DEFINE_integer('num_channel', 3, "num channel")
tf.app.flags.DEFINE_integer('num_dataset_parallel', 4, "num_dataset_parallel")
tf.app.flags.DEFINE_string('log_dir',
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                           "save dir")
tf.app.flags.DEFINE_integer('vis_epoch', 1, "vis_epoch")
tf.app.flags.DEFINE_integer('num_vis_steps', 10, "num_vis_steps")
tf.app.flags.DEFINE_integer('num_cam', 10, "num_cam")
tf.app.flags.DEFINE_integer('num_save_interval', 1, "num_save_interval")
tf.app.flags.DEFINE_integer('summary_interval', 10, "summary_interval")
tf.app.flags.DEFINE_integer('summary_images', 32, "summary_images")
tf.app.flags.DEFINE_integer('use_shuffle', True, "use shuffle")
tf.app.flags.DEFINE_integer('shuffle_buffer', 1000, "shuffle_buffer")
tf.app.flags.DEFINE_boolean('use_predict_of_test_for_embed_vis', True, "use_predict_of_test_for_embed_vis")
# tf.app.flags.DEFINE_string('restore_model_path', "checkpoint/model_epoch_9.ckpt", "model path to restore")
tf.app.flags.DEFINE_string('restore_model_path', None, "model path to restore")
tf.app.flags.DEFINE_string('preprocessing_name', None, "preprocessing name")
tf.app.flags.DEFINE_boolean('use_regularizer', False, "use_regularizer")

tf.app.flags.DEFINE_integer('input_size', None, "input_size")

tf.app.flags.DEFINE_string('trainer', None, "trainer file name in the trainer directory")

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd"  "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_boolean('polynomial_learning_rate_cycle', True, "Whether to cycle of polynomial learning rate")

if not FLAGS.trainer:
    trainer_name = "base_trainer"
else:
    trainer_name = FLAGS.trainer
trainer_path = 'trainer.%s' % trainer_name
train_func = util.get_func(trainer_path, "train")
if train_func:
    init.init(FLAGS)
    train_func(FLAGS)
