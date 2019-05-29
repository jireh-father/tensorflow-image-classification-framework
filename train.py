import tensorflow as tf
from core import trainer
import os, json
import traceback
from datetime import datetime

tf.app.flags.DEFINE_string('config', "config.json", "config file path")
tf.app.flags.DEFINE_string('dataset_name', "mnist", "dataset name")
tf.app.flags.DEFINE_string('dataset_dir', "./mnist", "dataset_dir")
tf.app.flags.DEFINE_string('model_name', "alexnet_v2", "model name")
tf.app.flags.DEFINE_string('loss_file', "softmax_cross_entropy", "loss_file")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch_size")
tf.app.flags.DEFINE_integer('validation_batch_size', 128, "validation_batch_size")
tf.app.flags.DEFINE_integer('epoch', 1, "epoch")
tf.app.flags.DEFINE_integer('total_steps', None, "total_steps")
tf.app.flags.DEFINE_integer('steps_per_epoch', None, "steps_per_epoch")
tf.app.flags.DEFINE_string('train_name', "train", "train dataset file name")
tf.app.flags.DEFINE_string('validation_name', "validation", "validation dataset file name")
tf.app.flags.DEFINE_boolean('train', True, "train")
tf.app.flags.DEFINE_boolean('validation', True, "validation")
tf.app.flags.DEFINE_boolean('use_summary', True, "use_summary")
tf.app.flags.DEFINE_boolean('use_trainable_variable_summary', False, "use_trainable_variable_summary")
tf.app.flags.DEFINE_float('train_fraction', 0.9, "train_fraction")
tf.app.flags.DEFINE_integer('num_channel', 1, "num channel")
tf.app.flags.DEFINE_integer('num_dataset_split', 4, "num_dataset_split")
tf.app.flags.DEFINE_integer('num_parallel_calls', 4, "num_parallel_calls")
tf.app.flags.DEFINE_integer('num_parallel_readers', 4, "num_parallel_readers")
tf.app.flags.DEFINE_boolean('cache_data', False, "cache_data")

tf.app.flags.DEFINE_string('log_dir',
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), "results"),
                           "save dir")

tf.app.flags.DEFINE_integer('keep_checkpoint_max', 5, "keep_checkpoint_max")
# tf.app.flags.DEFINE_string('restore_model_path', "checkpoint/model_epoch_9.ckpt", "model path to restore")
tf.app.flags.DEFINE_string('restore_model_path', None, "model path to restore")
tf.app.flags.DEFINE_boolean('use_weighted_loss', False, "use_weighted_loss")
tf.app.flags.DEFINE_boolean('use_train_cam', False, "use_train_cam")
tf.app.flags.DEFINE_boolean('use_validation_cam', False, "use_validation_cam")
tf.app.flags.DEFINE_boolean('use_bounding_box_visualization', False, "use_bounding_box_visualization")
tf.app.flags.DEFINE_integer('num_input_summary', 10, "num input summary")
tf.app.flags.DEFINE_integer('num_train_cam_epoch', 10, "num_train_cam_epoch")
tf.app.flags.DEFINE_integer('num_validation_cam_epoch', 10, "num_validation_cam_epoch")
tf.app.flags.DEFINE_boolean('use_train_embed_visualization', False, "use_train_embed_visualization")
tf.app.flags.DEFINE_boolean('use_validation_embed_visualization', False, "use_validation_embed_visualization")
tf.app.flags.DEFINE_integer('train_embed_visualization_interval', 1, "train_embed_visualization_interval epoch")
tf.app.flags.DEFINE_integer('validation_embed_visualization_interval', 1,
                            "validation_embed_visualization_interval epoch")
tf.app.flags.DEFINE_integer('num_train_embed_epoch', 200, "num_train_embed_epoch")
tf.app.flags.DEFINE_integer('num_validation_embed_epoch', 200, "num_validation_embed_epoch")
tf.app.flags.DEFINE_integer('save_interval', 1, "num_save_interval")
tf.app.flags.DEFINE_integer('summary_interval', 10, "summary_interval")
tf.app.flags.DEFINE_integer('summary_images', 32, "summary_images")
tf.app.flags.DEFINE_boolean('remove_original_images', False, "remove_original_images")
tf.app.flags.DEFINE_boolean('use_train_shuffle', True, "use shuffle")
tf.app.flags.DEFINE_integer('buffer_size', 10000, "buffer_size")
tf.app.flags.DEFINE_boolean('use_prediction_for_embed_visualization', False, "use_prediction_for_embed_visualization")
tf.app.flags.DEFINE_string('preprocessing_name', None, "preprocessing name")
tf.app.flags.DEFINE_boolean('use_regularizer', True, "use_regularizer")
tf.app.flags.DEFINE_integer('input_size', None, "input_size")
tf.app.flags.DEFINE_boolean('is_inference', False, "is_inference")
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
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_boolean('polynomial_learning_rate_cycle', True, "Whether to cycle of polynomial learning rate")
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                 'Comma-separated list of scopes of variables to exclude when restoring '
                 'from a checkpoint.')

tf.app.flags.DEFINE_string('trainable_scopes', None, 'Comma-separated list of scopes to filter the set of variables to train.'
                                           'By default, None would train all the variables.')

FLAGS = tf.app.flags.FLAGS
FLAGS.dataset_name


def begin_trainer(flags):
    trainer.main(flags)


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def make_train_key(base_dir, flags):
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    return "%s_%s_%s_%s_batchsize_%d_epoch_%d_%s" % (
        base_dir, flags.dataset_name, flags.model_name, flags.optimizer, flags.batch_size, flags.epoch, now)


if hasattr(FLAGS, "mark_as_parsed"):
    FLAGS.mark_as_parsed()

args = {}
try:
    iterator = iter(FLAGS)
    for key in iterator:
        args[key] = getattr(FLAGS, key)
except TypeError:
    for key in FLAGS.__dict__["__flags"]:
        args[key] = FLAGS.__dict__["__flags"][key]

schedule_json = None
if os.path.isfile(FLAGS.config):
    schedule_json = json.load(open(FLAGS.config))
start_time = datetime.now().strftime('%Y%m%d%H%M%S')
if not schedule_json:
    if not os.path.isabs(FLAGS.log_dir):
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.log_dir)
        args["log_dir"] = log_dir
    args["log_dir"] = args["log_dir"] + "_" + start_time
    begin_trainer(Dict2Obj(args))
else:
    backup = {}
    for config in schedule_json:
        for key in config:
            backup[key] = args[key]
            args[key] = config[key]
        backup["log_dir"] = FLAGS.log_dir
        #base_dir = FLAGS.log_dir
        base_dir = args["log_dir"]
        #if not os.path.isabs(FLAGS.log_dir):
        #    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.log_dir)
        base_dir = os.path.join(base_dir, start_time)
        args_obj = Dict2Obj(args)
        train_key = make_train_key(base_dir, args_obj)
        if not os.path.isdir(train_key):
            os.makedirs(train_key)
        args["log_dir"] = train_key
        f = open(os.path.join(train_key, "train_parameters.txt"), "w")
        f.write("%s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("%s\n" % json.dumps(args))
        for key in args:
            f.write("%s:%s\n" % (key, str(args[key])))
        f.close()
        args_obj.log_dir = train_key
        begin_trainer(args_obj)
        for key in backup:
            args[key] = backup[key]
        args["log_dir"] = backup["log_dir"]
        backup = {}
