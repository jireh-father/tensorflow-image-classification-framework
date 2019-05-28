import tensorflow as tf
import os
from datetime import datetime
from core import util, tfrecorder_builder
import json
from factory import model_factory
from visualizer.grad_cam_plus_plus import GradCamPlusPlus
from slim.preprocessing import preprocessing_factory
import numpy as np


tf.app.flags.DEFINE_string('dataset_dir', "./mnist", "dataset_dir")
tf.app.flags.DEFINE_string('label_path', "./label.json", "label_path")
tf.app.flags.DEFINE_string('model_name', "alexnet_v2", "model name")
tf.app.flags.DEFINE_string('loss_file', "softmax_cross_entropy", "loss_file")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch_size")
tf.app.flags.DEFINE_integer('num_channel', 3, "num channel")

tf.app.flags.DEFINE_string('log_dir',
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), "inference"),
                           "save dir")

tf.app.flags.DEFINE_string('restore_model_path', None, "model path to restore")
tf.app.flags.DEFINE_integer('buffer_size', 10000, "buffer_size")
tf.app.flags.DEFINE_string('preprocessing_name', None, "preprocessing name")
tf.app.flags.DEFINE_integer('input_size', None, "input_size")
tf.app.flags.DEFINE_integer('num_input_summary', 32, "num_input_summary")
tf.app.flags.DEFINE_bool('is_inference', True, "is_inference")
tf.app.flags.DEFINE_boolean('use_weighted_loss', False, "use_weighted_loss")
tf.app.flags.DEFINE_boolean('train', False, "train")
tf.app.flags.DEFINE_boolean('use_regularizer', False, "use_regularizer")
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, "num_preprocessing_threads")

tf.app.flags.DEFINE_integer('top_k', 5, "top_k")



FLAGS = tf.app.flags.FLAGS
FLAGS.dataset_name

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

start_time = datetime.now().strftime('%Y%m%d%H%M%S')
log_dir = args.log_dir + "/" + start_time

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

filenames = tfrecorder_builder.get_filenames(args.dataset_dir)
nums_samples = len(filenames)


label_map = json.load(open(args.label_path))

model = model_factory.build_model(None, args)

if model is None:
    raise Exception("There is no model name.(%s)" % args.model_name)


inputs, labels, logits, end_points, is_training, global_step, default_last_conv_name = model

cam = GradCamPlusPlus(logits, end_points[default_last_conv_name],
                           inputs, is_training)

def parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=args.num_channel)

    if args.preprocessing_name and args.preprocessing_name in preprocessing_factory.preprocessing_fn_map:
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(args.preprocessing_name,
                                                                         is_training=False)
        image_decoded = image_preprocessing_fn(image_decoded, args.input_size, args.input_size)
    else:
        if not args.preprocessing_name:
            args.preprocessing_name = "base_preprocessing"
        preprocessing_f = util.get_attr('preprocessing.%s' % args.preprocessing_name, "preprocessing_inference")
        if not preprocessing_f:
            preprocessing_f = util.get_attr('preprocessing.base_preprocessing', "preprocessing_inference")
        image_decoded = preprocessing_f(image_decoded, tf.convert_to_tensor(args.input_size),
                                tf.convert_to_tensor(args.input_size), args)
    

    return image_decoded


dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(parse_function, num_parallel_calls=args.num_preprocessing_threads)
dataset = dataset.batch(args.batch_size)
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

query_vecs = np.zeros((len(queries), int(args.embedding_size)))

steps = int(nums_samples / args.batch_size)
if nums_samples % args.batch_size > 0:
    steps += 1
for i in range(steps):
    batch_imgs = sess.run(next_batch)
    feed_dict = {inputs: batch_imgs, is_training: False}
    tmp_logits = sess.run(logits, feed_dict=feed_dict)
    top_k_args = tmp_logits.argsort()[-args.top_k:][::-1]
    for j, tmp_qe in enumerate(tmp_query_vecs):
        query_vecs[i * args.eval_batch_size + j] = tmp_qe

