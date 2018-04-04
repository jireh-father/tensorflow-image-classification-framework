import importlib
import glob
import os
import tensorflow as tf


def get_attr(file_path, func_name):
    try:
        module = __import__(file_path + "", globals(), locals(), [func_name])
        return getattr(module, func_name)

    except ImportError:
        return None
    except AttributeError:
        return None


def count_label(label_file):
    with open(label_file) as f:
        for i, l in enumerate(f):
            pass
        return i + 1
    return None


def get_tfrecord_filenames(dataset_name, dataset_dir, train_or_validation='train'):
    return glob.glob(os.path.join(dataset_dir, "%s_%s*.tfrecord" % (dataset_name, train_or_validation)))


def count_records(tfrecord_filenames):
    c = 0
    for fn in tfrecord_filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c
