import importlib
import glob
import os
import tensorflow as tf


def get_func(file_path, func_name):
    try:
        # print(":" + file_path + ":")
        # module = importlib.import_module("trainer.base_trainer")
        module = __import__(file_path + "", globals(), locals(), [func_name])
        print(dir(module))
        print(module)
        # module = importlib.import_module(file_path)
        return getattr(module, func_name)

    except ImportError:
        print("There is no the %s file." % file_path)
    except AttributeError:
        print("There is no the %s function in %s file." % (func_name, file_path))
    return None


def get_module(file_path):
    try:
        # print(":" + file_path + ":")
        # module = importlib.import_module("trainer.base_trainer")
        return __import__(file_path)
        # return importlib.import_module(file_path)

    except ImportError:
        print("There is no the %s file." % file_path)
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
