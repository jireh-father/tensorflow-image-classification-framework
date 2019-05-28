import importlib
import glob
import os
import tensorflow as tf
import sys

def get_attr(file_path, func_name):
    try:
        module = __import__(file_path + "", globals(), locals(), [func_name])
        return getattr(module, func_name)

    except ImportError:
        return None
    except AttributeError:
        return None


def count_label(label_file):
    if sys.version_info[0] < 3:
        with open(label_file) as f:
            for i, l in enumerate(f):
                pass
            return i + 1
    else:
        with open(label_file, encoding="utf-8") as f:
            for i, l in enumerate(f):
                pass
            return i + 1



def get_tfrecord_filenames(dataset_name, dataset_dir, train_or_validation='train'):
    return glob.glob(os.path.join(dataset_dir, "%s_%s*.tfrecord" % (dataset_name, train_or_validation)))


def count_records(tfrecord_filenames):
    c = 0
    for fn in tfrecord_filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def count_each_classes(dataset_dir, dataset_name, train_name="train"):
    tfrecords_filenames = glob.glob(os.path.join(dataset_dir, "%s_%s*tfrecord" % (dataset_name, train_name)))
    labels = {}
    for tfrecords_filename in tfrecords_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = int(example.features.feature['image/class/label']
                        .int64_list
                        .value[0])
            if not label in labels:
                labels[label] = 0
            labels[label] += 1
    return labels


def count_each_class(dataset_dir, dataset_name, batch_size=128,
                     train_name="train"):
    def map_func(self, example_proto, batch_position=0):
        features = {
            "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        label = parsed_features["image/class/label"]

        return label

    files = tf.data.Dataset.list_files(os.path.join(dataset_dir, "%s_%s*tfrecord" % (dataset_name, train_name)))
    num_parallel_readers = len(glob.glob(os.path.join(dataset_dir, "%s_%s*tfrecord" % (dataset_name, train_name))))
    num_parallel_calls = num_parallel_readers
    if hasattr(tf.contrib.data, "parallel_interleave"):
        ds = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
    else:
        ds = files.interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers)

    counter = tf.data.Dataset.range(batch_size)
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)

    if hasattr(tf.contrib.data, "map_and_batch"):
        ds = ds.apply(tf.contrib.data.map_and_batch(map_func=map_func, batch_size=batch_size))
    else:
        ds = ds.map(map_func=map_func, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=batch_size)

    iterator = ds.make_one_shot_iterator()
    labels = iterator.get_next()
