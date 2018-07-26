from core import util
import os
import glob
import tensorflow as tf
from slim.preprocessing import preprocessing_factory


class Dataset(object):
    def __init__(self, sess, batch_size, shuffle, is_training, config, input_size):
        self.sess = sess
        self.config = config
        self.input_size = input_size

        if is_training:
            conf_key = "train_name"
            dataset_map = self.train_dataset_map
        else:
            conf_key = "validation_name"
            dataset_map = self.test_dataset_map

        files = tf.data.Dataset.list_files(
            os.path.join(config.dataset_dir, "%s_%s*tfrecord" % (config.dataset_name, getattr(config, conf_key))))

        if hasattr(tf.contrib.data, "parallel_interleave"):
            ds = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=config.num_parallel_readers))
        else:
            ds = files.interleave(tf.data.TFRecordDataset, cycle_length=config.num_parallel_readers)

        if config.cache_data:
            ds = ds.take(1).cache().repeat()

        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        ds = tf.data.Dataset.zip((ds, counter))
        ds = ds.prefetch(buffer_size=batch_size)
        # ds = ds.repeat()
        if shuffle:
            ds = ds.shuffle(buffer_size=config.buffer_size)

        if hasattr(tf.contrib.data, "map_and_batch"):
            ds = ds.apply(tf.contrib.data.map_and_batch(map_func=dataset_map, batch_size=batch_size))
        else:
            ds = ds.map(map_func=dataset_map, num_parallel_calls=config.num_parallel_calls)
            ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=batch_size)

        self.iterator = ds.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def get_next_batch(self):
        return self.sess.run(self.next_batch)

    def init(self, input_size):
        self.sess.run(self.iterator.initializer,
                      feed_dict={self.input_size: input_size})

    def pre_process(self, example_proto, is_training):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_image(parsed_features["image/encoded"], self.config.num_channel)
        preprocessed = False
        if self.config.preprocessing_name and self.config.preprocessing_name in preprocessing_factory.preprocessing_fn_map:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(self.config.preprocessing_name,
                                                                             is_training=is_training)

            image = tf.clip_by_value(
                image_preprocessing_fn(image, tf.convert_to_tensor(self.config.input_size),
                                       tf.convert_to_tensor(self.config.input_size)), -1, 1.0)
            preprocessed = True
        elif self.config.preprocessing_name:
            glob.glob("./preprocessing/*.py")
            preprocessing_files = os.listdir("./preprocessing")
            preprocessing_f = util.get_attr('preprocessing.%s' % preprocessing_files, "preprocessing")

            if preprocessing_f:
                image = preprocessing_f(image, tf.convert_to_tensor(self.config.input_size),
                                        tf.convert_to_tensor(self.config.input_size))
                preprocessed = True

        if not preprocessed:
            image = tf.image.resize_images(image, [tf.convert_to_tensor(self.config.input_size),
                                                   tf.convert_to_tensor(self.config.input_size)])
            image = tf.clip_by_value(tf.image.per_image_standardization(image), -1., 1.0)

        if len(parsed_features["image/class/label"].get_shape()) == 0:
            label = tf.one_hot(parsed_features["image/class/label"], self.config.num_class)
        else:
            label = parsed_features["image/class/label"]

        return image, label

    def train_dataset_map(self, example_proto, batch_position=0):
        return self.pre_process(example_proto, True)

    def test_dataset_map(self, example_proto, batch_position=0):
        return self.pre_process(example_proto, False)
