import glob
import os
import tensorflow as tf
from defined.preprocessing import preprocessing_factory


class Dataset(object):
    def __init__(self, sess, batch_size, shuffle, is_training, config):
        self.sess = sess
        self.config = config
        if is_training:
            conf_key = "train_name"
        else:
            conf_key = "validation_name"
        filenames = glob.glob(
            os.path.join(config.dataset_dir, "%s_%s*tfrecord" % (config.dataset_name, getattr(config, conf_key))))

        if shuffle:
            self.iterator = tf.data.TFRecordDataset(filenames).map(self.train_dataset_map,
                                                                   config.num_dataset_parallel).shuffle(
                buffer_size=config.buffer_size).batch(batch_size).make_initializable_iterator()
        else:
            self.iterator = tf.data.TFRecordDataset(filenames).map(self.test_dataset_map,
                                                                   config.num_dataset_parallel).batch(
                batch_size).make_initializable_iterator()

        self.next_batch = self.iterator.get_next()

    def get_next_batch(self):
        return self.sess.run(self.next_batch)

    def init(self):
        self.sess.run(self.iterator.initializer)

    def pre_process(self, example_proto, is_training):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        if self.config.preprocessing_name:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(self.config.preprocessing_name,
                                                                             is_training=is_training)
            image = tf.image.decode_image(parsed_features["image/encoded"], self.config.num_channel)
            image = tf.clip_by_value(
                image_preprocessing_fn(image, self.config.input_size, self.config.input_size), -1, 1.0)
        else:
            image = tf.clip_by_value(tf.image.per_image_standardization(
                tf.image.resize_images(tf.image.decode_jpeg(parsed_features["image/encoded"], self.config.num_channel),
                                       [self.config.input_size, self.config.input_size])), -1., 1.0)

        if len(parsed_features["image/class/label"].get_shape()) == 0:
            label = tf.one_hot(parsed_features["image/class/label"], self.config.num_class)
        else:
            label = parsed_features["image/class/label"]

        return image, label

    def train_dataset_map(self, example_proto):
        return self.pre_process(example_proto, True)

    def test_dataset_map(self, example_proto):
        return self.pre_process(example_proto, False)
