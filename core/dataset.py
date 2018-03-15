import glob
import os
import tensorflow as tf


class Dataset(object):
    def __init__(self, sess, shuffle, is_trainig, config):
        self.sess = sess
        if is_trainig:
            conf_key = "train_name"
        else:
            conf_key = "validation_name"
        filenames = glob.glob(
            os.path.join(config.dataset_dir, config.dataset_name + ("_%s*tfrecord" % config[conf_key])))

        if shuffle:
            self.iterator = tf.data.TFRecordDataset(filenames).map(train_dataset_map,
                                                                   config.num_dataset_parallel).shuffle(
                buffer_size=config.shuffle_buffer).batch(config.batch_size).make_initializable_iterator()
        else:
            self.iterator = tf.data.TFRecordDataset(filenames).map(test_dataset_map,
                                                                   config.num_dataset_parallel).batch(
                config.batch_size).make_initializable_iterator()

        self.next_batch = self.iterator.get_next()

    def get_next_batch(self):
        return self.sess.run(self.next_batch)

    def init(self):
        sess.run(self.iterator.initializer)

    def pre_process(example_proto, training):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        if config.preprocessing_name:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(config.preprocessing_name,
                                                                             is_training=training)
            image = tf.image.decode_image(parsed_features["image/encoded"], num_channel)
            image = tf.clip_by_value(
                image_preprocessing_fn(image, model_image_size, model_image_size), -1, 1.0)
        else:
            image = tf.clip_by_value(tf.image.per_image_standardization(
                tf.image.resize_images(tf.image.decode_jpeg(parsed_features["image/encoded"], num_channel),
                                       [model_image_size, model_image_size])), -1., 1.0)

        if len(parsed_features["image/class/label"].get_shape()) == 0:
            label = tf.one_hot(parsed_features["image/class/label"], num_classes)
        else:
            label = parsed_features["image/class/label"]

        return image, label

    def dataset_map(example_proto):
        return pre_process(example_proto, True)

    def test_dataset_map(example_proto):
        return pre_process(example_proto, False)
