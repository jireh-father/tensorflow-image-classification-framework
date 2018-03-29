import tensorflow as tf


def dataset_map(example_proto, training):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    if conf.preprocessing_name:
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(conf.preprocessing_name,
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
