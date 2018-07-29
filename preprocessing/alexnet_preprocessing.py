import tensorflow as tf


def train(image, w, h, config):
    print("in preprocessing")
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    tf.summary.image('original_image', tf.expand_dims(image, 0), max_outputs=10)
    image = tf.random_crop(image, [h, w, config.num_channel])
    tf.summary.image('random_crop', tf.expand_dims(image, 0), max_outputs=10)
    image = tf.image.random_flip_left_right(image)
    tf.summary.image('random_flip_left_right', tf.expand_dims(image, 0), max_outputs=10)
    if config.num_channel == 3:
        image = tf.image.random_hue(image, 0.3)
        tf.summary.image('random_hue', tf.expand_dims(image, 0), max_outputs=10)
    image = tf.image.per_image_standardization(image)
    return image


def validation(image, w, h, config):
    image = tf.image.resize_image_with_crop_or_pad(image, h, w)
    image = tf.image.random_flip_left_right(image)
    if config.num_channel == 3:
        image = tf.image.random_hue(image, 0.3)
    image = tf.image.per_image_standardization(image)
    return image


def preprocessing(image, w, h, config, is_training):
    if is_training:
        return train(image, w, h, config)
    else:
        return validation(image, w, h, config)
