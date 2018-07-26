import tensorflow as tf


def train(image, w, h):
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    image = tf.extract_image_patches(image, w)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.per_image_standardization(image)
    return image


def validation(image, w, h):
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    image = tf.extract_image_patches(image, w)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.per_image_standardization(image)
    return image


def preprocessing(image, w, h, is_training):
    if is_training:
        return train(image, w, h)
    else:
        return validation(image, w, h)
