import tensorflow as tf


def preprocessing(image, w, h, config, is_training):
    image = tf.image.resize_image_with_crop_or_pad(image, h, w)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    return image
