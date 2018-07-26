import tensorflow as tf


def preprocessing(image, w, h, is_training):
    image = tf.clip_by_value(tf.image.per_image_standardization(tf.image.resize_images(image, [w, h])), -1., 1.0)
    return image
