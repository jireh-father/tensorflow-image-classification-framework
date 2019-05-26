import tensorflow as tf


def preprocessing(image, w, h, config, is_training):
    # image = tf.clip_by_value(tf.image.per_image_standardization(tf.image.resize_images(image, [w, h])), -1., 1.0)
    tf.dtypes.cast(config.image_w / 2, tf.int64)
    tf.dtypes.cast(config.image_h / 2, tf.int64)

    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(image, 0)
    # image = tf.image.resize_image_with_pad(image, w, h)
    image = tf.image.resize_bilinear(image, [w, h], align_corners=False)
    image = tf.squeeze(image, [0])

    image = tf.divide(image, 255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # top : 1/4, bot : 1/6, side : 1/5
    return image
