import tensorflow as tf


def preprocessing(image, w, h, config, is_training):
    image = tf.clip_by_value(tf.image.per_image_standardization(tf.image.resize_images(image, [w, h])), -1., 1.0)
    print("test", image)
    return image
