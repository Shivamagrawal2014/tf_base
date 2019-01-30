import tensorflow as tf


def cast_to(image, dtype):
    return tf.cast(image, dtype=dtype)


def random_brightness(tensor):
    return tf.image.random_brightness(tensor, )

def random_contrast():
