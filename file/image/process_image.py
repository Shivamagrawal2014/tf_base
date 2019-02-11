import tensorflow as tf
IMAGE_RESIZE = None


def saturate(image):
    saturation = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return saturation


def random_hue(image):
    hue = tf.image.random_hue(image, max_delta=0.2)
    return hue


def random_contrast(image):
    constrast = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return constrast


def cast_image_dtype(image, dtype=tf.float32):
    type_casted = tf.cast(image, dtype=dtype, name='cast_image_dtype')
    return type_casted


def random_brightness(image, max_delta=0.125):
    bright = tf.image.random_brightness(image, max_delta=max_delta)
    return bright


def random_flip_left_right(image):
    flip = tf.image.random_flip_left_right(image)
    return flip


def random_up_down(image):
    up_down = tf.image.random_flip_up_down(image)
    return up_down


def transpose(image):
    trans = tf.image.transpose_image(image)
    return trans


def resize_image_with_crop_or_pad(image, height=244, width=244):
    cropped_or_padded = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return cropped_or_padded


class DistortImage(object):

    @classmethod
    def color(cls, image):
        with tf.name_scope(name='distort_color'):
            image = cast_image_dtype(image)
            with tf.name_scope('random_condition'):
                rand_int1 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int2 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int3 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int4 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int5 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                five = tf.constant(5, dtype=tf.int32)

            image = tf.cond(rand_int1 > five, lambda: random_brightness(image, max_delta=32.0 / 255.0),
                            lambda: image, name='random_brightness')
            image = tf.cond(rand_int2 > five, lambda: saturate(image), lambda: image, name='random_saturate')
            image = tf.cond(rand_int3 > five, lambda: random_hue(image), lambda: image, name='random_hue')
            image = tf.cond(rand_int4 > five, lambda: random_contrast(image), lambda: image,
                            name='random_contrast')
            image = tf.cond(rand_int5 > five, lambda: tf.clip_by_value(image, 0, 1, name='clip_by_value'),
                            lambda: image, name='clip_by_value')

        return image

    @classmethod
    def shape(cls, image):
        with tf.name_scope(name='distort_shape'):
            image = cast_image_dtype(image, dtype=tf.uint8)
            with tf.name_scope('random_condition'):
                rand_int1 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int2 = tf.random_uniform([], 1, 10, dtype=tf.int32)
                rand_int3 = tf.random_uniform([], 1, 10, dtype=tf.int32)

                five = tf.constant(5, dtype=tf.int32)
            image = tf.cond(rand_int1 > five, lambda: random_flip_left_right(image), lambda: image,
                            name='random_left_right')
            image = tf.cond(rand_int2 > five, lambda: random_up_down(image), lambda: image, name='random_up_down')
            if IMAGE_RESIZE is not None:
                if not tf.shape(image).get_shape().as_list()[1:3] != IMAGE_RESIZE:
                    image = resize_image_with_crop_or_pad(image, height=IMAGE_RESIZE[0], width=IMAGE_RESIZE[1])
            image = tf.cond(rand_int3 > five, lambda: transpose(image), lambda: image, name='random_transpose')
        return image


def distort_image(image):
    distort = DistortImage
    # only_recolored_image = distort.color(image)
    with tf.name_scope(name='distort_image'):
        only_reshaped_image = distort.shape(image)
        reshaped_then_recolored_image = distort.color(only_reshaped_image)
        # recolored_then_reshaped_image = distort.shape(only_recolored_image)
    return (only_reshaped_image,), reshaped_then_recolored_image


def encode_label_batch(label_batch):
    sess = tf.get_default_session()
    # print(classes)
    class_mapping = tf.convert_to_tensor(classes)
    class_depth = class_mapping.shape[0]
    mapping_strings = class_mapping
    labels = label_batch
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
    ids = table.lookup(labels)
    sess.run(table.init)
    with tf.name_scope('one_hot_encoding'):
        labels = tf.one_hot(ids, depth=class_depth)
    return labels


class ConvolutionalBatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.

    Use:
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
      update_assignments = bn.get_assigner()
      x = bn.normalize(y, train=training?)
      (the output x will be batch-normalized).
    """

    def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):

        self.mean = tf.get_variable(
            name='mean', shape=[depth], initializer=tf.constant_initializer(0.0),  trainable=False)
        self.variance = tf.get_variable(
            name='variance', shape=[depth], initializer=tf.constant_initializer(1.0), trainable=False)
        self.beta = tf.get_variable(
            name='beta', shape=[depth], initializer=tf.constant_initializer(0.0),  trainable=False)
        self.gamma = tf.get_variable(
            name='gamma', shape=[depth], initializer=tf.constant_initializer(1.0), trainable=False)
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon
        self.scale_after_norm = scale_after_norm

    def get_assigner(self):
        """Returns an EWMA apply op that must be invoked after optimization."""
        return self.ewma_trainer.apply([self.mean, self.variance])

    def normalize(self, x, train=True):

        """Returns a batch-normalized version of x."""
        if train:
             mean, variance = tf.nn.moments(x, [0, 1, 2])
             assign_mean = self.mean.assign(mean)
             assign_variance = self.variance.assign(variance)
             with tf.control_dependencies([assign_mean, assign_variance]):
                 return tf.nn.batch_norm_with_global_normalization(
                     x, mean, variance, self.beta, self.gamma,
                     self.epsilon, self.scale_after_norm)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, local_beta, local_gamma,
                self.epsilon, self.scale_after_norm)
