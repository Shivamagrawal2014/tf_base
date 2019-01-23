import tensorflow as tf


class Variable(object):
    def __init__(self,
                 initializer,
                 shape,
                 name):
        self._initializer = initializer
        self._shape = shape
        self._name = name

    def __call__(self):
        return tf.get_variable(name=self._name,
                               initializer=self._initializer,
                               shape=self._shape)


class Weight(object):
    def __init__(self,
                 shape,
                 initializer=None,
                 name=None,
                 mean=0.0,
                 stddev=.02,
                 seed=0):

        self._name = name or self.__class__.__name__
        self._shape = shape
        self._initializer = initializer or tf.truncated_normal_initializer(mean=mean, stddev=stddev,
                                                                           seed=seed)
        self._W = Variable(initializer=self._initializer,
                           shape=self._shape, name=self._name)

    def __call__(self):
        w = self._W()
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        return w


class Bias(object):
    def __init__(self,
                 shape,
                 initializer=None,
                 name=None):

        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

        self._shape = shape
        self._initializer = initializer or tf.zeros_initializer()
        self._B = Variable(initializer=self._initializer,
                           shape=self._shape[-1], name=self._name)

    def __call__(self):
        b = self._B()
        tf.add_to_collection(tf.GraphKeys.BIASES, b)
        return b


def weight(shape,
           initializer=None,
           name=None,
           mean=0.0,
           stddev=0.02,
           seed=0
           ):
    return Weight(shape=shape, initializer=initializer, name=name, mean=mean, stddev=stddev, seed=seed)


def bias(shape,
         initializer=None,
         name=None):
    return Bias(shape=shape, initializer=initializer, name=name)


def get_weight_bias(shape,
                    weight_initializer=None,
                    bias_initializer=None,
                    weight_name='W',
                    bias_name='B'):
    w = weight(shape=shape, initializer=weight_initializer, name=weight_name)
    b = bias(shape=shape, initializer=bias_initializer, name=bias_name)
    return w, b


if __name__ == '__main__':
    w, b = get_weight_bias([300, 300, 3])
    print(w(), b())