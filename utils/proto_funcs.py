import tensorflow as tf


def isinstance_dict(x):
    return isinstance(x, dict)


def isinstance_list(x):
    return isinstance(x, list)


def isinstance_tuple(x):
    return isinstance(x, tuple)


def isinstance_bytes(x):
    return isinstance(x, bytes)


def isinstance_int(x):
    return isinstance(x, int)


def isinstance_float(x):
    return isinstance(x, float)


def isinstance_bytes_list(x):
    return isinstance(x, tf.train.BytesList)


def isinstance_int64_list(x):
    return isinstance(x, tf.train.Int64List)


def isinstance_floats_list(x):
    return isinstance(x, tf.train.FloatList)


def isinstance_feature(x):
    return isinstance(x, tf.train.Feature)


def isinstance_type_a_or_b(x, a, b):
    return isinstance(x, (a, b))


def isinstance_type_bytes_or_bytes_list(x):
    return isinstance_type_a_or_b(x, bytes, tf.train.BytesList)


def isinstance_type_int_or_int64_list(x):
    return isinstance_type_a_or_b(x, int, tf.train.Int64List)


def isinstance_type_float_or_float_list(x):
    return isinstance_type_a_or_b(x, float, tf.train.FloatList)


class _List(object):

    @staticmethod
    def _bytes_list(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_bytes(value):
            value = [value]
        else:
            raise TypeError('%s is not of bytes Type' % value)
        return tf.train.BytesList(value=value)

    @staticmethod
    def _int64_list(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_int(value):
            value = [value]
        else:
            raise TypeError('%s is not of int Type' % value)
        return tf.train.Int64List(value=value)

    @staticmethod
    def _float_list(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_float(value):
            value = [value]
        else:
            raise TypeError('%s is not of float Type' % value)
        return tf.train.FloatList(value=value)

    @classmethod
    def bytes_list(cls, value):
        return cls._bytes_list(value)

    @classmethod
    def int64_list(cls, value):
        return cls._int64_list(value)

    @classmethod
    def float_list(cls, value):
        return cls._float_list(value)


class _Feature(object):

    @staticmethod
    def _bytes_feature(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_bytes(value):
            value = [value]
        else:
            raise TypeError('%s is not of bytes Type' % value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _int64_feature(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_int(value):
            value = [value]
        else:
            raise TypeError('%s is not of int Type' % value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        if isinstance_list(value):
            value = value
        elif isinstance_tuple(value):
            value = list(value)
        elif isinstance_float(value):
            value = [value]
        else:
            raise TypeError('%s is not of float Type' % value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_list_feature(value):
        return tf.train.Feature(bytes_list=value)

    @staticmethod
    def _int64_list_feature(value):
        return tf.train.Feature(int64_list=value)

    @staticmethod
    def _float_list_feature(value):
        return tf.train.Feature(float_list=value)

    @classmethod
    def bytes_feature(cls, value):
        return cls._bytes_feature(value=value)

    @classmethod
    def int64_feature(cls, value):
        return cls._int64_feature(value=value)

    @classmethod
    def float_feature(cls, value):
        return cls._float_feature(value=value)

    @classmethod
    def bytes_list_feature(cls, value):
        return cls._bytes_list_feature(value=value)

    @classmethod
    def int64_list_feature(cls, value):
        return cls._int64_list_feature(value=value)

    @classmethod
    def float_list_feature(cls, value):
        return cls._float_list_feature(value=value)


def apply_to_elem(func, e):
    return func(e)


def apply_to_dict_elem(func, d):
    return {x: func(d[x]) for x in d.keys()}


def apply_to_list_elem(func, l):
    return [func(x) for x in l]


def apply_to_tuple_elem(func, t):
    return [func(x) for x in t]


def apply_to_whole(func, data):
    return apply_to_elem(func, data)


def each_dict_elem(isinstance_func, d):
    return all(apply_to_dict_elem(isinstance_func, d).values())


def each_list_elem(isinstance_func, l):
    return all(apply_to_list_elem(isinstance_func, l))


def each_tuple_elem(isinstance_func, t):
    return all(apply_to_tuple_elem(isinstance_func, t))


def each_dict_elem_type_a_b(d, a, b):
    return all([isinstance_type_a_or_b(d[x], a, b) for x in d])


def each_tuple_or_list_elem_type_a_b(t_or_l, a, b):
    return all([isinstance_type_a_or_b(x, a, b) for x in t_or_l])
