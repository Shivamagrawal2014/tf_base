"""
Making the dictionary assignment and input controlled

"""
import numpy as np
import tensorflow as tf
from collections import namedtuple as nt
import collections

# Refer : https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564


class descriptorseq(object):
    class Seq(dict):
        pass

    class new(object):
        def __init__(self, value):
            if isinstance(value, descriptorseq.put):
                self._value = value
            else:
                raise AssertionError('Input format not supported')

        def new_get(self):
            return self._value.value

        def new_set(self, value):
            if isinstance(value, descriptorseq.put):
                self._value = value
            else:
                raise AssertionError('Can\'t set the value')

        newly = property(new_get, new_set, None, 'I am the new property')

    class SEQ(Seq):
        def __getitem__(self, key):
            if key in self and isinstance(descriptorseq.Seq.__getitem__(self, key), descriptorseq.new):

                return descriptorseq.Seq.__getitem__(self, key).newly
            else:
                raise AssertionError('Setting Value Not allowed')

        def __setitem__(self, key, value):
            if key in self and isinstance(descriptorseq.Seq.__getitem__(self, key), descriptorseq.new):
                # print('editing entry for %s' % key)
                descriptorseq.Seq.__getitem__(self, key).newly = value
            elif (key not in self) and isinstance(value, descriptorseq.new):
                # print('making new entry with key [%s] and value [%s]' % (key, str(value.new_get())))
                descriptorseq.Seq.__setitem__(self, key, value)
            else:
                raise AssertionError('Setting Value not Allowed')

    class put(nt('seq', 'value')):
        pass

    def __str__(self):
        return


class DescSeq(descriptorseq.SEQ):

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def newval(self, value):
        return descriptorseq.new(descriptorseq.put(value))

    def editval(self, value):
        return descriptorseq.put(value)


def desseq():
    return DescSeq()


def nested_non_nested_seperation(data):
    sanitizer_tuple = (int, float, bytes, str)
    # only keep iterables
    nests = [y for y in data if isinstance(y, collections.Iterable) and not isinstance(y, sanitizer_tuple)]
    # only keep non-iterables
    non_nests = [y for y in data if y not in nests]
    return nests, non_nests


def isinstance_elem(iterable, data_type):
    if isinstance(iterable, collections.Mapping):
        iterable = iterable.values()
    nested, non_nested = nested_non_nested_seperation(iterable)
    if non_nested == iterable:
        return all(map(lambda x: isinstance(x, data_type), non_nested))
    return all([all(map(lambda x: isinstance_elem(x, data_type), nested)),
                all(map(lambda x: isinstance(x, data_type), non_nested))])


# Converting the values into features
# _int64 is used for numeric values
def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# _float is used for floating point values


def _floats_feature(value): return tf.train.Feature(float_list=tf.train.FloatList(value=value))
# _bytes is used for string/char values


def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64list_feature(value): return tf.train.Feature(int64_list=value)


def _floatlist_feature(value): return tf.train.Feature(float_list=value)


def _bytelist_feature(value): return tf.train.Feature(bytes_list=value)


def _feature_featurelist(value): return tf.train.FeatureList(feature=value)


def cast_dict_val_to_int64_feature(d): return {name: _int64_feature(value) for name, value in d.items()}


def cast_dict_val_to_floats_feature(d): return {name: _floats_feature(value) for name, value in d.items()}


def cast_dict_val_to_bytes_feature(d): return {name: _bytes_feature(value) for name, value in d.items()}


def cast_dict_val_to_int64list_feature(d): return {name: _int64list_feature(value) for name, value in d.items()}


def cast_dict_val_to_floatslist_feature(d): return {name: _floatlist_feature(value) for name, value in d.items()}


def cast_dict_val_to_byteslist_feature(d): return {name: _bytelist_feature(value) for name, value in d.items()}


def cast_dict_val_to_feature_list(d): return {name: _feature_featurelist(value) for name, value in d.items()}


def listify_dictval(d): return {name: [value] for name, value in d.items()}


def newlyfy_dictval(d): return {name: value.newly for name, value in d.items()}


def isinstancedictval(d, t): return not (False in list(map(lambda x: isinstance(x, t), d.values())))


def isinstancedictvalindict(d,t): return not (False in [isinstance(j, t) for i in d.values() for j in i.values()])


def isinstancelistvalindict(d,t): return not (False in [isinstance(j, t) for i in d.values() for j in i])


class ToProto(object):
    class Feature(object):

        def __init__(self):
            self._intfeature = desseq()  # a descriptor sequence to hold the value for int
            self._floatfeature = desseq()  # a descriptor sequence to hold the value for float
            self._bytesfeature = desseq()  # a descriptor sequence to hold the value for bytes

        @property
        def intfeature(self):  # int descriptor for int feature
            return self._intfeature

        @property
        def floatfeature(self):  # float descriptor for float feature
            return self._floatfeature

        @property
        def bytefeature(self):  # bytes descriptor for bytes feature
            return self._bytesfeature

        @intfeature.getter
        def intfeature(self):
            return seq_to_norm_dict(self._intfeature)

        @intfeature.setter
        def intfeature(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                if isinstance(value, int):
                    self._intfeature[name] = self._intfeature.newval(_int64_feature([value]))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, int), value):
                        raise TypeError('Value entered must be of \'Int\' type')
                    else:
                        self._intfeature[name] = self._intfeature.newval(_int64_feature(value))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, int), value):
                        raise TypeError('Value entered must be of \'Int\' type')
                    else:
                        self._intfeature[name] = self._intfeature.newval(_int64_feature(list(value)))
                elif isinstance(value, tf.train.Int64List):
                    self._intfeature[name] = self._intfeature.newval(tf.train.Feature(int64_list=value))

                elif False in map(lambda x: isinstance(x, int), value):
                    raise TypeError('Value entered must be of \'Int\' type')

            elif isinstance(value, dict):
                for name, value in value.items():
                    name, value = name, value
                    print('name :', name, 'value :', value)
                    if isinstance(value, int):
                        self._intfeature[name] = self._intfeature.newval(_int64_feature([value]))

                    elif isinstance(value, (list, np.ndarray)):
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        if False in map(lambda x: isinstance(x, int), value):
                            if not (False in map(lambda x: isinstance(x, list), value)):
                                if not isinstance_elem(value, int):
                                    raise TypeError('Value entered must be of \'Int\' type')
                                else:
                                    self._intfeature[name] = self._intfeature.newval([_int64_feature(i) for i in value])
                        else:
                            self._intfeature[name] = self._intfeature.newval(_int64_feature(value))

                    elif isinstance(value, tuple):
                        if False in map(lambda x: isinstance(x, int), value):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeature[name] = self._intfeature.newval(_int64_feature(list(value)))

                    elif isinstance(value, descriptorseq.new):

                        if False in map(lambda x: isinstance(x, int), value.newly):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeature[name] = self._intfeature.newval(_int64_feature(value.newly))

                    elif isinstance(value, tf.train.Int64List):
                        self._intfeature[name] = self._intfeature.newval(tf.train.Feature(int64_list=value))

                    elif False in map(lambda x: isinstance(x, int), value):
                        raise TypeError('Value entered must be of \'Int\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, int), value.newly):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeature[name] = self._intfeature.newval(_int64_feature(value.newly))

        @floatfeature.getter
        def floatfeature(self):
            return seq_to_norm_dict(self._floatfeature)

        @floatfeature.setter
        def floatfeature(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                if isinstance(value, float):
                    self._floatfeature[name] = self._floatfeature.newval(_floats_feature([value]))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, float), value):
                        raise TypeError('Value entered must be of \'float\' type')
                    else:
                        self._floatfeature[name] = self._floatfeature.newval(_floats_feature(value))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, float), value):
                        raise TypeError('Value entered must be of \'float\' type')
                    else:
                        self._floatfeature[name] = self._floatfeature.newval(_floats_feature(list(value)))

                elif isinstance(value, tf.train.Int64List):
                    self._floatfeature[name] = self._floatfeature.newval(tf.train.Feature(float_list=value))

                elif False in map(lambda x: isinstance(x, float), value):
                    raise TypeError('Value entered must be of \'float\' type')

            elif isinstance(value, dict):
                for name, value in value.items():
                    name, value = name, value
                    print('name :', name, 'value :', value)
                    if isinstance(value, float):
                        self._floatfeature[name] = self._floatfeature.newval(_floats_feature([value]))

                    elif isinstance(value, (list, np.ndarray)):
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        if False in map(lambda x: isinstance(x, float), value):
                            if not (False in map(lambda x: isinstance(x, list), value)):
                                if not isinstance_elem(value, float):
                                    raise TypeError('Value entered must be of \'float\' type')
                                else:
                                    self._floatfeature[name] = \
                                        self._floatfeature.newval([_floats_feature(i) for i in value])
                            # raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeature[name] = self._floatfeature.newval(_floats_feature(value))

                    elif isinstance(value, tuple):
                        if False in map(lambda x: isinstance(x, float), value):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeature[name] = self._floatfeature.newval(_floats_feature(list(value)))

                    elif isinstance(value, descriptorseq.new):

                        if False in map(lambda x: isinstance(x, float), value.newly):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeature[name] = self._floatfeature.newval(_floats_feature(value.newly))

                    elif isinstance(value, tf.train.Int64List):
                        self._intfeature[name] = self._floatfeature.newval(tf.train.Feature(float_list=value))

                    elif False in map(lambda x: isinstance(x, float), value):
                        raise TypeError('Value entered must be of \'float\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, float), value.newly):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeature[name] = self._floatfeature.newval(_floats_feature(value.newly))

        @bytefeature.getter
        def bytefeature(self):
            return seq_to_norm_dict(self._bytesfeature)

        @bytefeature.setter
        def bytefeature(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                if isinstance(value, bytes):
                    self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature([value]))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, bytes), value):
                        raise TypeError('Value entered must be of \'bytes\' type')
                    else:
                        self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(value))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, bytes), value):
                        raise TypeError('Value entered must be of \'bytes\' type')
                    else:
                        self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(list(value)))
                elif isinstance(value, tf.train.Int64List):
                    self._bytesfeature[name] = self._bytesfeature.newval(tf.train.Feature(bytes_list=value))

                elif False in map(lambda x: isinstance(x, bytes), value):
                    raise TypeError('Value entered must be of \'bytes\' type')

            elif isinstance(value, dict):
                for name, value in value.items():
                    name, value = name, value
                    print('name :', name, 'value :', value)
                    if isinstance(value, bytes):
                        self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature([value]))

                    elif isinstance(value, (list, np.ndarray)):
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        if False in map(lambda x: isinstance(x, bytes), value):
                            if not (False in map(lambda x: isinstance(x, list), value)):
                                if not isinstance_elem(value, bytes):
                                    raise TypeError('Value entered must be of \'bytes\' type')
                                else:
                                    self._bytesfeature[name] = \
                                        self._bytesfeature.newval([_bytes_feature(i) for i in value])
                        else:
                            self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(value))

                    elif isinstance(value, tuple):
                        if False in map(lambda x: isinstance(x, bytes), value):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(list(value)))

                    elif isinstance(value, descriptorseq.new):

                        if False in map(lambda x: isinstance(x, bytes), value.newly):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(value.newly))

                    elif isinstance(value, tf.train.Int64List):
                        self._bytesfeature[name] = self._bytesfeature.newval(tf.train.Feature(bytes_list=value))

                    elif False in map(lambda x: isinstance(x, bytes), value):
                        raise TypeError('Value entered must be of \'bytes\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, bytes), value.newly):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeature[name] = self._bytesfeature.newval(_bytes_feature(value.newly))

        def __str__(self):
            return "int: {myint}, float: {myfloat}, byte: {mybyte}".format(
                myint=self.intfeature, myfloat=self.floatfeature, mybyte=self.bytefeature)

    class Features(object):

        def __init__(self, merge_all=None):
            self._intfeatures = desseq()    # a descriptor sequence to hold the value for int
            self._floatfeatures = desseq()  # a descriptor sequence to hold the value for float
            self._bytesfeatures = desseq()  # a descriptor sequence to hold the value for bytes
            self._mixedfeatures = desseq()   # a descriptor sequence to hold the value for mixed types
            self._pack_in_single_feature = merge_all

            if self._pack_in_single_feature is None:
                self._pack_in_single_feature = True

        @property
        def intfeatures(self):  # int descriptor for int feature
            return self._intfeatures

        @property
        def floatfeatures(self):  # float descriptor for float feature
            return self._floatfeatures

        @property
        def bytefeatures(self):  # bytes descriptor for bytes feature
            return self._bytesfeatures

        @property
        def mixedfeatures(self):
            return self._mixedfeatures

        @intfeatures.getter
        def intfeatures(self):
            return seq_to_norm_dict(self._intfeatures)

        @intfeatures.setter
        def intfeatures(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                print('inside intfeatures :')
                if isinstance(value, int):
                    self._intfeatures[name] = self._intfeatures.newval(
                        tf.train.Features(feature={name: _int64_feature([value])}))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, int), value):
                        raise TypeError('Value entered must be of \'Int\' type')
                    else:
                        self._intfeatures[name] = self._intfeatures.newval(
                            tf.train.Features(feature={name: _int64_feature(value)}))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, int), value):
                        raise TypeError('Value entered must be of \'Int\' type')
                    else:
                        self._intfeatures[name] = self._intfeatures.newval(
                            tf.train.Features(feature={name: _int64_feature(list(value))}))
                elif isinstance(value, tf.train.Int64List):
                    self._intfeatures[name] = self._intfeatures.newval(
                        tf.train.Features(feature={name: tf.train.Feature(int64_list=value)}))

                elif isinstance(value, tf.train.Feature):
                    self._intfeatures[name] = self._intfeatures.newval(tf.train.Features(feature={name: value}))

                elif False in map(lambda x: isinstance(x, int), value):
                    raise TypeError('Value entered must be of \'Int\' type')

            elif isinstance(value, dict):
                if self._pack_in_single_feature is True:

                    if isinstancedictval(value, int):
                        self._intfeatures['intfeatures'] = self._intfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_int64_feature(listify_dictval(value))))

                    elif isinstancedictval(value, list):
                        if not isinstance_elem(value, int):
                            raise TypeError('Value entered must be of \'int\' type')
                        else:
                            self._intfeatures['intfeatures'] = self._intfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_int64_feature(value)))

                    elif isinstancedictval(value, tuple):
                        if not isinstance_elem(value, int):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeatures['intfeatures'] = self._intfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_int64_feature(value)))

                    elif isinstancedictval(value, tf.train.Int64List):
                        self._intfeatures['intfeatures'] = self._intfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_int64list_feature(value)))

                    elif isinstancedictval(value, descriptorseq.new):
                        if not isinstance_elem(newlyfy_dictval(value), int):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeatures['intfeatures'] = self._intfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_int64_feature(newlyfy_dictval(value))))

                    elif isinstancedictval(value, tf.train.Feature):
                        self._intfeatures['intfeatures'] = self._intfeatures.newval(tf.train.Features(feature=value))

                if self._pack_in_single_feature is False:
                    for name, value in value.items():
                        name, value = name, value
                        print('name :', name, 'value :', value)
                        if isinstance(value, int):
                            self._intfeatures[name] = self._intfeatures.newval(
                                tf.train.Features(feature={name: _int64_feature([value])}))

                        elif isinstance(value, list):
                            if False in map(lambda x: isinstance(x, int), value):
                                raise TypeError('Value entered must be of \'Int\' type')
                            else:
                                self._intfeatures[name] = self._intfeatures.newval(
                                    tf.train.Features(feature={name: _int64_feature(value)}))

                        elif isinstance(value, tuple):
                            if False in map(lambda x: isinstance(x, int), value):
                                raise TypeError('Value entered must be of \'Int\' type')
                            else:
                                self._intfeatures[name] = self._intfeatures.newval(
                                    tf.train.Features(feature={name: _int64_feature(list(value))}))

                        elif isinstance(value, descriptorseq.new):

                            if False in map(lambda x: isinstance(x, int), value.newly):
                                raise TypeError('Value entered must be of \'Int\' type')
                            else:
                                self._intfeatures[name] = self._intfeatures.newval(
                                    tf.train.Features(feature={name: _int64_feature(value.newly)}))

                        elif isinstance(value, tf.train.Int64List):
                            self._intfeatures[name] = self._intfeatures.newval(
                                tf.train.Features(feature={name: tf.train.Feature(int64_list=value)}))

                        elif isinstance(value, tf.train.Feature):
                            self._intfeatures[name] = self._intfeatures.newval(tf.train.Features(feature={name: value}))

                        elif False in map(lambda x: isinstance(x, int), value):
                            raise TypeError('Value entered must be of \'Int\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, int), value.newly):
                            raise TypeError('Value entered must be of \'Int\' type')
                        else:
                            self._intfeatures[name] = self._intfeatures.newval(
                                tf.train.Features(feature={name: _int64_feature(value.newly)}))

        @floatfeatures.getter
        def floatfeatures(self):
            return seq_to_norm_dict(self._floatfeatures)

        @floatfeatures.setter
        def floatfeatures(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                if isinstance(value, float):
                    self._floatfeatures[name] = self._floatfeatures.newval(
                        tf.train.Features(feature={name: _floats_feature([value])}))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, float), value):
                        raise TypeError('Value entered must be of \'float\' type')
                    else:
                        self._floatfeatures[name] = self._floatfeatures.newval(
                            tf.train.Features(feature={name: _floats_feature(value)}))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, float), value):
                        raise TypeError('Value entered must be of \'float\' type')
                    else:
                        self._floatfeatures[name] = self._floatfeatures.newval(
                            tf.train.Features(feature={name: _floats_feature(list(value))}))

                elif isinstance(value, tf.train.FloatList):
                    self._floatfeatures[name] = self._floatfeatures.newval(
                        tf.train.Features(feature={name: tf.train.Feature(float_list=value)}))

                elif isinstance(value, tf.train.Feature):
                    self._floatfeatures[name] = self._floatfeatures.newval(tf.train.Features(feature={name: value}))

                elif False in map(lambda x: isinstance(x, float), value):
                    raise TypeError('Value entered must be of \'float\' type')

            elif isinstance(value, dict):
                if self._pack_in_single_feature is True:

                    if isinstancedictval(value, float):
                        self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_floats_feature(listify_dictval(value))))

                    elif isinstancedictval(value, list):
                        if not isinstance_elem(value, float):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_floats_feature(value)))

                    elif isinstancedictval(value, tuple):
                        if not isinstance_elem(value, float):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_floats_feature(value)))

                    elif isinstancedictval(value, tf.train.FloatList):
                        self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_floatslist_feature(value)))

                    elif isinstancedictval(value, descriptorseq.new):
                        if not isinstance_elem(newlyfy_dictval(value), float):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_floats_feature(newlyfy_dictval(value))))

                    elif isinstancedictval(value, tf.train.Feature):
                        self._floatfeatures['floatfeatures'] = self._floatfeatures.newval(
                            tf.train.Features(feature=value))

                elif self._pack_in_single_feature is False:
                    for name, value in value.items():
                        name, value = name, value
                        print('name :', name, 'value :', value)
                        if isinstance(value, float):
                            self._floatfeatures[name] = self._floatfeatures.newval(
                                tf.train.Features(feature={name: _floats_feature([value])}))

                        elif isinstance(value, list):
                            if False in map(lambda x: isinstance(x, float), value):
                                raise TypeError('Value entered must be of \'float\' type')
                            else:
                                self._floatfeatures[name] = self._floatfeatures.newval(
                                    tf.train.Features(feature={name: _floats_feature(value)}))

                        elif isinstance(value, tuple):
                            if False in map(lambda x: isinstance(x, float), value):
                                raise TypeError('Value entered must be of \'float\' type')
                            else:
                                self._floatfeatures[name] = self._floatfeatures.newval(
                                    tf.train.Features(feature={name: _floats_feature(list(value))}))

                        elif isinstance(value, descriptorseq.new):

                            if False in map(lambda x: isinstance(x, float), value.newly):
                                raise TypeError('Value entered must be of \'float\' type')
                            else:
                                self._floatfeatures[name] = self._floatfeatures.newval(
                                    tf.train.Features(feature={name: _floats_feature(value.newly)}))

                        elif isinstance(value, tf.train.FloatList):
                            self._floatfeatures[name] = self._floatfeatures.newval(
                                tf.train.Features(feature={name: tf.train.Feature(float_list=value)}))

                        elif isinstance(value, tf.train.Feature):
                            self._floatfeatures[name] = self._floatfeatures.newval(
                                tf.train.Features(feature={name: value}))

                        elif False in map(lambda x: isinstance(x, float), value):
                            raise TypeError('Value entered must be of \'float\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, float), value.newly):
                            raise TypeError('Value entered must be of \'float\' type')
                        else:
                            self._floatfeatures[name] = self._floatfeatures.newval(
                                tf.train.Features(feature={name: _floats_feature(value.newly)}))

        @bytefeatures.getter
        def bytefeatures(self):
            return seq_to_norm_dict(self._bytesfeatures)

        @bytefeatures.setter
        def bytefeatures(self, value):
            if isinstance(value, (tuple, list)):
                name, value = *value[0:1], *value[1:]
                print('name :', name, 'value :', value)
                if isinstance(value, bytes):
                    self._bytesfeatures[name] = self._bytesfeatures.newval(
                        tf.train.Features(feature={name: _bytes_feature([value])}))

                elif isinstance(value, (list)):
                    if False in map(lambda x: isinstance(x, bytes), value):
                        raise TypeError('Value entered must be of \'bytes\' type')
                    else:
                        self._bytesfeatures[name] = self._bytesfeatures.newval(
                            tf.train.Features(feature={name: _bytes_feature(value)}))

                elif isinstance(value, tuple):
                    if False in map(lambda x: isinstance(x, bytes), value):
                        raise TypeError('Value entered must be of \'bytes\' type')
                    else:
                        self._bytesfeatures[name] = self._bytesfeatures.newval(
                            tf.train.Features(feature={name: _bytes_feature(list(value))}))

                elif isinstance(value, tf.train.BytesList):
                    self._bytesfeatures[name] = self._bytesfeatures.newval(
                        tf.train.Features(feature={name: tf.train.Feature(bytes_list=value)}))

                elif isinstance(value, tf.train.Feature):
                    self._bytesfeatures[name] = self._bytesfeatures.newval(tf.train.Features(feature={name: value}))

                elif False in map(lambda x: isinstance(x, bytes), value):
                    raise TypeError('Value entered must be of \'bytes\' type')

            elif isinstance(value, dict):
                if self._pack_in_single_feature is True:

                    if isinstancedictval(value, bytes):
                        self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_bytes_feature(listify_dictval(value))))

                    elif isinstancedictval(value, list):
                        if not isinstance_elem(value, bytes):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_bytes_feature(value)))

                    elif isinstancedictval(value, tuple):
                        if not isinstance_elem(value, bytes):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_bytes_feature(value)))

                    elif isinstancedictval(value, tf.train.BytesList):
                        self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                            tf.train.Features(feature=cast_dict_val_to_byteslist_feature(value)))

                    elif isinstancedictval(value, descriptorseq.new):
                        if not isinstance_elem(newlyfy_dictval(value), bytes):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                                tf.train.Features(feature=cast_dict_val_to_bytes_feature(newlyfy_dictval(value))))

                    elif isinstancedictval(value, tf.train.Feature):
                        self._bytesfeatures['bytefeatures'] = self._bytesfeatures.newval(
                            tf.train.Features(feature=value))

                elif self._pack_in_single_feature is False:
                    for name, value in value.items():
                        name, value = name, value
                        print('name :', name, 'value :', value)
                        if isinstance(value, bytes):
                            self._bytesfeatures[name] = self._bytesfeatures.newval(
                                tf.train.Features(feature={name: _bytes_feature([value])}))

                        elif isinstance(value, list):
                            if False in map(lambda x: isinstance(x, bytes), value):
                                raise TypeError('Value entered must be of \'bytes\' type')
                            else:
                                self._bytesfeatures[name] = self._bytesfeatures.newval(
                                    tf.train.Features(feature={name: _bytes_feature(value)}))

                        elif isinstance(value, tuple):
                            if False in map(lambda x: isinstance(x, bytes), value):
                                raise TypeError('Value entered must be of \'bytes\' type')
                            else:
                                self._bytesfeatures[name] = self._bytesfeatures.newval(
                                    tf.train.Features(feature={name: _bytes_feature(list(value))}))

                        elif isinstance(value, descriptorseq.new):

                            if False in map(lambda x: isinstance(x, bytes), value.newly):
                                raise TypeError('Value entered must be of \'bytes\' type')
                            else:
                                self._bytesfeatures[name] = self._bytesfeatures.newval(
                                    tf.train.Features(feature={name: _bytes_feature(value.newly)}))

                        elif isinstance(value, tf.train.BytesList):
                            self._bytesfeatures[name] = self._bytesfeatures.newval(
                                tf.train.Features(feature={name: tf.train.Feature(bytes_list=value)}))

                        elif isinstance(value, tf.train.Feature):
                            self._bytesfeatures[name] = self._bytesfeatures.newval(
                                tf.train.Features(feature={name: value}))

                        elif False in map(lambda x: isinstance(x, bytes), value):
                            raise TypeError('Value entered must be of \'bytes\' type')

            elif isinstance(value, DescSeq):
                for name, value in value.items():
                    name, value = name, value
                    if isinstance(value, descriptorseq.new):
                        if False in map(lambda x: isinstance(x, bytes), value.newly):
                            raise TypeError('Value entered must be of \'bytes\' type')
                        else:
                            self._bytesfeatures[name] = self._bytesfeatures.newval(
                                tf.train.Features(feature={name: _bytes_feature(value.newly)}))


        @mixedfeatures.getter
        def mixedfeatures(self):
            return seq_to_norm_dict(self._mixedfeatures)

        @mixedfeatures.setter
        def mixedfeatures(self, value):
            if isinstance(value, dict):
                if self._pack_in_single_feature is True:

                    if isinstancedictval(value, tf.train.Feature):
                        self._mixedfeatures['mixedfeatures'] = self._mixedfeatures.newval(
                            tf.train.Features(feature=value))

                    else:
                        if isinstancedictval(value, list):
                            if isinstancelistvalindict(value, tf.train.Feature):
                                self._mixedfeatures['mixedfeatures'] = \
                                    self._mixedfeatures.newval(tf.train.FeatureLists(feature_list=cast_dict_val_to_feature_list(value)))
                            else:
                                raise TypeError('Expected dict type input with Feature Values')

                else:
                    raise AssertionError('Mixed Features does not support Single Feature Mapping')


def seq_to_norm_dict(f):
    result = {k: v.newly for k, v in f.items()}
    return result


def feature():
    data_feature = ToProto.Feature()
    return data_feature


def features(merge_all=None):
    data_features = ToProto.Features(merge_all)
    return data_features


def test_feature_and_features():
    BYTE_TEST = {'A': [b'a', b'c', b'd', b'e', b'f'], 'B': [b'b']}
    FLOAT_TEST = {'A': [1.1, 2.1, 3.1], 'B': [4.1, 5.1]}
    INT_TEST = {'A': [1, 2, 3], 'B': [4, 5]}

    # TO_MULTI_DICT = lambda d: {name: {name: value} for name, value in d.items()}
    not_in_single = features(merge_all=False)
    in_single = features()
    test = feature()

    test.bytefeature = BYTE_TEST
    test.floatfeature = FLOAT_TEST
    test.intfeature = INT_TEST

    not_in_single.bytefeatures = test.bytefeature
    not_in_single.floatfeatures = test.floatfeature
    not_in_single.intfeatures = test.intfeature
    in_single.bytefeatures = test.bytefeature
    in_single.floatfeatures = test.intfeature
    in_single.intfeatures = test.floatfeature

    nsb = not_in_single.bytefeatures
    nsf = not_in_single.floatfeatures
    nsi = not_in_single.intfeatures
    sb = in_single.bytefeatures
    sf = in_single.floatfeatures
    si = in_single.intfeatures
    print(nsb, sb, nsf, sf, nsi, si)


class DataToFeatures(object):
    def __init__(self, data, merge_all=True):
        self._merge_all = merge_all
        self._feature = feature()
        self._features = features(merge_all=self._merge_all)
        self._data = data
        self._mix = dict()

    @property
    def create_intfeature(self):
        self._feature.intfeature = self._data['intfeatures']
        self._mix.update(self._feature.intfeature)
        self._features.intfeatures = self._feature.intfeature
        if self._merge_all:
            self._data['intfeatures'] = self._features.intfeatures['intfeatures']
        else:
            self._data['intfeatures'] = self._features.intfeatures
        return

    @property
    def create_int64_list(self):
        self._feature.intfeature = self._data['int64_list']
        self._mix.update(self._feature.intfeature)
        self._data['int64_list'] = self._feature.intfeature
        return

    @property
    def create_bytefeature(self):
        self._feature.bytefeature = self._data['bytefeatures']
        self._mix.update(self._feature.bytefeature)
        self._features.bytefeatures = self._feature.bytefeature
        if self._merge_all:
            self._data['bytefeatures'] = self._features.bytefeatures['bytefeatures']
        else:
            self._data['bytefeatures'] = self._features.bytefeatures
        return

    @property
    def create_bytes_list(self):
        self._feature.bytefeature = self._data['bytes_list']
        self._mix.update(self._feature.bytefeature)
        self._data['bytes_list'] = self._feature.bytefeature
        return

    @property
    def create_float_feature(self):
        self._feature.floatfeature = self._data['floatfeatures']
        self._mix.update(self._feature.floatfeature)
        self._features.floatfeatures = self._feature.floatfeature
        if self._merge_all:
            self._data['floatfeatures'] = self._features.floatfeatures['floatfeatures']
        else:
            self._data['floatfeatures'] = self._features.floatfeatures
        return

    @property
    def create_float_list(self):
        self._feature.floatfeature = self._data['float_list']
        self._mix.update(self._feature.floatfeature)
        self._data['float_list'] = self._feature.floatfeature
        return

    @property
    def create_mixed_feature(self):
        self._features.mixedfeatures = self._mix
        print('Mixed Feature Dict', self._features.mixedfeatures)
        self._data['mixedfeatures'] = self._features.mixedfeatures['mixedfeatures']
        return


# noinspection PyPropertyDefinition
class CreateDictForData(object):

    def __init__(self):
        from collections import defaultdict as dfd
        self._dict = dfd(dict)

    @property
    def intfeatures(self):
        return self._dict['intfeatures']

    @intfeatures.setter
    def to_intfeatures_key(self, value):
        self._dict['intfeatures'].update(value)

    @property
    def int64_list(self):
        return self._dict['int64_list']

    @int64_list.setter
    def to_int64_list_key(self, value):
        self._dict['int64_list'].update(value)

    @property
    def floatfeatures(self):
        return self._dict['floatfeatures']

    @floatfeatures.setter
    def to_floatfeatures_key(self, value):
        self._dict['floatfeatures'].update(value)

    @property
    def float_list(self):
        return self._dict['float_list']

    @float_list.setter
    def to_float_list_key(self,value):
        self._dict['float_list'].update(value)

    @property
    def bytefeatures(self):
        return self._dict['bytefeatures']

    @bytefeatures.setter
    def to_bytefeatures_key(self, value):
        self._dict['bytefeatures'].update(value)

    @property
    def bytes_list(self):
        return self._dict['bytes_list']

    @bytes_list.setter
    def to_bytes_list_key(self, value):
        self._dict['bytes_list'].update(value)

    @property
    def dict(self):
        return self._dict


class CreateFeature(object):
    def __init__(self, merge_all=True):
        self._set_dict = CreateDictForData()
        self._set_feature = DataToFeatures(self._set_dict.dict, merge_all=merge_all)

    @property
    def set(self):
        return self._set_dict

    @property
    def feature(self):
        _feature = self._set_feature
        return _feature

    @property
    def dict(self):
        return self._set_dict.dict


def test_CreateFeature():
    BYTE_TEST = {'A': [b'a', b'c', b'd', b'e', b'f'], 'B': [b'b']}
    FLOAT_TEST = {'C': [1.1, 2.1, 3.1], 'D': [4.1, 5.1]}
    INT_TEST = {'E': [1, 2, 3], 'F': [4, 5]}
    data = CreateFeature()
    data.set.to_bytefeatures_key = BYTE_TEST
    data.set.to_intfeatures_key = INT_TEST
    data.set.to_floatfeatures_key = FLOAT_TEST
    _ = data.feature.create_bytefeature
    _ = data.feature.create_intfeature
    _ = data.feature.create_float_feature
    _ = data.feature.create_mixed_feature
    print(data.set.dict)


class Feature(object):

    def __init__(self, int_dict=None, float_dict=None, byte_dict=None):
        self._int_dict = int_dict
        self._float_dict = float_dict
        self._byte_dict = byte_dict
        self._create = CreateFeature(merge_all=True)

    @property
    def make_int_list(self):
        self._create.set.to_int64_list_key = self._int_dict
        _ = self._create.feature.create_int64_list

    @property
    def make_float_list(self):
        self._create.set.to_float_list_key = self._float_dict
        _ = self._create.feature.create_float_list

    @property
    def make_byte_list(self):
        self._create.set.to_bytes_list_key = self._byte_dict
        _ = self._create.feature.create_bytes_list

    @property
    def make_mixed_feature(self):
        _ = self._create.feature.create_mixed_feature

    @property
    def dict(self):
        return self._create.dict


class Features(object):

    def __init__(self, int_dict=None, float_dict=None, byte_dict=None, merge_all=True):
        self._int_dict = int_dict
        self._float_dict = float_dict
        self._byte_dict = byte_dict
        self._create = CreateFeature(merge_all=merge_all)

    @property
    def make_int_feature(self):
        self._create.set.to_intfeatures_key = self._int_dict
        _ = self._create.feature.create_intfeature

    @property
    def make_float_feature(self):
        self._create.set.to_floatfeatures_key = self._float_dict
        _ = self._create.feature.create_float_feature

    @property
    def make_byte_feature(self):
        self._create.set.to_bytefeatures_key = self._byte_dict
        _ = self._create.feature.create_bytefeature

    @property
    def make_mixed_feature(self):
        _ = self._create.feature.create_mixed_feature

    @property
    def dict(self):
        return self._create.dict


def test_Feature():
    BYTE_TEST = {'A': [b'a', b'c', b'd', b'e', b'f'], 'B': [b'b']}
    FLOAT_TEST = {'C': [1.1, 2.1, 3.1], 'D': [4.1, 5.1]}
    INT_TEST = {'E': [1, 2, 3], 'F': [4, 5]}
    _feature = Feature(int_dict=INT_TEST, float_dict=FLOAT_TEST, byte_dict=BYTE_TEST)
    _ = _feature.make_int_list
    _ = _feature.make_byte_list
    _ = _feature.make_float_list
    _ = _feature.make_mixed_feature
    print(_feature.dict)


def test_Features():
    BYTE_TEST = {'A': [b'a', b'c', b'd', b'e', b'f'], 'B': [b'b']}
    FLOAT_TEST = {'C': [1.1, 2.1, 3.1], 'D': [4.1, 5.1]}
    INT_TEST = {'E': [1, 2, 3], 'F': [4, 5]}
    _features = Features(int_dict=INT_TEST, float_dict=FLOAT_TEST, byte_dict=BYTE_TEST)
    _ = _features.make_int_feature
    _ = _features.make_byte_feature
    _ = _features.make_float_feature
    _ = _features.make_mixed_feature
    print(_features.dict)


class Protofy(object):

    def __init__(self, int_dict=None, float_dict=None, byte_dict=None):
        self._feature = Feature
        self._features = Features
        self._int_dict = int_dict
        self._float_dict = float_dict
        self._byte_dict = byte_dict
        self._dict = None

    @property
    def feature(self):
        _feature = self._feature(int_dict=self._int_dict, float_dict=self._float_dict, byte_dict=self._byte_dict)
        if self._int_dict is not None:
            _ = _feature.make_int_list
        if self._float_dict is not None:
            _ = _feature.make_float_list
        if self._byte_dict is not None:
            _ = _feature.make_byte_list
        _ = _feature.make_mixed_feature
        self._dict = _feature.dict

    @property
    def features(self):
        _features = self._features(int_dict=self._int_dict, float_dict=self._float_dict, byte_dict=self._byte_dict)
        if self._int_dict is not None:
            _ = _features.make_int_feature
        if self._float_dict is not None:
            _ = _features.make_float_feature
        if self._byte_dict is not None:
            _ = _features.make_byte_feature
        _ = _features.make_mixed_feature
        self._dict = _features.dict

    @property
    def dict(self):
        return self._dict


def protofy(int_dict=None, float_dict=None, byte_dict=None):
    _proto = Protofy(int_dict=int_dict, float_dict=float_dict, byte_dict=byte_dict)
    _ = _proto.feature
    return _proto.dict


def protofy_sequence(context_dict, sequence_dict):
    context = protofy(**context_dict)
    sequence = protofy(**sequence_dict)
    return context, sequence


if __name__ == '__main__':
    import cProfile
    cProfile.run('protofy(int_dict={\'testing_int\': [[1, 3, 5], [1, 3, 5]]})')



