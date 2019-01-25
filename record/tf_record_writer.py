import tensorflow as tf
import os


class TFRecordWriterBase(object):

    def _features(self):
        return NotImplemented

    def _options(self, allow_compression=True):
        """
        :param allow_compression:
        :return:
        """
        assert allow_compression in (True, False, None, 'ZLIB', 'GZIP',
                                     tf.python_io.TFRecordCompressionType.GZIP,
                                     tf.python_io.TFRecordCompressionType.ZLIB,
                                     tf.python_io.TFRecordCompressionType.NONE)

        if allow_compression in (True, 'GZIP', tf.python_io.TFRecordCompressionType.GZIP):
            option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        elif allow_compression in (False, 'ZLIB', tf.python_io.TFRecordCompressionType.ZLIB):
            option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        else:
            option = tf.python_io.TFRecordCompressionType.NONE
        return option

    @staticmethod
    def _create_tf_record_name(tf_record_name):
        """
        :param tf_record_name:
        :return:
        """
        return '.'.join([tf_record_name, 'tfr'])

    def _writer(self, tf_record_name, allow_compression=None):
        """
        :param tf_record_name:
        :param allow_compression:
        :return:
        """
        options = self._options(allow_compression)
        writer = tf.python_io.TFRecordWriter(tf_record_name, options=options)
        return writer

    @staticmethod
    def _example(features):
        """
        :param features:
        :return:
        """
        return tf.train.Example(features=features)

    def _features_to_examples(self):
        return [self._example(self._features()[_feature_key]) for _feature_key in self._features()]

    @staticmethod
    def _serialize_examples(examples):
        """
        :param examples:
        :return:
        """
        return list(map(lambda x: x.SerializeToString(), examples))

    def _to_tfr(self, tfrecord_name, tfrecord_folder, allow_compression=None):
        _examples = self._features_to_examples()
        _serialized_examples = self._serialize_examples(_examples)

        tf_path = os.path.join(tfrecord_folder, self._create_tf_record_name(tfrecord_name))
        file = self._writer(tf_path, allow_compression)
        eg_counter = 0
        with file:
            for _example in _serialized_examples:
                eg_counter += 1
                file.write(_example)
                print('written eg: - {count}.'.format(count=eg_counter))
        print('Completed!')
        return tf_path





