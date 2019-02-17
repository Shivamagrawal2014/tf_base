import tensorflow as tf


class TFRecordReaderBase(object):

    def __init__(self, is_sequence_example):
        self._tf_record_path = None
        self._is_sequential_data = is_sequence_example

        # feature mappings data
        self._example_feature_dict = None
        self._sequence_example_feature_dict = None
        self._context_feature_dict = None
        self._sequence_feature_dict = None

        self._apply_func = None

    @staticmethod
    def _compression_options(tf_record_compression):
        assert tf_record_compression in \
               (True, False, None, 'GZIP', 'ZLIB', tf.python_io.TFRecordCompressionType.GZIP,
                tf.python_io.TFRecordCompressionType.ZLIB, tf.python_io.TFRecordCompressionType.NONE)
        if tf_record_compression in (False, None, tf.python_io.TFRecordCompressionType.NONE):
            options = None

        elif tf_record_compression in (True, 'GZIP', tf.python_io.TFRecordCompressionType.GZIP):
            options = 'GZIP'

        else:
            assert tf_record_compression in ('ZLIB', tf.python_io.TFRecordCompressionType.ZLIB)
            options = 'ZLIB'

        return options

    @property
    def _example_map(self):
        return self._example_feature_dict

    @_example_map.setter
    def _example_map(self, feature_mapping):
        if self._example_feature_dict is None:
            print('Setting feature mapping...')
            self._example_feature_dict = feature_mapping
            print('Set to :', self._example_feature_dict)

    @property
    def _sequence_example_map(self):
        return self._context_feature_dict, self._sequence_feature_dict

    @_sequence_example_map.setter
    def _sequence_example_map(self, feature_mapping):
        if self._sequence_feature_dict is None or self._context_feature_dict is None:
            print('Setting feature mapping...')
            (context_dict, sequence_dict) = self._sequence_example_feature_dict = feature_mapping
            self._context_feature_dict = context_dict
            self._sequence_feature_dict = sequence_dict
            print('Set to :', self._context_feature_dict, self._sequence_feature_dict)

    @property
    def _feature_parser(self):
        return self._apply_func

    @_feature_parser.setter
    def _feature_parser(self, func=None):
        if self._apply_func is None:
            if func:
                assert callable(func)
                self._apply_func = func
            else:
                self._apply_func = self._dummy_apply_func

    @staticmethod
    def _dummy_apply_func(*args, **kwargs):
        if args and kwargs:
            return args, kwargs
        else:
            if args:
                return args
            elif kwargs:
                return kwargs
            else:
                return

    def _data_set_serialized_output(self, tf_record_path, tf_record_compression: bool = True):
        """
        :param tf_record_path:
        :param tf_record_compression:
        :return:
        """
        if self._tf_record_path is None:
            if isinstance(tf_record_path, str):
                self._tf_record_path = [tf_record_path]
            else:
                self._tf_record_path = tf_record_path

        print('file_name path : ', self._tf_record_path)
        options = self._compression_options(tf_record_compression=tf_record_compression)
        serialized_output = tf.contrib.data.TFRecordDataset(
            filenames=self._tf_record_path, compression_type=options)
        # self.tf.add_to_collection()
        print('Serialized Output :', type(serialized_output).__name__+'.')
        return serialized_output

    @staticmethod
    def _parse_single_example(serialized_output,
                              read_format_feature_dict):
        """
        :param serialized_output:
        :param read_format_feature_dict:
        :return:
        """
        print('parsing serialized output to single example. ', serialized_output)
        print('with read_format :', read_format_feature_dict)
        _parse_single_example = tf.parse_single_example(
            serialized_output, features=read_format_feature_dict)
        print('parsed serialized output to single records.', _parse_single_example)
        return _parse_single_example

    @staticmethod
    def _parse_single_sequence_example(serialized_output,
                                       read_format_context_feature_dict,
                                       read_format_sequence_feature_dict):
        """
        :param serialized_output:
        :param read_format_context_feature_dict:
        :param read_format_sequence_feature_dict:
        :return:
        """

        _parsed_single_sequence_example = \
            tf.parse_single_sequence_example(serialized_output,
                                             read_format_context_feature_dict,
                                             read_format_sequence_feature_dict)

        return _parsed_single_sequence_example

    def _single_example(self, serialized_output):
        """
        :param serialized_output:
        :return:
        """
        read_format_feature_dict = self._example_map
        _parsed_single_example = self._parse_single_example(serialized_output,
                                                            read_format_feature_dict)
        return _parsed_single_example

    def _single_sequence_example(self, serialized_output):
        """
        :param serialized_output:
        :return:
        """
        read_format_context_feature_dict, read_format_sequence_feature_dict = self._sequence_example_map
        _parsed_single_sequence_example = self._parse_single_sequence_example(
            serialized_output,
            read_format_context_feature_dict,
            read_format_sequence_feature_dict)

        return _parsed_single_sequence_example

    def _mini_batch_example(self, serialized_output):
        return self._apply_func(self._single_example(serialized_output))

    def _mini_batch_sequence_example(self, serialized_output):
        return self._apply_func(self._single_sequence_example(serialized_output))

    def _map_shuffle_batch_repeat(self,
                                  tf_record_path,
                                  apply_func,
                                  buffer_size,
                                  batch_size,
                                  epochs_size,
                                  tf_record_compression
                                  ):
        data_set = self._data_set_serialized_output(
            tf_record_path=tf_record_path, tf_record_compression=tf_record_compression)
        data_set = data_set.map(apply_func)
        data_set = data_set.shuffle(buffer_size=buffer_size)
        data_set = data_set.batch(batch_size=batch_size)
        data_set = data_set.repeat(count=epochs_size)
        return data_set

    def _batch_sequence_example(self,
                                tf_record_path,
                                buffer_size,
                                batch_size,
                                epochs_size,
                                tf_record_compression
                                ):
        return self._map_shuffle_batch_repeat(tf_record_path=tf_record_path,
                                              apply_func=self._mini_batch_sequence_example,
                                              buffer_size=buffer_size,
                                              batch_size=batch_size,
                                              epochs_size=epochs_size,
                                              tf_record_compression=tf_record_compression)

    def _batch_example(self,
                       tf_record_path,
                       buffer_size,
                       batch_size,
                       epochs_size,
                       tf_record_compression
                       ):
        return self._map_shuffle_batch_repeat(tf_record_path=tf_record_path,
                                              apply_func=self._mini_batch_example,
                                              buffer_size=buffer_size,
                                              batch_size=batch_size,
                                              epochs_size=epochs_size,
                                              tf_record_compression=tf_record_compression)

    def _get_batch(self,
                   tf_record_path,
                   buffer_size=10000,
                   batch_size=15,
                   epochs_size=2000,
                   tf_record_compression=True):
        buffer_size = buffer_size or 10000
        batch_size = batch_size or 15
        epochs_size = epochs_size or 2000
        if self._is_sequential_data:
            data = self._batch_sequence_example(tf_record_path=tf_record_path,
                                                buffer_size=buffer_size,
                                                batch_size=batch_size,
                                                epochs_size=epochs_size,
                                                tf_record_compression=tf_record_compression)
        else:
            data = self._batch_example(tf_record_path=tf_record_path,
                                       buffer_size=buffer_size,
                                       batch_size=batch_size,
                                       epochs_size=epochs_size,
                                       tf_record_compression=tf_record_compression)
        return data

    @staticmethod
    def _summary_writer(summary_dir, graph):
        summ_writer = tf.summary.FileWriter(logdir=summary_dir, graph=graph)
        return summ_writer


class TFRecordExampleReader(TFRecordReaderBase):

    def __init__(self, is_sequence_example=False):
        super(TFRecordExampleReader, self).__init__(is_sequence_example)
        self._example_map = self.feature_map()
        self._feature_parser = self.feature_parser

    def feature_map(self):
        return NotImplemented

    def feature_parser(self, parsed_single_example):
        return NotImplemented


class TFRecordSequenceExampleReader(TFRecordReaderBase):

    def __init__(self, is_sequence_example=True):
        super(TFRecordSequenceExampleReader, self).__init__(is_sequence_example)
        self._example_map = self.feature_map()
        self._feature_parser = self.feature_parser

    def feature_map(self):
        return NotImplemented

    def feature_parser(self):
        return NotImplemented


if __name__ == '__main__':
    tf_record_path = r'C:\Users\shivam.agarwal\PycharmProjects\TopicAPI\data\image\GH_images.tfr'
    reader = TFRecordExampleReader()
    feat_map = FeatureMap()
    reader._example_map = feat_map._image_map
    reader._feature_parser = feat_map._image_feature_parser
    data = reader._get_batch(tf_record_path=tf_record_path,
                             batch_size=4,
                             epochs_size=20)

    data = data.make_one_shot_iterator()
    # init = data.initializer
    sess = reader.session
    # sess.run(init)
    data = data.get_next()
    summarizer = reader._summary_writer('../summary', graph)

    with summarizer:
        try:
            for _ in range(21):
                print((sess.run(data)[0]))
            print('Completed!')
        except tf.errors.OutOfRangeError:
            print('Data Exhausted!')
