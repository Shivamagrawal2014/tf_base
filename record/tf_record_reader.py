import tensorflow as tf
from graph.tf_graph_api import GraphAPI
from record.feature_map import FeatureMap

Graph = GraphAPI()


class TFRecordReaderBase(metaclass=Graph()):

    def __init__(self, is_sequence_example):
        self._tf_path = None
        self._is_sequential_data = is_sequence_example

        # feature mappings data
        self._example_feature_dict = None
        self._sequence_example_feature_dict = None
        self._context_feature_dict = None
        self._sequence_feature_dict = None

        self._apply_func = None

    # def session(self):
    #     return self._session
    #
    # def graph(self):
    #     return self._graph

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
        self._example_feature_dict = feature_mapping

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

    def _data_set_serialized_output(self, tf_path, tf_record_compression: bool = True):
        """
        :param tf_path:
        :param tf_record_compression:
        :return:
        """
        if self._tf_path is None:
            if isinstance(tf_path, str):
                self._tf_path = [tf_path]
            else:
                self._tf_path = tf_path

        print('file_name path : ', self._tf_path)
        options = self._compression_options(tf_record_compression=tf_record_compression)
        serialized_output = tf.data.TFRecordDataset(filenames=self._tf_path, compression_type=options)
        # self.tf.add_to_collection()
        print('Serialized Output :', type(serialized_output).__name__+'.')
        return serialized_output

    def _parse_single_example(self,
                              serialized_output,
                              read_format_feature_dict
                               ):
        """
        :param serialized_output:
        :param read_format_feature_dict:
        :return:
        """
        _parse_single_example = tf.parse_single_example(
            serialized_output, features=read_format_feature_dict)
        return _parse_single_example

    def _parse_single_sequence_example(self,
                                       serialized_output,
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

    def _single_sequence_example(self,
                                 serialized_output,
                                 ):
        """
        :param serialized_output:
        :return:
        """
        read_format_context_feature_dict, read_format_sequence_feature_dict = self._sequence_example_map
        _parsed_single_sequence_example = self._parse_single_sequence_example(serialized_output,
                                                                              read_format_context_feature_dict,
                                                                              read_format_sequence_feature_dict)

        return _parsed_single_sequence_example

    def _mini_batch_example(self, serialized_output):
        return self._apply_func(self._single_example(serialized_output))

    def _mini_batch_sequence_example(self, serialized_output):
        return self._apply_func(self._single_sequence_example(serialized_output))

    def _map_shuffle_batch_repeat(self,
                                  tf_path,
                                  apply_func,
                                  buffer_size,
                                  batch_size,
                                  epochs_size
                                  ):
        data_set = self._data_set_serialized_output(tf_path=tf_path)
        data_set = data_set.map(apply_func)
        data_set = data_set.shuffle(buffer_size=buffer_size)
        data_set = data_set.batch(batch_size=batch_size)
        data_set = data_set.repeat(count=epochs_size)
        return data_set

    def _batch_sequence_example(self,
                                tf_path,
                                buffer_size,
                                batch_size,
                                epochs_size
                                ):
        return self._map_shuffle_batch_repeat(tf_path=tf_path,
                                              apply_func=self._mini_batch_sequence_example,
                                              buffer_size=buffer_size,
                                              batch_size=batch_size,
                                              epochs_size=epochs_size)

    def _batch_example(self,
                       tf_path,
                       buffer_size,
                       batch_size,
                       epochs_size
                       ):
        return self._map_shuffle_batch_repeat(tf_path=tf_path,
                                              apply_func=self._mini_batch_example,
                                              buffer_size=buffer_size,
                                              batch_size=batch_size,
                                              epochs_size=epochs_size)

    def _get_batch(self,
                   tf_path,
                   buffer_size=10000,
                   batch_size=15,
                   epochs_size=2000):
        buffer_size = buffer_size or 10000
        batch_size = batch_size or 15
        epochs_size = epochs_size or 2000
        if self._is_sequential_data:
            data = self._batch_sequence_example(tf_path=tf_path,
                                                buffer_size=buffer_size,
                                                batch_size=batch_size,
                                                epochs_size=epochs_size)
        else:
            data = self._batch_example(tf_path=tf_path,
                                       buffer_size=buffer_size,
                                       batch_size=batch_size,
                                       epochs_size=epochs_size)
        return data

    def summary_writer(self, summary_dir):
        summ_writer = tf.summary.FileWriter(logdir=summary_dir, graph=self._graph)
        return summ_writer


if __name__ == '__main__':
    tf_record_path = r'C:\Users\shivam.agarwal\PycharmProjects\TopicAPI\data\image\GH_images.tfr'
    reader = TFRecordReader(is_sequence_example=False)
    feat_map = FeatureMap()
    reader._example_map = feat_map._image_map
    reader._feature_parser = feat_map._image_feature_parser
    data = reader._get_batch(tf_path=tf_record_path,
                             batch_size=4,
                             epochs_size=20)

    data = data.make_one_shot_iterator()
    # init = data.initializer
    sess = reader.session()
    # sess.run(init)
    data = data.get_next()
    summarizer = reader.summary_writer('../summary')

    with summarizer:
        try:
            for _ in range(21):
                print((sess.run(data)[0]))
            print('Completed!')
        except tf.errors.OutOfRangeError:
            print('Data Exhausted!')
