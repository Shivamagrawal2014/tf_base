from file.record.record_writer import TFRecordWriterBase
from file.record.record_reader import TFRecordExampleReader
from file.image.open_image import OpenImage
from file.record.protofy import protofy
from graph.tf_graph_api import GraphAPI

from typing import List, Tuple
import tensorflow as tf


Graph = GraphAPI()


class ImageTFRecordWriter(TFRecordWriterBase, OpenImage):

    def __init__(self, folder: str, extensions: List[str],
                 size: Tuple[int, int, int] = None, show=False):
        super(ImageTFRecordWriter, self).__init__(folder=folder,
                                                  extensions=extensions,
                                                  size=size,
                                                  show=show)

    def _features(self):
        return self.open_image

    def _protofy_image(self, image, shape):
        if not isinstance(image, (str, bytes)):
            if hasattr(image, 'tostring'):
                image = image.tostring()
            else:
                print(type(image))
                image = image.encode('utf-8')
        return protofy(byte_dict={'pixel': image}, int_dict={'shape': list(shape)})

    def to_tfr(self, tfrecord_name, save_folder, allow_compression=None):
        return self._to_tfr(tfrecord_name, save_folder, allow_compression)


class ImageTFRecordReader(TFRecordExampleReader):
    __metaclass__ = Graph()

    def __init__(self):
        super(ImageTFRecordReader, self).__init__()

    @property
    def feature_map(self):
        return {'pixel': tf.FixedLenFeature([], dtype=tf.string),
                'shape': tf.FixedLenFeature([3], dtype=tf.int64)}

    @property
    def feature_parser(self):
        def _parser(parsed_single_example):
            example = parsed_single_example

            pixel = tf.decode_raw(
                example['pixel'], out_type=tf.int64, name='decode_raw_pixel')

            pixel = tf.cast(pixel, dtype=tf.uint8, name='cast_pixel_to_uint8')

            shape = tf.map_fn(
                lambda x: tf.cast(x, dtype=tf.int32, name='shape_cast'), example['shape'])

            pixel = tf.reshape(pixel, shape)
            return pixel, shape
        return _parser

    def batch(self,
              tf_path,
              buffer_size=10000,
              batch_size=15,
              epochs_size=2000):
        self._get_batch(self, tf_path, buffer_size, batch_size, epochs_size)


if __name__ == '__main__':
    # images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
    #                              size=(50, 50, 0), show=False)
    # record = images.to_tfr(tfrecord_name='ubuntu_images3',
    #                        save_folder='/home/shivam/Documents/', allow_compression=True)

    reader = ImageTFRecordReader()

    reader.example_parser()
