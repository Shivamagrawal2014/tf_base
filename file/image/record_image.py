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
                image = image.encode('utf-8')
        return protofy(byte_dict={'pixel': image}, int_dict={'shape': list(shape)})

    def to_tfr(self, tfrecord_name, save_folder, allow_compression=None):
        return self._to_tfr(tfrecord_name, save_folder, allow_compression)


class ImageTFRecordReader(TFRecordExampleReader):
    __metaclass__ = metaclass=Graph()

    def __init__(self):
        super(ImageTFRecordReader, self).__init__()

    def feature_map(self, feature_dict: dict=None):
        return feature_dict or {'pixel': tf.FixedLenFeature([], dtype=tf.string),
                                'shape': tf.FixedLenFeature([3], dtype=tf.int64)}
    def feature_parser(self):



if __name__ == '__main__':
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(50, 50, 0), show=False)
    images.to_tfr(tfrecord_name='ubuntu_images1',
                  save_folder='/home/shivam/Documents/', allow_compression=True)
