from file.record.tf_record_writer import TFRecordWriterBase
from file.record import TFRecordReaderBase
from file.image.open_image import OpenImage
from typing import List, Tuple
import tensorflow as tf
from utils.protobuf import protofy


class ImageFeature(OpenImage):

    def __init__(self, folder: str,
                 extensions: List[str],
                 size: Tuple[int, int, int] = None,
                 show=False):

        super(ImageFeature, self).__init__(folder=folder,
                                           extensions=extensions,
                                           size=size,
                                           show=show)

    def _protofy_image(self, image, shape):
        if not isinstance(image, (str, bytes)):
            if hasattr(image, 'tostring'):
                image = image.tostring()
            else:
                image = image.encode('utf-8')
        return protofy(byte_dict={'pixel': image}, int_dict={'shape': list(shape)})


class ImageTFRecordWriter(TFRecordWriterBase):

    def __init__(self, folder: str, extensions: List[str],
                 size: Tuple[int, int, int] = None, show=False):
        super(ImageTFRecordWriter, self).__init__()
        self._images = ImageFeature(folder=folder,
                                    extensions=extensions,
                                    size=size,
                                    show=show)

    def _features(self):
        return self._images.open_image

    def to_tfr(self, tfrecord_name, save_folder, allow_compression=None):
        return self._to_tfr(tfrecord_name, save_folder, allow_compression)


class ImageTFRecordReader(TFRecordReaderBase):

    def __init__(self, is_sequence_example):
        super(ImageTFRecordReader, self).__init__(is_sequence_example)

    # def _feature_map(self, feature_dict: dict=None):
    #     self.


def image_mapping():
    return {'pixel': tf.FixedLenFeature([], dtype=tf.string),
            'shape': tf.FixedLenFeature([3], dtype=tf.int64)}


if __name__ == '__main__':
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(50, 50, 0), show=False)
    images.to_tfr(tfrecord_name='ubuntu_images1',
                  save_folder='/home/shivam/Documents/', allow_compression=True)
