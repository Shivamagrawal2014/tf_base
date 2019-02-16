from file.record.writer import TFRecordWriterBase
from file.record.reader import TFRecordExampleReader
from file.image.open_image import OpenImage
from file.record.protofy import protofy
from graph.tf_graph_api import GraphAPI
from typing import List, Tuple
import tensorflow as tf
from six import add_metaclass
import os

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

    def _protofy_image(self, image, shape, file_path):
        if hasattr(image, 'tostring'):
            image = image.tostring()
        label = os.path.splitext(file_path)[0]
        return protofy(byte_dict={'pixel': image, 'label': label}, int_dict={'shape': list(shape)})

    def to_tfr(self, tfrecord_name, save_folder, allow_compression=None):
        return self._to_tfr(tfrecord_name, save_folder, allow_compression)


@add_metaclass(Graph())
class ImageTFRecordReader(TFRecordExampleReader):

    def __init__(self):
        super(ImageTFRecordReader, self).__init__()

    def feature_map(self):
        return {'pixel': tf.FixedLenFeature([], dtype=tf.string),
                'label': tf.FixedLenFeature([], dtype=tf.string),
                'shape': tf.FixedLenFeature([3], dtype=tf.int64)}

    def feature_parser(self, parsed_single_example):
        example = parsed_single_example

        pixel = tf.decode_raw(
            example['pixel'], out_type=tf.uint8, name='decode_raw_pixel')

        pixel = tf.cast(pixel, dtype=tf.uint8, name='cast_pixel_to_uint8')
        shape = tf.cast(example['shape'], dtype=tf.int32, name='shape_cast')
        pixel = tf.reshape(pixel, shape)
        label = example['label']
        return pixel, label

    def summary_writer(self, summary_dir, graph=None):
        graph = graph or self.graph
        return self._summary_writer(summary_dir, graph)

    def batch(self,
              tf_path,
              buffer_size=10000,
              batch_size=15,
              epochs_size=2000):
        return self._get_batch(tf_path, buffer_size, batch_size, epochs_size)


def test_write():
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(20, 20, 0), show=False)
    record = images.to_tfr(tfrecord_name='ubuntu_images_2',
                           save_folder='/home/shivam/Documents/', allow_compression=True)
    return record


if __name__ == '__main__':
    record = r'/home/shivam/Documents/ubuntu_images_2.tfr'  # test_write()
    reader = ImageTFRecordReader()
    data = reader.batch(tf_path=record, batch_size=2, epochs_size=1)
    data = data.make_one_shot_iterator()
    # init = data.initializer
    sess = reader.session
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.run(init)
    data = data.get_next()
    summarizer = reader.summary_writer('../summary', sess.graph)
    try:
        for _ in range(21):
            image, shape = sess.run(data)
            print(image.shape, shape)
        print('Completed!')
    except tf.errors.OutOfRangeError:
        print('Data Exhausted!')
    finally:
        summarizer.close()
