from record.tf_record_writer import TFRecordWriterBase
from file.image.open_image import OpenImage
from typing import List, Tuple


class ImageTFRecordWriter(TFRecordWriterBase):

    def __init__(self, folder: str, extensions: List[str],
                 size: Tuple[int, int, int] = None, show=False):
        super(ImageTFRecordWriter, self).__init__()
        self._images = OpenImage(folder=folder,
                                 extensions=extensions,
                                 size=size,
                                 show=show)

    def _features(self):
        return self._images.open_image

    def to_tfr(self, tfrecord_name, save_folder, allow_compression=None):
        return self._to_tfr(tfrecord_name, save_folder, allow_compression)


if __name__ == '__main__':
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(50, 50, 0), show=False)
    images.to_tfr(tfrecord_name='ubuntu_images',
                  save_folder='/home/shivam/Documents/', allow_compression=True)
