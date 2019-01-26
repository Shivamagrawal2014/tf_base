from file.image import OpenImage
import unittest


def main():

    class TestOpenImage(unittest.TestCase):

        def __init__(self):
            super(TestOpenImage, self).__init__()
            self._open_image = OpenImage('/home/shivam/Documents/test_images', ['jpg'])

        def test_image(self):
            # self.assertEqual(self._open_image.open_image,
            #                  {'hanuman.jpg': '/home/shivam/Documents/test_images/hanuman.jpg'})
            self.assertEqual(self._open_image.image_folder_files,
                             {'jpg': {'test_images': ['/home/shivam/Documents/test_images/hanuman.jpg']}})
            self.assertEqual(self._open_image.image_folders, {'jpg': ['test_images']})
            self.assertEqual(self._open_image.image_files,
                             {'jpg': [['/home/shivam/Documents/test_images/hanuman.jpg']]})

    test_image = TestOpenImage()
    test_image.test_image()


if __name__ == '__main__':
    main()
