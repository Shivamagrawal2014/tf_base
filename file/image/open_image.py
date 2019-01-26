from typing import List, Tuple
from file.open_file import OpenFile
from file import IMAGE_TYPE_EXTENSION
from file.image.parse_image import ParseImage


class OpenImage(OpenFile):

    def __init__(self, folder: str, extensions: List[str], size: Tuple[int, int, int] = None, show=False):
        assert all((ext in IMAGE_TYPE_EXTENSION for ext in extensions))
        super(OpenImage, self).__init__(folder, extensions)
        self.__parse_image = ParseImage(size, show)

    @property
    def open_image(self):
        return self._open_files()

    @property
    def image_folder_files(self):
        return self._extension_folder_files

    @property
    def image_folders(self):
        return self._extensions_to_folder

    @property
    def image_files(self):
        return self._extensions_to_files

    def _parse_image(self, file_path):
        image, shape = self.__parse_image(file_path)
        return self._protofy_image(image, shape)

    def _protofy_image(self, image, shape):
        return NotImplemented


def main():
    image = OpenImage('/home/shivam/Documents/', ['jpg'],
                      size=(500, 500, 0), show=False).open_image
    print(image)


if __name__ == '__main__':
    main()

