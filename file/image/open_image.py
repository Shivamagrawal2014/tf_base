from file.open_file import OpenFile
from typing import List, Tuple
from file import IMAGE_TYPE_EXTENSION
from file.image.parse_image import ParseImage


class OpenImage(OpenFile):

    def __init__(self, folder: str, extensions: List[str], size: Tuple[int, int, int] = None):
        assert all((ext in IMAGE_TYPE_EXTENSION for ext in extensions))
        super(OpenImage, self).__init__(folder, extensions)
        self.__parse_image = ParseImage(size)

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
        return self.__parse_image(file_path)


def main():
    image = OpenImage('/home/shivam/Documents/', ['jpg'], size=(20, 20, 3)).open_image
    print(image)


if __name__ == '__main__':
    main()

