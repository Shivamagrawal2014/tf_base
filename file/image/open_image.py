from file.open_file import OpenFile
from typing import List
from file import IMAGE_TYPE_EXTENSION


class OpenImage(OpenFile):

    def __init__(self, folder: str, extensions: List[str]):
        assert all((ext in IMAGE_TYPE_EXTENSION for ext in extensions))
        super(OpenImage, self).__init__(folder, extensions)

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
        return file_path


def main():
    from pprint import pprint
    image = OpenImage('/home/shivam/Documents/test_images', ['jpg'])
    open_image = image.open_image
    image_files = image.image_files
    image_folders = image.image_folders
    image_folders_files = image.image_folder_files
    pprint(open_image)
    pprint(image_files)
    pprint(image_folders)
    pprint(image_folders_files)


if __name__ == '__main__':
    main()

