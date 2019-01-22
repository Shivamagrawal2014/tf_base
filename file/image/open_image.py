from file.open_file import OpenFile
from typing import List
from file import IMAGE_TYPE_EXTENSION


class OpenImage(OpenFile):

    def __init__(self, folder: str, extensions: List[str]):
        assert all((ext in IMAGE_TYPE_EXTENSION for ext in extensions))
        super(OpenImage, self).__init__(folder, extensions)

    @property
    def open_image(self):
        return super(OpenImage, self)._open_files()

    @property
    def image_folders(self):
        return super(OpenImage, self)._extensions_to_folder

    @property
    def image_files(self):
        return super(OpenImage, self)._extensions_to_files

    def _parse_image(self, file_path):
        return file_path


if __name__ == '__main__':
    from pprint import pprint
    image = OpenImage('/home/shivam/Documents/', ['jpg'])
    open_image = image.open_image
    image_files = image.image_files
    image_folders = image.image_folders
    pprint(open_image)
