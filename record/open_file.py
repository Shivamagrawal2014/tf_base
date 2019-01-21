from . import os
from . import List
from . import (IMAGE_TYPE_EXTENSION,
               AUDIO_TYPE_EXTENSION,
               DOC_TYPE_EXTENSION,
               URL_TYPE_EXTENSION)
from . import FileType
from . import _is_clean_folder, glob, _file_extension


class OpenFile(object):

    def __init__(self, folder:str, extensions:List[str]):
        self._folder = folder
        self._extensions = extensions
        self._opened = False
        self._file_types = None
        self._extension_folder_files = None
        self._extension_folder = None
        self._extension_files = None

    @staticmethod
    def _files_with_extension(folder: str, extension: str):
        _folder_dict = dict()
        for root, subfolders, files in os.walk(folder):
            for sub_folder in subfolders:
                if _is_clean_folder(sub_folder):
                    _folder_dict[sub_folder] = \
                        glob(os.path.join(os.path.join(root, sub_folder),
                                          '.'.join(['*', extension])))

            if any(files):
                _folder_dict[root] = [file for file in files
                                      if file.endswith(extension)]
        return _folder_dict

    def _find_files(self, folder: str, extensions: List[str]):
        if self._extension_folder_files is None:
            self._extension_folder_files = dict()
            for extension in extensions:
                self._extension_folder_files[extension] = self._files_with_extension(folder, extension)

        return self._extension_folder_files

    @property
    def _extensions_to_folder(self):
        if self._extension_folder is None:
            self._extension_folder = {extension: list(self._extension_folder_files[extension])
                                      for extension in self._extension_folder_files}
        return self._extension_folder

    @property
    def _extensions_to_files(self):
        if self._extension_files is None:
            self._extension_files = {extension: list(self._extension_folder_files[extension].values())
                                     for extension in self._extension_folder_files}
        return self._extension_files

    def _open_files(self):
        if self._opened is False:
            self._opened = dict()
            for extension in self._extension_folder_files:
                if self._file_types is None:
                    self._file_types = dict()
                for folder in self._extension_folder_files[extension]:
                    for file in self._extension_folder_files[extension][folder]:
                        if _file_extension(file) in AUDIO_TYPE_EXTENSION:
                            self._opened[file] = self._open_audio(file)
                            if _file_extension(file) not in self._file_types:
                                self._file_types[''] =
                        elif _file_extension(file) in IMAGE_TYPE_EXTENSION:
                            self._opened[file] = self._parse_image(file)
                        elif _file_extension(file) in DOC_TYPE_EXTENSION:
                            self._opened[file] = self._parse_document(file)
                        else:
                            assert _file_extension(file) in URL_TYPE_EXTENSION
                            self._opened = self._parse_url(file)


    def _parse_audio(self, file_path):
        return NotImplemented

    def _parse_image(self, file_path):
        return NotImplemented

    def _parse_document(self, file_path):
        return NotImplemented

    def _parse_url(self, file_path):
        return NotImplemented