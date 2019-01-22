import os
from typing import List
from glob import glob
from file import FileType
from file import (AUDIO_TYPE_EXTENSION,
                  IMAGE_TYPE_EXTENSION,
                  DOC_TYPE_EXTENSION,
                  URL_TYPE_EXTENSION,
                  SQL_TYPE_EXTENSION)

from file import _is_clean_folder, _file_extension


class OpenFile(object):

    def __init__(self, folder: str, extensions: List[str]):
        self._folder = folder
        self._extensions = extensions
        self._opened = False
        self._file_types = None
        self._extension_folder_files = None
        self._extension_folder = None
        self._extension_files = None
        self(folder, extensions)

    @staticmethod
    def _files_with_extension(folder: str, extension: str):
        _folder_dict = dict()
        for root, subfolders, files in os.walk(folder):
            for sub_folder in subfolders:
                if _is_clean_folder(sub_folder):
                    found_files = glob(os.path.join(
                        os.path.join(root, sub_folder), '.'.join(['*', extension])))
                    if any(found_files):
                        _folder_dict[sub_folder] = found_files

            if any(files):
                found_files = [file for file in files
                                      if file.endswith(extension)]
                if any(found_files):
                    _folder_dict[root] = found_files
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
        if self._file_types is None:
            self._file_types = dict()

        if self._opened is False:
            self._opened = dict()
            for extension in self._extension_folder_files:
                for folder in self._extension_folder_files[extension]:
                    for file in self._extension_folder_files[extension][folder]:

                        # Audio Handling
                        if _file_extension(file) in AUDIO_TYPE_EXTENSION:
                            if FileType.AUDIO not in self._file_types:
                                self._file_types[FileType.AUDIO] = list()
                            if _file_extension(file) not in self._file_types[FileType.AUDIO]:
                                self._file_types[FileType.AUDIO].append(_file_extension(file))
                            self._opened[file] = self._open_audio(os.path.join(folder, file))

                        # Image Handling
                        if _file_extension(file) in IMAGE_TYPE_EXTENSION:
                            if FileType.IMAGE not in self._file_types:
                                self._file_types[FileType.IMAGE] = list()
                            if _file_extension(file) not in self._file_types[FileType.IMAGE]:
                                self._file_types[FileType.IMAGE].append(_file_extension(file))
                            self._opened[file] = self._open_image(os.path.join(folder, file))

                        # Document Handling
                        elif _file_extension(file) in DOC_TYPE_EXTENSION:
                            if FileType.DOC not in self._file_types:
                                self._file_types[FileType.DOC] = list()
                            if _file_extension(file) not in self._file_types[FileType.DOC]:
                                self._file_types[FileType.DOC].append(_file_extension(file))
                            self._opened[file] = self._open_document(os.path.join(folder, file))

                        # URL Handling
                        elif _file_extension(file) in URL_TYPE_EXTENSION:
                            if FileType.URL not in self._file_types:
                                self._file_types[FileType.URL] = list()
                            if _file_extension(file) not in self._file_types[FileType.URL]:
                                self._file_types[FileType.URL].append(_file_extension(file))
                            self._opened[file] = self._open_url(os.path.join(folder, file))

                        # SQL handling
                        elif _file_extension(file) in SQL_TYPE_EXTENSION:
                            if FileType.SQL not in self._file_types:
                                self._file_types[FileType.SQL] = list()
                            if _file_extension(file) not in self._file_types[FileType.SQL]:
                                self._file_types[FileType.SQL].append(_file_extension(file))
                            self._opened[file] = self._open_sql(os.path.join(folder, file))

                        else:
                            continue
        return self._opened

    def _open_audio(self, file_path):
        return self._parse_audio(file_path)

    def _open_image(self, file_path):
        return self._parse_image(file_path)

    def _open_document(self, file_path):
        return self._parse_document(file_path)

    def _open_url(self, url_file):
        return self._parse_url(url_file)

    def _open_sql(self, sql_file):
        return self._parse_sql(sql_file)

    def _parse_audio(self, file_path):
        return NotImplemented

    def _parse_image(self, file_path):
        return NotImplemented

    def _parse_document(self, file_path):
        return NotImplemented

    def _parse_url(self, file_path):
        return NotImplemented

    def _parse_sql(self, sql_file):
        return NotImplemented

    def __call__(self, folder: str, extensions: List[str]):
        _ = self._find_files(folder, extensions)
        _ = self._extensions_to_files
        _ = self._extensions_to_folder
