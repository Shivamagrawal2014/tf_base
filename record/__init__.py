import os
from enum import Enum
from typing import List
from glob import glob


IMAGE_TYPE_EXTENSION = ['jpeg', 'png']
AUDIO_TYPE_EXTENSION = ['mp3', 'wav']
DOC_TYPE_EXTENSION = ['pdf', 'txt', 'json', 'doc', 'docx']
URL_TYPE_EXTENSION = ['html']


class FileType(Enum):
    AUDIO = 'audio'
    IMAGE = 'image'
    URL = 'html'
    DOC = 'doc'


def _is_string(folder):
    return isinstance(folder, str)


def _is_not_dunder(folder):
    if folder[:2] != '__' and folder[-2:] != '__':
        return True
    return False


def _is_clean_folder(folder):
    try:
        assert _is_string(folder)
        assert _is_not_dunder(folder)
        return True
    except Exception:
        return False


def _file_extension(file_name):
    return os.path.splitext(file_name)[1]
