from typing import List
from file.open_file import OpenFile
from file import AUDIO_TYPE_EXTENSION
from file.audio.parse_audio import ParseAudio


class OpenAudio(OpenFile):

    def __init__(self, folder: str, extensions: List[str]):
        super(OpenAudio, self).__init__(folder=folder, extensions=extensions)
        self._parse_audio = ParseAudio()

    @property
    def open_audio(self):
        return self._open_files()

    @property
    def audio_folder_files(self):
        return self._extension_folder_files

    @property
    def audio_folders(self):
        return self._extensions_to_folder

    @property
    def audio_files(self):
        return self._extensions_to_files

    def _parse_audio(self, file_path):
        image, shape = self._parse_audio(file_path)
        return self._protofy_image(image, shape)

    def _protofy_audio(self, pcm, ):