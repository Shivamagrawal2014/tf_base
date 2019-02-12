from typing import List
from file.open_file import OpenFile
from file import AUDIO_TYPE_EXTENSION
from file.audio.parse_audio import ParseAudio


class OpenAudio(OpenFile):

    def __init__(self, folder: str, extensions: List[str]):
        assert all((ext in AUDIO_TYPE_EXTENSION for ext in extensions))
        super(OpenAudio, self).__init__(folder=folder, extensions=extensions)
        self.__parse_audio = ParseAudio()

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
        audio_content = self.__parse_audio(file_path)
        if audio_content != file_path: pcm, frm_rate = audio_content
        else: pcm, frm_rate = [[0]], 0
        return self._protofy_audio(pcm, frm_rate)

    def _protofy_audio(self, pcm, frame_rate):
        return  NotImplemented


if __name__ == '__main__':
