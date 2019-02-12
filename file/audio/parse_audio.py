from pydub import AudioSegment as _aud_seg
from io import BytesIO
import os

__all__ = ['ParseAudio']


class ParseAudio(object):

    @staticmethod
    def mp3(file: str):
        try:
            fh = _aud_seg.from_file(file, format='mp4')
            return fh.get_array_of_samples().tolist(), fh.frame_rate
        except Exception as e:
            return file

    @staticmethod
    def wav(file: str):
        try:
            with open(file, 'rb') as f:
                fo = BytesIO(f.read())
            fh = _aud_seg.from_file_using_temporary_files(fo)
            return fh.get_array_of_samples().tolist(), fh.frame_rate
        except Exception as e:
            return file

    def __call__(self, file_name):
        ext = os.path.splitext(file_name)[1][1:]
        if ext == 'wav':
            return self.wav(file_name)
        elif ext == 'mp3':
            return self.mp3(file_name)
        else:
            return file_name

