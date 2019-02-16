from typing import Union, List
from python_speech_features import mfcc


class PCMToMFCC(object):

    def __init__(self, pcm: Union[List[int]], sample_rate: int):
        self(pcm, sample_rate)

    def __call__(self, pcm, sample_rate):
        return mfcc(pcm, sample_rate)



