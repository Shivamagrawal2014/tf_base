from typing import Union, List
from python_speech_features import mfcc


class PCMToMFCC(object):

    def __init__(self, pcm: Union[List[int]], sample_rate: int):
        return mfcc(pcm, sample_rate)

