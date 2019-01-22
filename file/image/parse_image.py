import cv2 as cv
from typing import Tuple


class ParseImage(object):

    def __init__(self, size: Tuple[int, int, int] = None):
        self._read = cv.imread
        if size is not None:
            if len(size) > 2:
                size = size[:2]
        self._resize = size

    def read(self, image):
        return self._read(image)

    def __call__(self, image):
        image = self.read(image)
        shape = image.shape
        if self._resize is not None:
            if shape != self._resize:
                image = cv.resize(image, self._resize)
        return image, image.shape
