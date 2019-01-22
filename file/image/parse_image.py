import cv2 as cv
from typing import Tuple
from matplotlib import pyplot as plt


class ParseImage(object):

    def __init__(self, size: Tuple[int, int, int] = None, show=False):
        self._read = cv.imread
        if size is not None:
            if len(size) > 2:
                size = size[:2]
        self._resize = size
        self._show_image = show

    def read(self, image):
        return self._read(image)

    def __call__(self, image):
        image = self.read(image)
        shape = image.shape
        if self._resize is not None:
            if shape != self._resize:
                image = cv.resize(image, self._resize)
        if self._show_image:
            plt.imshow(image, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        return image, image.shape
