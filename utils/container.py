import sys


class LightList(object):

    def __init__(self, iterator):
        self._element = iter(iterator)

    def __iter__(self):
        try:
            return self._element
        except StopIteration:
            return

