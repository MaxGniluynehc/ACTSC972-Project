import numpy as np


def moving_average(array, window_size):
    s = np.cumsum(array)
    s[window_size:] = s[window_size:] - s[:-window_size]
    s[window_size-1:] = s[window_size-1:]/window_size
    s[:(window_size - 1)] = s[:(window_size - 1)] / np.arange(1, window_size)
    return s




