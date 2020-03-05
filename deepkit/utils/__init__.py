import io
import os

import numpy as np


def in_self_execution():
    """
    Returns true if the script is directly executed without `deepkit` cli.
    """
    return 'DEEPKIT_JOB_ACCESSTOKEN' not in os.environ


def array_to_img(x, scale=True):
    """
    x should be shape (channels, width, height)
    """
    from PIL import Image
    if x.ndim != 3:
        raise Exception('Unsupported shape : ', str(x.shape), '. Need (channels, width, height)')
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[0] == 3:
        # RGB
        if x.dtype != 'uint8':
            x = x.astype('uint8')
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[0] == 1:
        # grayscale
        if x.dtype != 'uint8':
            x = x.astype('uint8')
        return Image.fromarray(x.reshape(x.shape[1], x.shape[2]), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[0])


def numpy_to_binary(array):
    buffer = io.BytesIO()

    if isinstance(array, np.ndarray):
        np.save(buffer, array)

    return buffer.getvalue()


def get_parameter_by_path(dictionary, path):
    if not dictionary:
        return None

    if path in dictionary:
        return dictionary[path]

    current = dictionary

    for item in path.split('.'):
        if item not in current:
            return None

        current = current[item]

    return current
