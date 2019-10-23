"""You guessed it. Preprocessors for the taper modeling data.

Kyle Roth. 2019-06-21.
"""


import numpy as np
from PIL import Image

from training_tools.models import _utils


def H_x_resample(x, y, resize):
    """Get just the H field x-component, and resample the target images to 128x128.

    Args:
        x: predictive features for each target image.
        y (tuple): target images for the mode and all six components, each of shape (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): target images, resampled so that each is of shape (1, resize, resize).
    """
    return x, np.expand_dims(np.array(Image.fromarray(np.real(y[1][0])).resize((resize, resize))), 0)


# get specific versions of H_x_resample
H_x_resample_8 = _utils.subfunc(H_x_resample, resize=8)
H_x_resample_16 = _utils.subfunc(H_x_resample, resize=16)
H_x_resample_32 = _utils.subfunc(H_x_resample, resize=32)
H_x_resample_64 = _utils.subfunc(H_x_resample, resize=64)
H_x_resample_128 = _utils.subfunc(H_x_resample, resize=128)
H_x_resample_256 = _utils.subfunc(H_x_resample, resize=256)


def rotate_samples(x,y):
    """Rotate data so that the phase is zero

    Args:
        x: predictive features for each target image.
        y (tuple): target images for the mode and all six components, each of shape (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): The data in the real plane
    """

    phase = np.angle(y).mean()
    image = y * np.exp(-phase * 1j)
    return x, np.real(image)


@_utils.typer
def _type(s):
    if s.lower().startswith('h_x_resample_') and s[13:] in ('8', '16', '32', '64', '128', '256'):
        return 'H_x_resample_{}'.format(s[13:])
    return None


_get = _utils.getter(globals())
