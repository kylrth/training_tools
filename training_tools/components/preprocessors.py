"""You guessed it. Preprocessors for the field profile data.

Kyle Roth. 2019-06-21.
"""


from PIL import Image
import numpy as np

from training_tools.components import _utils


def rotate_samples(y):
    """Rotate data so that the phase is zero

    Args:
        y (tuple): target images for the mode and all six components, each of shape
                   (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): The data in the real plane
    """
    phase = np.angle(y).mean()
    image = y * np.exp(-phase * 1j)
    return np.real(image)


def H_x(x, y):
    """Get just the H field x-component.

    Args:
        x: predictive features for a target image.
        y (tuple): targets for the mode and all six components, each of shape
                   (1, n, m).
    Returns:
        : the same list of predictive features.
        (np.ndarray): target image of shape (1, n, m).
    """
    rotated = rotate_samples(y[1][0])
    return x, np.expand_dims(rotated, 0)


def H_x_resample(x, y, resize):
    """Get just the H field x-component, and resample the target image to shape
    (1, resize, resize).

    Args:
        x: predictive features for a target image.
        y (tuple): target images for the mode and all six components, each of shape
                   (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): target image, resampled to shape (1, resize, resize).
    """
    resized = np.array(
        Image.fromarray(rotate_samples(y[1][0])).resize((resize, resize))
    )
    return x, np.expand_dims(resized, 0)


# get specific versions of H_x_resample
H_x_resample_8 = _utils.subfunc(H_x_resample, resize=8)
H_x_resample_16 = _utils.subfunc(H_x_resample, resize=16)
H_x_resample_32 = _utils.subfunc(H_x_resample, resize=32)
H_x_resample_64 = _utils.subfunc(H_x_resample, resize=64)
H_x_resample_128 = _utils.subfunc(H_x_resample, resize=128)
H_x_resample_256 = _utils.subfunc(H_x_resample, resize=256)


def H_y_resample(x, y, resize):
    """Get just the H field y-component, and resample the target image to shape
    (1, resize, resize).

    Args:
        x: predictive features for a target image.
        y (tuple): target images for the mode and all six components, each of shape
                   (1, n > resize, m > resize).
    Returns:
        : the same list of predictive features.
        (np.ndarray): target image, resampled to shape (1, resize, resize).
    """
    resized = np.array(
        Image.fromarray(rotate_samples(y[2][0])).resize((resize, resize))
    )
    return x, np.expand_dims(resized, 0)


# get specific versions of H_x_resample
H_y_resample_8 = _utils.subfunc(H_y_resample, resize=8)
H_y_resample_16 = _utils.subfunc(H_y_resample, resize=16)
H_y_resample_32 = _utils.subfunc(H_y_resample, resize=32)
H_y_resample_64 = _utils.subfunc(H_y_resample, resize=64)
H_y_resample_128 = _utils.subfunc(H_y_resample, resize=128)
H_y_resample_256 = _utils.subfunc(H_y_resample, resize=256)


def E_x_resample(x, y, resize):
    """Get just the E field x-component, and resample the target images to shape
    (1, resize, resize).

    Args:
        x: predictive features for each target image.
        y (tuple): target images for the mode and all six components, each of shape
                   (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): target images, resampled so that each is of shape
                      (1, resize, resize).
    """
    resized = np.array(
        Image.fromarray(rotate_samples(y[4][0])).resize((resize, resize))
    )
    return x, np.expand_dims(resized, 0)


# get specific versions of E_x_resample
E_x_resample_8 = _utils.subfunc(E_x_resample, resize=8)
E_x_resample_16 = _utils.subfunc(E_x_resample, resize=16)
E_x_resample_32 = _utils.subfunc(E_x_resample, resize=32)
E_x_resample_64 = _utils.subfunc(E_x_resample, resize=64)
E_x_resample_128 = _utils.subfunc(E_x_resample, resize=128)
E_x_resample_256 = _utils.subfunc(E_x_resample, resize=256)


def E_y_resample(x, y, resize):
    """Get just the E field y-component, and resample the target images to shape
    (1, resize, resize).

    Args:
        x: predictive features for each target image.
        y (tuple): target images for the mode and all six components, each of shape
                   (1, n > resize, m > resize).
        resize (int): side length for a single output image.
    Returns:
        : the same list of predictive features.
        (np.ndarray): target images, resampled so that each is of shape
                      (1, resize, resize).
    """
    resized = np.array(
        Image.fromarray(rotate_samples(y[5][0])).resize((resize, resize))
    )
    return x, np.expand_dims(resized, 0)


# get specific versions of E_y_resample
E_y_resample_8 = _utils.subfunc(E_y_resample, resize=8)
E_y_resample_16 = _utils.subfunc(E_y_resample, resize=16)
E_y_resample_32 = _utils.subfunc(E_y_resample, resize=32)
E_y_resample_64 = _utils.subfunc(E_y_resample, resize=64)
E_y_resample_128 = _utils.subfunc(E_y_resample, resize=128)
E_y_resample_256 = _utils.subfunc(E_y_resample, resize=256)


@_utils.typer
def _type(s):
    if s[13:] in ("8", "16", "32", "64", "128", "256"):
        if s.lower().startswith("h_x_resample_"):
            return "H_x_resample_{}".format(s[13:])
        if s.lower().startswith("h_y_resample_"):
            return "H_y_resample_{}".format(s[13:])
        if s.lower().startswith("e_x_resample_"):
            return "E_x_resample_{}".format(s[13:])
        if s.lower().startswith("e_y_resample_"):
            return "E_y_resample_{}".format(s[13:])
    return None


_get = _utils.getter(globals())
