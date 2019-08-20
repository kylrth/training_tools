"""Image resizing methods for use in TensorFlow.

Kyle Roth. 2019-06-29.
"""


from tensorflow.image import ResizeMethod  # pylint: disable=import-error,unused-import

from training_tools.components import _utils


@_utils.typer
def _type(s):
    s = s.lower()
    if s == "bilinear":
        return "BILINEAR"
    if "nearest" in s and "neighbor" in s:
        return "NEAREST_NEIGHBOR"
    if s == "bicubic":
        return "BICUBIC"
    if s == "area":
        return "AREA"
    return None


_get = _utils.getter(ResizeMethod.__dict__)
