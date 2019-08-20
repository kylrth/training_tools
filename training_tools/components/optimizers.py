"""You guessed it. Optimization algorithms for training neural networks in TensorFlow.

Kyle Roth. 2019-06-24.
"""


# pylint: disable=import-error,unused-import,wrong-import-order
from tensorflow.train import AdamOptimizer, GradientDescentOptimizer

from training_tools.components import _utils


@_utils.typer
def _type(s):
    s = s.lower()
    if "gradient" in s and "descent" in s:
        return "GradientDescentOptimizer"
    if s in ("adam", "adamoptimizer"):
        return "AdamOptimizer"
    return None


_get = _utils.getter(globals())
