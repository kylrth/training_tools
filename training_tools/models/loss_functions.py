"""You guessed it. Loss functions for training neural networks in TensorFlow.

Kyle Roth. 2019-06-24.
"""


from tensorflow.losses import mean_squared_error, softmax_cross_entropy  # pylint: disable=import-error,unused-import
from models import _utils


@_utils.typer
def _type(s):
    s = s.lower()
    if s in ('tf.losses.mean_squared_error', 'mse', 'mean_squared_error'):
        return 'mean_squared_error'
    if s in ('tf.losses.softmax_cross_entropy', 'cross_entropy', 'cross entropy'):
        return 'softmax_cross_entropy'
    return None


_get = _utils.getter(globals())
