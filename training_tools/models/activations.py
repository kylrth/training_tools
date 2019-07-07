"""You guessed it. Activation functions for training neural networks in TensorFlow.

Kyle Roth. 2019-06-24.
"""


from training_tools.models import _utils


# add these names to the namespace so they can be returned by _get
relu = 'relu'
tanh = 'tanh'


@_utils.typer
def _type(s):
    s = s.lower()
    if s in (relu, tanh):
        return s
    return None


_get = _utils.getter(globals())
