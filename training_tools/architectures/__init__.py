"""If you thought these were neural network architectures for TensorFlow, you'd be right. All of these are written for
eager mode.

Each module in this package represents an architecture, and must define a _get function that accepts a tf.Dataset that
the model will be trained on. The idea is that these architectures can be defined according to the shape of the data, so
the user won't need to finagle with dimensions as much.

Kyle Roth. 2019-06-08.
"""


import os
from training_tools.utils import sanitize_word


# dynamically import _type and _get from every submodule
for f in os.listdir(os.path.dirname(__file__)):
    if f == '__init__.py':
        continue
    if f.endswith('s.py') and ' ' not in f and f != '_utils.py':
        # pylint: disable=exec-used  # I know what I'm doing!
        exec('from training_tools.architectures.{} import _type as {}'.format(f[:-3], f[:-4]))
        exec('from training_tools.architectures.{} import _get as get_{}'.format(f[:-3], f[:-4]))
        # for example:
        # from preprocessors import _type as preprocessor
        # from preprocessors import _get as get_preprocessor
