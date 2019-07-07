"""This module defines models for functions and parameters used during neural network training.

Each module in this package represents a model, and the file must be named "<model_type>s.py". (Remember the s!) The
module must include a _type function and a _get function. Both should accept a single argument: a string specifying the
object to get. _type returns a string uniquely identified by the object requested. This is used by argparse as a 'type',
to perform command line argument checking. _get returns the actual object requested, and only accepts strings returned
by _type. See the preprocessors module for an example.

Kyle Roth. 2019-06-21.
"""


import os


# dynamically import _type and _get from every submodule
for f in os.listdir(os.path.dirname(__file__)):
    if f == '__init__.py':
        continue
    if f.endswith('s.py') and ' ' not in f and f != '_utils.py':
        # pylint: disable=exec-used  # I know what I'm doing!
        exec('from .{} import _type as {}'.format(f[:-3], f[:-4]))
        exec('from .{} import _get as get_{}'.format(f[:-3], f[:-4]))
        # for example:
        # from preprocessors import _type as preprocessor
        # from preprocessors import _get as get_preprocessor
