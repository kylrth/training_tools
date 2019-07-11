"""Utility functions for experiments.

Kyle Roth. 2019-03-26.
"""


import argparse
import os

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def str2bool(s):
    """Convert a string to boolean. Used by argparse to convert commands from CLI.

    Args:
        s (str): string from CLI.
    Returns:
        (bool): truth value implied from string.
    Raises:
        (argparse.ArgumentTypeError): raised if string does not imply a boolean value.
    """
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def natural_number(s):
    """Ensure the integer received is greater than or equal to zero.

    Args:
        s (str): string from CLI.
    Returns:
        (int): integer greater than zero.
    Raises:
        (argparse.ArgumentTypeError): raised if integer retreived from string is less than zero.
    """
    s = int(s)
    if s >= 0:
        return s
    raise argparse.ArgumentTypeError('integer is less than zero')


def save_image(image, filepath):
    """Plot an image taken from the dataset, and save it to a file

    Args:
        image (np.ndarray or tf tensor): grayscale image to be plotted (reshapeable to square).
        filepath (str): path to the location where the image should be written.
    """
    image = np.array(image)
    d = int(np.sqrt(image.size))
    fig, ax = plt.subplots()
    try:
        im = ax.imshow(np.reshape(image, (d, d)), cmap=cm.get_cmap('BrBG'))  # RdBu also a good choice
        fig.colorbar(im, ax=ax)
        plt.savefig(filepath)
    except ValueError as e:
        if 'cannot reshape array' in str(e):
            raise ValueError('image of shape {} cannot be plotted as a square'.format(image.shape))
        raise
    finally:
        plt.close(fig)


class QualitativeTest:
    """Apply the model to create some images, storing the results in the experiment directory under samples/."""
    def __init__(self, exp_dir, features, labels, model, to_save=2):
        """Store expected output and initial results.

        Args:
            exp_dir (str): path to experiment directory.
            features: features that the model takes as input.
            labels: target for the model for the given features.
            model: model to apply to features.
            to_save (int): number of results to save.
        """
        self.sample_dir = os.path.join(exp_dir, 'samples')
        self.features = features
        self.to_save = to_save

        os.makedirs(self.sample_dir, exist_ok=True)

        # store the expected output
        for i in range(self.to_save):
            plt.figure()
            save_image(labels[i], os.path.join(self.sample_dir, '{}_expected.png'.format(i)))

        # store the untrained results
        self.test(model, 0)

    def test(self, model, epochs):
        """Apply the model to the stored features to get new sample output.

        Args:
            model: model to apply to features.
            epochs (int): current number of epochs of training. Used in the output file names.
        """
        predicted = model(self.features)
        for i in range(self.to_save):
            plt.figure()
            save_image(predicted[i], os.path.join(self.sample_dir, '{}_epochs_{}.png'.format(i, epochs)))


def conv_output(kernel, stride):
    """Define a function that calculates the output shape of the convolution given the input shape.

    Args:
        kernel (int): width of kernel.
        stride (int): length of stride.
    Returns:
        (function): function accepting a call signature (dim1, dim2) and returning (dim1, dim2).
    """
    def func(dim):
        return int((dim - kernel) / stride + 1)

    func.__name__ = 'conv_output_{}_{}'.format(kernel, stride)
    func.__doc__ = """Calculate the output shape given the input shape.

        Kernel size: {}
        Stride: {}

        Args:
            dim (int): length of an input dimension.
        Returns:
            (int): length of the corresponding output dimension.
        """.format(kernel, stride)
    return func


def conv_transpose_output(kernel, stride):
    """Define a function that calculates the output shape of the transpose convolution given the input shape.

    Args:
        kernel (int): width of kernel.
        stride (int): length of stride.
    Returns:
        (function): function accepting a call signature (dim1, dim2) and returning (dim1, dim2).
    """
    def func(dim):
        return (dim - 1) * stride + kernel

    func.__name__ = 'conv_output_{}_{}'.format(kernel, stride)
    func.__doc__ = """Calculate the output shape given the input shape.

        Kernel size: {}
        Stride: {}

        Args:
            dim (int): length of an input dimension.
        Returns:
            (int): length of the corresponding output dimension.
        """.format(kernel, stride)
    return func


class BetterArgParser:
    """Like argparse.ArgumentParser, but with value checking and attribute hierarchies for easy passing to mag."""
    def __init__(self):
        """Create the internal ArgumentParser object, and other attribute containers."""
        self._argparse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._args = None  # where the argument values will be placed after calling parse_args

        # map from argument names to their full hierarchical, user-specified names
        self.full_names = {}

        # map from argument names to checkers
        self.checkers = {}

        # arguments which should be hidden in the mag config (prefixed with an underscore)
        self.hidden = set()

    def add_argument(self, arg, argtype, help_str, default=None, hidden=False):
        """Add an argument to be specified on the command line.

        Args:
            arg (str): name of named argument (not positional). If hierarchy is desired, specify by prefixing like
                       "parent_name.attribute". The argument will still be "--attribute value". Later, this can be read
                       from the mag Experiment object as exp.parent_name.attribute.
            argtype (callable): function accepting a string and returning an object of the correct type. This could be a
                                factory function like `int`, or a validation function created by the user. Must return a
                                str or Number for mag to be able to record it.
            help_str (str): the help string displayed when `-h` is specified. Here we require it to put the user in the
                            good habit of documenting these arguments.
            default: the value the argument takes on if it is not specified. If this is object is callable, it must be a
                     function that performs any desired post-processing for defaults. The call signature must accept an
                     object containing the other argument values as attributes. The function must return a new value for
                     the argument. For example, if the default value of an argument `argue` depends on the value of
                     another argument `other`, the argument passed here could look like this:

                                    def argue_default(args):
                                        '''Validate the `argue` argument.'''
                                        if args.other == 'sibling':
                                            return True
                                        return False

                     The callable is only called if the argument wasn't specified in the command line.
            hidden (bool): whether the corresponding mag configuration attribute should be prefixed with an underscore.
                           This means the argument doesn't affect the experiment output and shouldn't be included in the
                           experiment directory name.
        """
        short_name = arg.split('.')[-1]

        if callable(default):
            # don't specify a default, and use the checking function later
            self._argparse.add_argument('--' + short_name, type=argtype, help=help_str)
            self.checkers[short_name] = default
        else:
            # use the default
            self._argparse.add_argument('--' + short_name, type=argtype, default=default, help=help_str)

        self.full_names[short_name] = arg
        if hidden:
            self.hidden.add(short_name)

    def add_arguments(self, arguments, parent=None):
        """Add all the arguments defined in the list by calling `add_argument` on each element of `arguments`.

        Args:
            arguments (list): list of lists, each containing arguments to pass to `add_argument`.
            parent (str): if specified, all arguments will be added to the mag config under this name.
        """
        if parent is None:
            for argument in arguments:
                self.add_argument(*argument)
        else:
            for argument in arguments:
                self.add_argument(parent + '.' + argument[0], *argument[1:])

    def add_flag(self, arg, help_str, hidden=False):
        """Add an argument that has a value of True when present and False when not present.

        Args:
            arg (str): name of named argument (not positional). If hierarchy is desired, specify by prefixing like
                       "parent_name.attribute". The argument will still be "--attribute value". Later, this can be read
                       from the mag Experiment object as exp.parent_name.attribute.
            help_str (str): the help string displayed when `-h` is specified. Here we require it to put the user in the
                            good habit of documenting these arguments.
            hidden (bool): whether the corresponding mag configuration attribute should be prefixed with an underscore.
                           This means the argument doesn't affect the experiment output and shouldn't be included in the
                           experiment directory name.
        """
        short_name = arg.split('.')[-1]

        self._argparse.add_argument('--' + short_name, action='store_true', help=help_str)

        self.full_names[short_name] = arg
        if hidden:
            self.hidden.add(short_name)

    def parse_args(self, args=None):
        """Parse the arguments and perform checking.

        Args:
            args (str): if specified, parse arguments from the string instead of from sys.argv.
        """
        self._args = self._argparse.parse_args(args=args)

        for arg in self.checkers:
            if getattr(self._args, arg) is None:
                # get the new value from the checker and set it to that
                setattr(self._args, arg, self.checkers[arg](self._args))

    @property
    def mag_config(self):
        """(dict) The mag configuration object."""
        out = {}

        for short_name in self.full_names:
            hierarchy = self.full_names[short_name].split('.')

            # mark hidden args as hidden
            if short_name in self.hidden:
                hierarchy[-1] = '_' + hierarchy[-1]

            # create sub-dicts for each parent name specified
            current_dict = out
            for parent in hierarchy[:-1]:
                if parent not in current_dict:
                    current_dict[parent] = {}
                current_dict = current_dict[parent]

            # store the value here
            current_dict[hierarchy[-1]] = getattr(self._args, short_name)

        return out
