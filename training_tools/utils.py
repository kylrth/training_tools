"""Utility functions for experiments.

Kyle Roth. 2019-03-26.
"""


import argparse
import inspect
import os
import time

from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np


def v_print(verbose, s):
    """If verbose is True, print the string, prepending with the current timestamp and the function
    that called this function.

    Useful for printing debugging information.

    Args:
        verbose (bool): whether to print.
        s (str): string to print. (This will be passed to str.format, so it could be
                 anything with a __repr__ function.)
    """
    if verbose:
        caller = inspect.currentframe().f_back.f_code.co_name
        print("{:.6f}: ({}) {}".format(time.time(), caller, s))


def sanitize_word(s):
    """Ensure that a string is in fact a single word with alphanumeric characters.

    Useful for avoiding code injection.

    Args:
        s (str): string to be inserted into code.
    Returns:
        (str): the same string, stripped of whitespace.
    Raises:
        ValueError: the string contained a character that is not alphanumeric.
    """
    s = s.strip()
    if not s.isalnum():
        raise ValueError('unexpected string "{}" received when a single word was expected')
    return s


def natural_number(s):
    """Ensure the integer received is greater than or equal to zero.

    This can be used as a "type" specified to argparse, like so:

        parser.add_argument("--njobs", type=natural_number, help="the number of jobs to submit")

    Args:
        s (str): string (hopefully) containing an integer greater than zero.
    Returns:
        (int): integer greater than zero.
    Raises:
        (argparse.ArgumentTypeError): raised if integer retrieved from string is less
                                      than zero.
    """
    s = int(s)
    if s >= 0:
        return s
    raise argparse.ArgumentTypeError("integer is less than zero")


def save_image(image, filepath):
    """Plot an image using a diverging color scheme with a colorbar.

    The call to plt.clf() reduces the overhead of calling this function many times.

    Args:
        image (np.ndarray or tf tensor): image to be plotted.
        filepath (str): path to the location where the image should be written.
    """
    image = np.array(image)
    ax = plt.gca()
    im = ax.imshow(image, cmap=cm.get_cmap("BrBG"))  # RdBu also a good choice
    plt.colorbar(im)
    plt.savefig(filepath)
    plt.clf()


class QualitativeTest:
    """Apply the model to create some images, storing the results in the experiment directory under
    samples/."""

    def __init__(self, exp_dir, features, labels, model, scaler=None, to_save=2):
        """Store expected output and initial results.

        Args:
            exp_dir (str): path to experiment directory.
            features: features that the model takes as input.
            labels: target for the model for the given features.
            model: model to apply to features.
            scaler: sklearn StandardScaler object used to normalize and de-normalize data.
            to_save (int): number of sample images to save.
        """
        self.sample_dir = os.path.join(exp_dir, "samples")
        self.features = features
        self.scaler = scaler
        self.to_save = to_save

        os.makedirs(self.sample_dir, exist_ok=True)

        if self.scaler is not None:
            # de-normalize the labels
            img_shape = labels[0].shape
            labels = self.scaler.inverse_transform(
                [img.numpy().flatten() for img in labels]
            ).reshape((labels.shape[0], *img_shape))

        self.labels = labels

        # store the expected output
        for i in range(self.to_save):
            save_image(self.labels[i], os.path.join(self.sample_dir, "{}_expected.png".format(i)))

        # store the untrained results
        self.test(model, 0)

    def test(self, model, epochs):
        """Apply the new model to the stored features to get new sample output.

        Args:
            model: model to apply to features.
            epochs (int): current number of epochs of training. Used in the output file
                          names.
        """
        # predict
        predicted = model(self.features)

        if self.scaler is not None:
            # de-normalize
            img_shape = predicted[0].shape
            predicted = self.scaler.inverse_transform(
                [img.numpy().flatten() for img in predicted]
            ).reshape((predicted.shape[0], *img_shape))

        for i in range(self.to_save):
            # save prediction
            save_image(
                predicted[i], os.path.join(self.sample_dir, "{}_epochs_{}.png".format(i, epochs))
            )
            # save plot of absolute error
            save_image(
                np.abs(predicted[i] - self.labels[i]),
                os.path.join(self.sample_dir, "{}_epochs_{}_abserr.png".format(i, epochs)),
            )


def conv_output(kernel, stride):
    """Define a function that calculates the output shape of the convolution given the
    input shape.

    Args:
        kernel (int): width of kernel.
        stride (int): length of stride.
    Returns:
        (function): function accepting a call signature (dim, times) and returning the
                    output dimension.
    """
    const = 1 - kernel / stride
    stride_inv = 1 / stride

    def func(dim, times=1):
        # one iteration is (dim - kernel) / stride + 1, so times iterations is
        return dim / stride ** times + const * sum(stride_inv ** power for power in range(times))

    # update the function name and docstring
    func.__name__ = "conv_output_{}_{}".format(kernel, stride)
    func.__doc__ = """Calculate the output shape given the input shape.

        Kernel size: {}
        Stride: {}

        Args:
            dim (int): length of an input dimension.
            times (int): number of layers in sequence. Default is one layer.
        Returns:
            (int): length of the corresponding output dimension.
        """.format(
        kernel, stride
    )

    return func


def conv_transpose_output(kernel, stride):
    """Define a function that calculates the output shape of applying a transpose
    convolution multiple times, given the input shape.

    Args:
        kernel (int): width of kernel.
        stride (int): length of stride.
    Returns:
        (function): function accepting a call signature (dim, times) and returning the
                    output dimension.
    """
    const = max(kernel - stride, 0)

    def func(dim, times=1):
        # one iteration is dim * stride + max(kernel - stride, 0)
        return dim * stride ** times + const * sum(stride ** power for power in range(times))

    # update the function name and docstring
    func.__name__ = "conv_output_{}_{}".format(kernel, stride)
    func.__doc__ = """Calculate the output shape given the input shape.

        Kernel size: {}
        Stride: {}

        Args:
            dim (int): length of an input dimension.
            times (int): number of layers in sequence. Default is one layer.
        Returns:
            (int): length of the corresponding output dimension.
        """.format(
        kernel, stride
    )

    return func


class BetterArgParser:
    """Like argparse.ArgumentParser, but with value checking and attribute hierarchies that make it
    easy to pass values to mag."""

    def __init__(self, description=None):
        """Create the internal ArgumentParser object, and other attribute containers."""
        if description is None:
            self._argparse = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        else:
            self._argparse = argparse.ArgumentParser(
                description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        self._args = None  # where the argument values will be placed after calling parse_args

        # map from argument names to their full hierarchical, user-specified names
        self.full_names = {}

        # map from argument names to checkers
        self.checkers = {}

        # arguments which should be hidden in the mag config (prefixed with "_")
        self.hidden = set()

    def add_argument(self, arg, argtype, help_str, default=None, hidden=False):
        """Add an argument to be specified on the command line.

        Args:
            arg (str): name of named argument (not positional). If hierarchy is desired, specify by
                       prefixing like "parent_name.attribute". The argument will still be
                       "--attribute ATTRIBUTE". Later, this can be read from the mag Experiment
                       object as exp.parent_name.attribute.
            argtype (callable): function accepting a string and returning an object of the correct
                                type. This could be a factory function like `int`, or a validation
                                function created by the user. Must return a str or Number for mag to
                                be able to record it.
            help_str (str): the help string displayed when `-h` is specified. Here we require it to
                            put the user in the good habit of documenting these arguments.
            default: the value the argument takes on if it is not specified. If this is object is
                     callable, it must be a function that performs desired post-processing for
                     defaults. The call signature must accept an object containing the other
                     argument values as attributes. The function must return a new value for the
                     argument. For example, if the default value of an argument `argue` depends on
                     the value of another argument `other`, the argument passed here could look like
                     this:

                                    def argue_default(args):
                                        '''Validate the `argue` argument.'''
                                        if args.other == "sibling":
                                            return True
                                        return False

                     The callable is only called if the argument wasn't specified in the command
                     line.
            hidden (bool): whether the corresponding mag configuration attribute should be prefixed
                           with an underscore. This means the argument doesn't affect the experiment
                           output and shouldn't be included in the experiment directory name.
        """
        short = arg.split(".")[-1]

        if callable(default):
            # don't specify a default, and use the checking function later
            self._argparse.add_argument("--" + short, type=argtype, help=help_str)
            self.checkers[short] = default
        else:
            # use the default
            self._argparse.add_argument("--" + short, type=argtype, default=default, help=help_str)

        self.full_names[short] = arg
        if hidden:
            self.hidden.add(short)

    def add_arguments(self, arguments, parent=None):
        """Add all the arguments defined in the list by calling `add_argument` on each element of
        `arguments`.

        Each element of `arguments` must unpack to a valid set of arguments to `add_argument`.

        Args:
            arguments (list): list of lists, each containing arguments to pass to `add_argument`.
            parent (str): if specified, all arguments will be added to the mag config under this
                          parent.
        """
        if parent is None:
            for argument in arguments:
                self.add_argument(*argument)
        else:
            for argument in arguments:
                self.add_argument(parent + "." + argument[0], *argument[1:])

    def add_flag(self, arg, help_str, hidden=False):
        """Add an argument that has a value of True when present and False when not present.

        Args:
            arg (str): name of named argument (not positional). If hierarchy is desired, specify by
                       prefixing with "parent_name.attribute". The argument will still be
                       "--attribute value". Later, this can be read from the mag Experiment object
                       as exp.parent_name.attribute.
            help_str (str): the help string displayed when `-h` is specified. Here we require it to
                            put the user in the good habit of documenting these arguments.
            hidden (bool): whether the corresponding mag configuration attribute should be prefixed
                           with an underscore. This implies the argument doesn't affect the
                           experiment output and shouldn't be included in the experiment directory
                           name.
        """
        short = arg.split(".")[-1]

        self._argparse.add_argument("--" + short, action="store_true", help=help_str)

        self.full_names[short] = arg
        if hidden:
            self.hidden.add(short)

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
        """(dict) The mag configuration object produced by the parsed arguments."""
        out = {}

        for short_name in self.full_names:
            hierarchy = self.full_names[short_name].split(".")

            # mark hidden args as hidden
            if short_name in self.hidden:
                hierarchy[-1] = "_" + hierarchy[-1]

            # create sub-dicts for each parent name specified
            current_dict = out
            for parent in hierarchy[:-1]:
                if parent not in current_dict:
                    current_dict[parent] = {}
                current_dict = current_dict[parent]

            # store the value here
            current_dict[hierarchy[-1]] = getattr(self._args, short_name)

        return out
