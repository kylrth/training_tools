"""Utility functions for defining components.

Kyle Roth. 2019-06-22.
"""


from argparse import ArgumentTypeError
from functools import partial
import inspect
import os


def subfunc(func, **kwargs):
    """Create a partial function derived from `func`, with the keyword arguments
    specified in kwargs.

    The __name__ attribute of each subfunction will carry the original __name__ plus the
    keyword arguments separated by underscores.

    Args:
        func (function): function to create a partial from.
        kwargs (dict): each keyword argument value must be a tuple containing values
                       that keyword argument can take on in a call to `func`.
    Returns:
        (functools.partial): callable function with an appropriate __name__ attribute.
    """
    out = partial(func, **kwargs)

    # give the function a __name__ attribute so that it can be recognized by SLURM_gen
    out.__name__ = func.__name__ + "".join("_{}".format(val) for val in kwargs.values())
    # add the docstring
    out.__doc__ = "Apply {name} with kwargs {kw}.\n\n" "Docstring for {name}:\n\n{doc}".format(
        name=func.__name__, kw=kwargs, doc=func.__doc__
    )

    return out


def composition(second, first, multiple_returns=True):
    """Combine two functions so that `composition(a_func, b_func)(x)` is equivalent to
    `a_func(b_func(x))`.

    Args:
        second (function): the second function to be applied to the argument.
        first (function): the first function to be applied to the argument.
        multiple_returns (bool): if True, the outputs of `first` are provided to
                                 `second` as separate parameters using the splat
                                 operator *. The output of `first` must be a tuple for
                                 this to work as expected.
    Returns:
        (function): a function calling `first` on its input, and then `second` on the
                    output of `first`.
    """
    if multiple_returns:

        def composed(*args, **kwargs):
            return second(*first(*args, **kwargs))

    else:

        def composed(*args, **kwargs):
            return second(first(*args, **kwargs))

    # add the docstring
    composed.__doc__ = """Apply `{first_name}`, and then `{second_name}`, returning """
    """the result.

    Docstring of `{first_name}`:

    {first_doc}

    Docstring of `{second_name}`:

    {second_doc}
    """.format(
        first_name=first.__name__,
        second_name=second.__name__,
        first_doc=first.__doc__,
        second_doc=second.__doc__,
    )
    # make the __name__ attribute a concatenation of the two functions' names
    composed.__name__ = "{}_then_{}".format(first.__name__, second.__name__)
    return composed


def typer(func):
    """Decorator for functions that accept strings and attempt to convert them to a
    unique string for every component option for their module.

    These functions accept a string `s` and return a string that's guaranteed to be
    accepted by the same module's `_get` method. If no suitable conversion is found, an
    `argparse.ArgumentTypeError` is raised with a specific message. (For this
    functionality to work, `func` must return None in that case.)

    Args:
        func (function): conversion function with call signature and return value as
                         described above.
    Returns:
        (function): decorated function with appropriate docstring and raised exception.
    """
    component_name = os.path.basename(inspect.stack()[1][0].f_code.co_filename[:-4])  # cut "s.py"

    def wrapper(s):
        out = func(s)
        if out is None:
            raise ArgumentTypeError("{} type not recognized".format(component_name))
        return out

    wrapper.__doc__ = """Convert the string given to a string identifying a {component} returned by
        get_{component}.

        Args:
            s (str): string specifying {article} {component}.
        Returns:
            (str): uniquely identifying string for a {component}.
        """.format(
        component=component_name,
        article="an" if component_name[0] in ("a", "e", "i", "o", "u") else "a",
    )

    return wrapper


def getter(namespace):
    """Create the function that converts strings to the components they represent.

    These functions accept only one string per component. That string must be the name
    it has in the namespace passed to this function. If `s` is not a component
    identifier, an `argparse.ArgumentTypeError` is raised with a specific message. (For
    this functionality to work, `func` must return None in that case.)

    Args:
        namespace (dict): dict of variables defined in a namespace; can be the output of
                          `globals()`.
    Returns:
        (function): getter function with appropriate docstring and raised exception.
    """
    component_name = inspect.stack()[1][0].f_code.co_filename[:-4]  # cut off "s.py"

    def _get(s):
        if s in namespace:
            return namespace[s]
        raise ArgumentTypeError("{} type not recognized".format(component_name))

    _get.__name__ = """Get the {component} specified by the string.

        Args:
            s (str): identifier of {article} {component}.
        Returns:
            : {component} object.
        """.format(
        component=component_name,
        article="an" if component_name[0] in ("a", "e", "i", "o", "u") else "a",
    )

    return _get
