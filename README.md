# training_tools

**This repo is archived. I created it when I was young and naive about good code design and what the deep learning frameworks offered out of the box.**

Tools I use for training and testing neural networks.

## Installation

```bash
pip install -e .  # don't forget the period
```

## Usage

### Architectures

Using the architectures is pretty straightforward.

```python
from training_tools.architectures import resize_conv

# ... load a mag configuration from the command line

# create the model
model = resize_conv.create_model(training_data, config.model, config._verbose)
```

More information on mag [here](https://github.com/ex4sperans/mag/).

### Components

The components module was designed for use with [`argparse`](https://docs.python.org/3/library/argparse.html). Argparse provides a `type` keyword argument when adding a new argument, and so if you'd like the argument to be an `int`, you specify `int`:

```python
parser.add_argument("--njobs", type=int, help="number of jobs")
```

Similarly, if you'd like the argument to be a TensorFlow loss function, you can do the following:

```python
from training_tools import components, utils

parser = utils.BetterArgParser()
parser.add_argument(
    arg="loss",
    argtype=components.loss_function,
    help_str="loss function for training",
    default="mean_squared_error",
)
```

Speaking of which, `utils.py` provides a `BetterArgParser` suited to working with `mag`, which we used in the code block above. Check out its docstring for more details.
