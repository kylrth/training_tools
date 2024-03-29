"""Neural network with transpose convolutions, or "deconvolutions" as the cool kids say.

Kyle Roth. 2019-07-10.
"""


import numpy as np
import tensorflow as tf

from training_tools import utils


def create_model(dataset, config, verbose):
    """Create sequential network with linear layers at input and output, and enough transpose
    convolutions to get the right size.

    Args:
        dataset (tf.Dataset): the dataset the model will be trained on. Used for determining input
                              and output shape.
        config (mag.config.Config): object with the following attributes:
                                    - 'initial_dense': the shape of the initial dense layer's
                                                       output. The channel dimension must be last.
                                    - 'kernel_size': the kernel size used by the deconvolutions.
                                    - 'stride': the stride length used by the deconvolutions.
                                    - 'layers': the number of deconvolutions to use. If negative,
                                                the number is chosen that produces an output of
                                                similar size to the final output, not including the
                                                filter dimension of the deconvolutions.
                                    - 'filter_dim': the number of filters for each deconvolution
                                                    layer.
                                    - 'activ': the activation to use after each hidden layer.
                                    - 'output_activ': the activation to use after the final layer.
        verbose (bool): whether to print debugging statements.
    Returns:
        (tf.keras.Sequential): TensorFlow model object.
    """
    input_shape, output_shape = dataset.output_shapes
    utils.v_print(
        verbose,
        "Creating deconvolutional network with input "
        "shape {} and output shape {}".format(input_shape, output_shape),
    )

    # initial dense output and final dense input must have dimensions (h, w, c)
    assert len(config.initial_dense) == 3
    utils.v_print(
        verbose, "Using initial dense layer with output shape {}".format(config.initial_dense)
    )

    # find the output shape of the transpose convolutions
    get_shape = utils.conv_transpose_output(config.kernel_size, config.stride)
    if config.layers < 0:
        # determine the number of deconvolutions that produces the output closest to the
        # desired size
        goal = np.prod(output_shape).value
        config.layers = 1
        diff = np.abs(np.prod(config.initial_dense) - goal)
        new_diff = np.abs(
            get_shape(config.initial_dense[0], config.layers)
            * get_shape(config.initial_dense[1], config.layers)
            - goal
        )
        while new_diff < diff:
            diff = new_diff
            config.layers += 1
            new_diff = np.abs(
                get_shape(config.initial_dense[0], config.layers)
                * get_shape(config.initial_dense[1], config.layers)
                - goal
            )
        config.layers -= 1

    final_shape = (
        get_shape(config.initial_dense[0], config.layers),
        get_shape(config.initial_dense[1], config.layers),
        config.filter_dim if config.layers else 1,  # no filter dimension if no conv layers
    )
    utils.v_print(
        verbose,
        "Using {} deconv layers each with kernel size {} and "
        "stride {} for an output of shape {}".format(
            config.layers, config.kernel_size, config.stride, final_shape
        ),
    )

    model = tf.keras.Sequential()

    # initial dense layer
    model.add(
        tf.keras.layers.Dense(
            units=np.prod(config.initial_dense), activation=config.activ, input_shape=input_shape
        )
    )
    # reshape is needed to return to the correct image tensor rank (3, plus 1 for batch)
    model.add(tf.keras.layers.Reshape(config.initial_dense))

    # transpose convolutional layers
    for _ in range(config.layers):
        model.add(
            tf.keras.layers.Conv2DTranspose(
                filters=config.filter_dim,
                kernel_size=config.kernel_size,
                strides=config.stride,
                activation=config.activ,
            )
        )

    # flatten
    model.add(tf.keras.layers.Reshape((np.prod(final_shape),)))

    # final dense output layer
    model.add(tf.keras.layers.Dense(units=np.prod(output_shape), activation=config.output_activ))
    model.add(tf.keras.layers.Reshape(output_shape))

    return model
