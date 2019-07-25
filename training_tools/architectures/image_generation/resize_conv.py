"""Neural network with layers composed of a resize followed by a convolution, resulting in a similar effect to a
transpose convolution but without the overlapping effects.

Kyle Roth. 2019-07-20.
"""


import numpy as np
import tensorflow as tf

from training_tools import utils
from training_tools.components import get_image_resize_method  # pylint:disable=no-name-in-module


class ResizeConvolution(tf.keras.layers.Conv2D):
    """This is designed to be a drop-in replacement for deconvolutions, without checkerboard artifacts. See
    https://distill.pub/2016/deconv-checkerboard/.

    Each resize step has the effect of two deconvolutions, so that the convolution brings the output to the desired
    shape as if a single deconvolution had been performed."""
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            **kwargs):
        """Perform Conv2D initialization, and prepare arguments for tf.image.resize_images.

        Args in addition to arguments to the Conv2D layer:
            resize_method (tf.image.ResizeMethod): method to specify to tf.image.resize_images.
        Args changed in call to the Conv2D layer:
            data_format: always set to 'channels_last' in order to use tf.image.resize_images.
        """
        self.resize_method = get_image_resize_method(resize_method)

        # convert strides and kernel_size to tuples if necessary
        if isinstance(strides, int):
            strides = (strides,) * 2
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        # get functions that calculate the size of each output dimension
        self.dim1_sizer = utils.conv_transpose_output(kernel_size[0], strides[0])
        self.dim2_sizer = utils.conv_transpose_output(kernel_size[1], strides[1])

        super(ResizeConvolution, self).__init__(
            filters,
            kernel_size,
            strides,
            padding,
            'channels_last',
            dilation_rate,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs
        )

    def call(self, inputs):
        """Perform resizing before the convolution."""
        inputs = tf.image.resize_images(
            inputs,
            (self.dim1_sizer(inputs.shape[1], 2), self.dim2_sizer(inputs.shape[2], 2)),
            method=self.resize_method)
        return super(ResizeConvolution, self).call(inputs)

    def compute_output_shape(self, input_shape):
        """Calculate the shape as if this layer were a Conv2DTranspose."""
        return tf.keras.layers.Conv2DTranspose.compute_output_shape(self, input_shape)


def create_model(dataset, config, verbose):
    """Create sequential network with linear layers at input and output, and enough resize-convolution layers to get the
    right size.

    Args:
        dataset (tf.Dataset): the dataset the model will be trained on. Used for determining input and output shape.
        config (mag.config.Config): mag configuration object. Must have the following attributes:
                                    - 'initial_dense': the shape of the initial dense layer's output. The channel
                                                       dimension must be last.
                                    - 'kernel_size': the kernel size used by the convolutions.
                                    - 'resize_method': method used to resize before each convolution. Must be equivalent
                                                       to a member of tf.image.ResizeMethod.
                                    - 'stride': the stride length used by the convolutions.
                                    - 'layers': the number of resize-convolutions to use. If set to -1, the number is
                                                chosen that produces an output of similar size to the final output, not
                                                including the filter dimension of the convolutions.
                                    - 'filter_dim': the number of filters for each convolution layer.
        verbose (bool): whether to print debugging statements.
    Returns:
        (tf.keras.Sequential): TensorFlow model object.
    """
    input_shape, output_shape = dataset.output_shapes
    utils.v_print(
        verbose,
        'Creating resize-convolutional network with input shape {} and output shape {}'.format(
            input_shape, output_shape)
    )

    # the initial dense output and final dense input must have dimensions (height, width, channel)
    assert len(config.initial_dense) == 3
    utils.v_print(verbose, 'Using initial dense layer with output shape {}'.format(config.initial_dense))

    # find the output shape of the convolution layers
    get_shape = utils.conv_transpose_output(config.kernel_size, config.stride)
    if config.layers < 0:
        # determine the number of resize-convolution layers that produces the output closest to the desired size
        goal = np.prod(output_shape).value
        config.layers = 1
        diff = np.abs(np.prod(config.initial_dense) - goal)
        new_diff = np.abs(get_shape(config.initial_dense[0], config.layers) *
                          get_shape(config.initial_dense[1], config.layers) - goal)
        while new_diff < diff:
            diff = new_diff
            config.layers += 1
            new_diff = np.abs(get_shape(config.initial_dense[0], config.layers) *
                              get_shape(config.initial_dense[1], config.layers) - goal)
        config.layers -= 1

    final_shape = (
        get_shape(config.initial_dense[0], config.layers),
        get_shape(config.initial_dense[1], config.layers),
        config.filter_dim
    )
    utils.v_print(
        verbose,
        'Using {} resize-conv layers each with kernel size {} and stride {} for an output of shape {}'.format(
            config.layers,
            config.kernel_size,
            config.stride,
            final_shape
        )
    )

    model = tf.keras.Sequential()

    # initial dense layer
    model.add(tf.keras.layers.Dense(
        units=np.prod(config.initial_dense),
        activation=config.activ,
        input_shape=input_shape
    ))
    # a reshape is needed to return to the correct image tensor rank (3, plus 1 for batch)
    model.add(tf.keras.layers.Reshape(config.initial_dense))

    # transpose convolutional layers
    for _ in range(config.layers):
        model.add(ResizeConvolution(
            filters=config.filter_dim,
            kernel_size=config.kernel_size,
            strides=config.stride,
            activation=config.activ,
            resize_method=config.resize_method
        ))

    # flatten
    model.add(tf.keras.layers.Reshape((np.prod(final_shape),)))

    # final dense output layer
    model.add(tf.keras.layers.Dense(units=np.prod(output_shape), activation=config.output_activ))
    model.add(tf.keras.layers.Reshape(output_shape))

    return model
