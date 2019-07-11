"""Neural network with transpose convolutions, or "deconvolutions" as the cool kids say.

Kyle Roth. 2019-07-10.
"""


def create_model(dataset, config):
    """Create sequential network with linear layers at input and output, and enough transpose convolutions to get the
    right size.

    Args:
        dataset (tf.Dataset): the dataset the model will be trained on. Used for determining input and output shape.
        config (mag.config.Config): mag configuration object.
    Returns:
        (callable): TensorFlow model object.
    """
    # TODO: get sizes from dataset

    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=config.initial_dense ** 2,
            activation=config.activ,
            name='d_in',
            input_shape=(None, 1, 2)
        ),
        tf.keras.layers.Reshape((config.initial_dense, config.initial_dense, 1), name='input_reshape'),
        *[
            tf.keras.layers.Conv2DTranspose(
                filters=config.conv_filter_dim,
                kernel_size=config.kernel_size,
                strides=config.stride,
                activation=config.activ,
                name='C2DT_{}'.format(i)
            ) for i in range(int(np.log2(output_len // 2 / config.initial_dense)))
            # enough to ensure size (batch, output_len // 2, output_len // 2, 1)
        ],
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=config.kernel_size,
            strides=config.stride,
            activation=config.activ,
            name='C2DT_last'
        ),
        # should come out (batch, 128, 128, 1)
        tf.keras.layers.Reshape((output_len ** 2,), name='dense_reshape'),
        tf.keras.layers.Dense(units=output_len ** 2, activation='tanh', name='dense_out'),
        tf.keras.layers.Reshape((output_len, output_len, 1), name='output_reshape')
    ], name='fc_conv2DT_model')
