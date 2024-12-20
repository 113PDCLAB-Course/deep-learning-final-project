from tensorflow.keras import layers, models


def group_conv_block(inputs, filters, strides, groups=32):
    # Split input channels into groups
    group_channels = filters // groups
    group_list = []

    for i in range(groups):
        x = layers.Conv2D(
            group_channels,
            kernel_size=3,
            strides=strides,
            padding="same",
            use_bias=False,
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        group_list.append(x)

    # Concatenate all groups
    return layers.Concatenate()(group_list)


def resnext_block(inputs, filters, strides=1, groups=32):
    # First 1x1 convolution
    x = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Group convolution
    x = group_conv_block(x, filters, strides, groups)

    # Final 1x1 convolution
    x = layers.Conv2D(filters * 2, kernel_size=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if strides != 1 or inputs.shape[-1] != filters * 2:
        shortcut = layers.Conv2D(
            filters * 2, kernel_size=1, strides=strides, padding="same", use_bias=False
        )(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def ResNext50(image_size):
    inputs = layers.Input(image_size)

    # Initial convolution
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x1 = x  # Skip connection 1
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Encoder path
    # Stage 1
    for i in range(3):
        x = resnext_block(x, filters=128, strides=1 if i > 0 else 1)
    x2 = x  # Skip connection 2

    # Stage 2
    for i in range(4):
        x = resnext_block(x, filters=256, strides=1 if i > 0 else 2)
    x3 = x  # Skip connection 3

    # Stage 3
    for i in range(6):
        x = resnext_block(x, filters=512, strides=1 if i > 0 else 2)
    x4 = x  # Skip connection 4

    # Stage 4 (Bridge)
    for i in range(3):
        x = resnext_block(x, filters=1024, strides=1 if i > 0 else 2)

    # Decoder path
    # Stage 4
    x = layers.Conv2DTranspose(512, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, x4])
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 3
    x = layers.Conv2DTranspose(256, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, x3])
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 2
    x = layers.Conv2DTranspose(128, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, x2])
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1
    x = layers.Conv2DTranspose(64, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, x1])
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final upsampling to match input size
    x = layers.Conv2DTranspose(32, (2, 2), strides=2, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model
