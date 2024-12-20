from tensorflow.keras import layers, models


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channels = init.shape[-1]

    # Squeeze operation
    se = layers.GlobalAveragePooling2D()(init)

    # Excitation operation
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)

    return layers.multiply([init, se])


def ASPP(inputs):
    shape = inputs.shape

    # Image pooling
    pool = layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    pool = layers.Conv2D(256, 1, padding="same")(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.ReLU()(pool)
    pool = layers.UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(pool)

    # Different dilation rates
    rate_1 = layers.Conv2D(256, 1, padding="same")(inputs)
    rate_1 = layers.BatchNormalization()(rate_1)
    rate_1 = layers.ReLU()(rate_1)

    rate_6 = layers.Conv2D(256, 3, padding="same", dilation_rate=6)(inputs)
    rate_6 = layers.BatchNormalization()(rate_6)
    rate_6 = layers.ReLU()(rate_6)

    rate_12 = layers.Conv2D(256, 3, padding="same", dilation_rate=12)(inputs)
    rate_12 = layers.BatchNormalization()(rate_12)
    rate_12 = layers.ReLU()(rate_12)

    rate_18 = layers.Conv2D(256, 3, padding="same", dilation_rate=18)(inputs)
    rate_18 = layers.BatchNormalization()(rate_18)
    rate_18 = layers.ReLU()(rate_18)

    # Concatenate all features
    concat = layers.Concatenate()([pool, rate_1, rate_6, rate_12, rate_18])
    conv = layers.Conv2D(256, 1, padding="same")(concat)
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)

    return conv


def residual_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    shortcut = layers.Conv2D(filters, 1, padding="same")(inputs)
    shortcut = layers.BatchNormalization()(shortcut)

    output = layers.Add()([shortcut, x])
    output = layers.ReLU()(output)
    output = squeeze_excite_block(output)

    return output


def attention_block(inputs, skip_connection, filters):
    # Process skip connection
    g1 = layers.Conv2D(filters, 1, padding="same")(skip_connection)
    g1 = layers.BatchNormalization()(g1)

    # Process inputs
    x1 = layers.Conv2D(filters, 1, padding="same")(inputs)
    x1 = layers.BatchNormalization()(x1)

    # Combine and create attention coefficients
    psi = layers.Activation("relu")(layers.Add()([g1, x1]))
    psi = layers.Conv2D(1, 1, padding="same")(psi)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation("sigmoid")(psi)

    # Apply attention
    return layers.multiply([skip_connection, psi])


def ResUNetPlusPlus(image_size):
    inputs = layers.Input(image_size)

    # Encoder
    enc1 = residual_block(inputs, 64)
    pool1 = layers.MaxPooling2D((2, 2))(enc1)

    enc2 = residual_block(pool1, 128)
    pool2 = layers.MaxPooling2D((2, 2))(enc2)

    enc3 = residual_block(pool2, 256)
    pool3 = layers.MaxPooling2D((2, 2))(enc3)

    enc4 = residual_block(pool3, 512)
    pool4 = layers.MaxPooling2D((2, 2))(enc4)

    # Bridge
    aspp = ASPP(pool4)

    # Decoder
    dec4 = layers.Conv2DTranspose(512, (2, 2), strides=2, padding="same")(aspp)
    dec4 = attention_block(dec4, enc4, 512)
    dec4 = layers.Concatenate()([dec4, enc4])
    dec4 = residual_block(dec4, 512)

    dec3 = layers.Conv2DTranspose(256, (2, 2), strides=2, padding="same")(dec4)
    dec3 = attention_block(dec3, enc3, 256)
    dec3 = layers.Concatenate()([dec3, enc3])
    dec3 = residual_block(dec3, 256)

    dec2 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding="same")(dec3)
    dec2 = attention_block(dec2, enc2, 128)
    dec2 = layers.Concatenate()([dec2, enc2])
    dec2 = residual_block(dec2, 128)

    dec1 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding="same")(dec2)
    dec1 = attention_block(dec1, enc1, 64)
    dec1 = layers.Concatenate()([dec1, enc1])
    dec1 = residual_block(dec1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(dec1)

    model = models.Model(inputs, outputs)
    return model
