from tensorflow.keras import layers, models


def convBlock(inputs, filters):
    c1 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)

    # Taking first input and implementing the second conv block
    c2 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(c1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)

    return c2


def encoderBlock(inputs, filters):
    c = convBlock(inputs, filters)
    m = layers.MaxPooling2D(strides=(2, 2))(c)
    return c, m


def decoderBlock(inputs, skip, filters=64):
    u = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    skip = layers.Concatenate()([u, skip])
    out = convBlock(skip, filters)
    return out


def uNet(image_size):
    inputs = layers.Input(image_size)

    # Construct the encoder blocks and increasing the filters by a factor of 2
    skip1, encoder_1 = encoderBlock(inputs, 64)
    skip2, encoder_2 = encoderBlock(encoder_1, 128)
    skip3, encoder_3 = encoderBlock(encoder_2, 256)
    skip4, encoder_4 = encoderBlock(encoder_3, 512)

    # Preparing the next block
    conv_block = convBlock(encoder_4, 1024)

    # Construct the decoder blocks and decreasing the filters by a factor of 2
    decoder_1 = decoderBlock(conv_block, skip4, 512)
    decoder_2 = decoderBlock(decoder_1, skip3, 256)
    decoder_3 = decoderBlock(decoder_2, skip2, 128)
    decoder_4 = decoderBlock(decoder_3, skip1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(decoder_4)

    model = models.Model(inputs, outputs)

    return model
