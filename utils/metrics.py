from typing import Tuple

import numpy as np
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-7) -> float:
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred) -> float:
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


def cal_accuracy(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> Tuple[float, float, float]:
    batch_size = 2
    predictions = []
    y_true = []

    for image_batch, mask_batch in dataset:
        batch_predictions = model.predict(image_batch, batch_size=batch_size)
        predictions.append(batch_predictions)
        y_true.append(mask_batch.numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # Reshape predictions if necessary (assuming single-channel masks)
    y_pred = np.squeeze(y_pred)
    y_pred = np.expand_dims(y_pred, axis=-1)

    threshold = 0.5
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate pixel-wise accuracy
    accuracy = np.mean(y_pred_binary == y_true)

    dice_coefficient_score = dice_coefficient(y_true, y_pred)
    dice_loss_score = dice_loss(y_true, y_pred)

    return accuracy, dice_coefficient_score, dice_loss_score
