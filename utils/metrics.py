import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


def dice_coefficient(y_true, y_pred, smooth=1e-7) -> float:
    # Convert inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred) -> float:
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


def cal_accuracy(model: tf.keras.Model, dataset: tf.data.Dataset):
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

    print(f"Pixel-wise Accuracy: {accuracy:.4f}")
    print(f"Dice Coefficient: {dice_coefficient_score:.4f}")
    print(f"Dice Loss: {dice_loss_score:.4f}")

    # Flatten arrays for classification report
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred_binary.ravel()

    # Generate classification report for binary data
    try:
        report = classification_report(
            y_true_flat, y_pred_flat, target_names=["Background", "Object"]
        )
        print(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
