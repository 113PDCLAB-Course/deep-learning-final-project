import glob
import json
import os
from typing import List

import cv2
import numpy as np
import tensorflow as tf


def load_annotations(path: str) -> dict:
    annotations = glob.glob(os.path.join(path, "*.json"))
    annotations = json.load(open(annotations[0]))

    return annotations


@tf.function
def preprocess_image(image_path: str, mask_path: str, target_size=(512, 512)):
    # Read and decode image with better memory handling
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, target_size, method="nearest")
    image = image / 255.0

    # Read and decode mask with better memory handling
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.cast(mask, tf.float32)
    mask = tf.image.resize(mask, target_size, method="nearest")
    mask = mask / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


def create_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    batch_size=2,  # 降低 batch size
    target_size=(512, 512),
    buffer_size=1000,
):
    # Convert string lists to tensors for better memory handling
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)

    # Create dataset with optimizations
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Add memory optimizations
    dataset = dataset.cache()  # 快取資料到記憶體
    dataset = dataset.shuffle(buffer_size)  # 隨機打亂資料
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, target_size),
        num_parallel_calls=tf.data.AUTOTUNE,  # 平行處理
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 預先載入下一個 batch

    return dataset


def load_data(image_paths: List[str], mask_paths: List[str], target_size=(512, 512)):
    X = [cv2.resize(cv2.imread(image), target_size) for image in image_paths]
    y = [
        cv2.resize(
            cv2.imread(image, cv2.IMREAD_GRAYSCALE),
            target_size,
        )
        for image in mask_paths
    ]
    X = np.array(X)
    y = np.expand_dims(np.array(y), axis=-1)

    X = X.astype("float32") / 255.0
    y = y.astype("float32") / 255.0
    y = (y > 0.5).astype(np.float32)

    return X, y
