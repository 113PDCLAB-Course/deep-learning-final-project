import glob
import json
import os
from typing import List

import tensorflow as tf


def load_annotations(path: str) -> dict:
    annotations = glob.glob(os.path.join(path, "*.json"))
    annotations = json.load(open(annotations[0]))

    return annotations


def preprocess_image(image_path: str, mask_path: str, target_size=(512, 512)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, target_size) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return image, mask


def create_dataset(
    image_paths: List[str], mask_paths: List[str], batch_size=32, target_size=(512, 512)
):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, mask: preprocess_image(img, mask, target_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# def load_data():
#     target_size = (512, 512)
#     train_mask_dir = '/kaggle/working/train_masks/'
#     X_train =  [cv2.resize(cv2.imread(train_path + image['file_name']),target_size) for image in train_annotations['images']]
#     y_train = [cv2.resize(cv2.imread(train_mask_dir + image['file_name'],cv2.IMREAD_GRAYSCALE),target_size ) for image in train_annotations['images']]
#     X_train = np.array(X_train)
#     y_train = np.expand_dims(np.array(y_train), axis=-1)

#     X_train = X_train.astype('float32') / 255.0
#     y_train = y_train.astype('float32') / 255.0
#     y_train = (y_train > 0.5).astype(np.float32)

#     val_mask_dir = '/kaggle/working/val_masks/'

#     X_val =  [cv2.resize(cv2.imread(val_path + image['file_name']),target_size) for image in val_annotations['images']]
#     y_val = [cv2.resize(cv2.imread(val_mask_dir + image['file_name'],cv2.IMREAD_GRAYSCALE),target_size) for image in val_annotations['images']]
#     X_val = np.array(X_val)
#     y_val = np.expand_dims(np.array(y_val), axis=-1)


#     X_val = X_val.astype('float32') / 255.0
#     y_val = y_val.astype('float32') / 255.0
#     y_val = (y_val > 0.5).astype(np.float32)

#     return X_train,y_train,X_val,y_val

# def load_test_data():
#     target_size = (512, 512)

#     test_mask_dir = '/kaggle/working/test_masks/'
#     X_test =  [cv2.resize(cv2.imread(test_path + image['file_name']),target_size) for image in test_annotations['images']]
#     y_test = [cv2.resize(cv2.imread(test_mask_dir + image['file_name'],cv2.IMREAD_GRAYSCALE),target_size) for image in test_annotations['images']]
#     X_test = np.array(X_test)
#     y_test = np.expand_dims(np.array(y_test), axis=-1)

#     X_test = X_test.astype('float32') / 255.0
#     y_test = y_test.astype('float32') / 255.0
#     y_test = (y_test > 0.5).astype(np.float32)
#     return X_test,y_test

# X_train,y_train,X_val,y_val = load_data()
# X_train.shape , y_train.shape  ,X_val.shape ,y_val.shape
