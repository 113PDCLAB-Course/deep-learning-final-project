import os
import shutil
from threading import Thread
from typing import List

import cv2
import numpy as np


def mask_image(image, annotation):
    # Create mask from annotation
    segmentation = annotation["segmentation"]
    segmentation = np.array(segmentation[0], dtype=np.int32).reshape(-1, 2)

    # Create empty mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Fill polygon on mask
    cv2.fillPoly(mask, [segmentation], color=255)

    return mask


def save_mask_images(name, annotations, img_dir_path):
    print(f"Creating {name} masks")

    # Create mask images directory
    mask_dir = os.path.join("/app/datasets/", f"{name}_masks")

    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)

    os.makedirs(mask_dir, exist_ok=True)

    img_len = len(annotations["images"])

    for i, (img, ann) in enumerate(
        zip(annotations["images"], annotations["annotations"])
    ):
        img_path = os.path.join(img_dir_path, img["file_name"])
        save_path = os.path.join(mask_dir, img["file_name"])

        # Load image
        image = cv2.imread(img_path)

        # Generate masked image
        masked_img = mask_image(image, ann)

        # Save masked image
        cv2.imwrite(save_path, masked_img)

        print(f"Processed {i + 1} / {img_len} for {name}")


# Example usage:
# create_mask_image("path/to/input/image.jpg", annotation_dict, "path/to/output/mask.jpg")
def save_mask_image(image_path: str, annotation: List[float], output_path: str):
    # Read image
    image = cv2.imread(image_path)

    masked_img = mask_image(image, annotation)

    # Save masked image
    cv2.imwrite(output_path, masked_img)

    return masked_img


def run_mask_tasks(tasks):
    threads = [
        Thread(target=save_mask_images, args=(name, annotations, dir_path))
        for name, annotations, dir_path in tasks
        if not os.path.exists(f"{dir_path}_masks")
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("Mask creation complete.")
