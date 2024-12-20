import os
import shutil

import kagglehub
import tensorflow as tf

from models.model_zoo import ModelZoo
from utils.testing import testing_models
from utils.training import training_models

os.environ["TF_KERAS"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def check_environment():
    # check gpu support
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if not gpus:
        print("No GPU support")
        return False

    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    return True


def download_dataset():
    if not os.path.exists("/app/datasets"):
        os.makedirs("/app/datasets")

    if os.listdir("/app/datasets") == []:
        print("Downloading dataset...")
        download_dir = kagglehub.dataset_download(
            "pkdarabi/brain-tumor-image-dataset-semantic-segmentation"
        )

        shutil.copytree(download_dir, "/app/datasets", dirs_exist_ok=True)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists")


def main():
    if not check_environment():
        print("Environment check failed")
        return

    download_dataset()

    model_zoo = ModelZoo()

    models = {
        "unet": model_zoo.get_model("unet"),
        "resnext50": model_zoo.get_model("resnext50"),
        "resunet-plus-plus": model_zoo.get_model("resunet-plus-plus"),
    }

    # training_models(models)

    testing_models(models)


if __name__ == "__main__":
    main()
