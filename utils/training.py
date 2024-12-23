import gc

import tensorflow as tf

from utils.dataset import load_annotations, load_data
from utils.mask import run_mask_tasks
from utils.parameters import train_mask_path, train_path, val_mask_path, val_path


def training_models(models):
    train_annotations = load_annotations(train_path)
    val_annotations = load_annotations(val_path)

    run_mask_tasks(
        [
            ("train", train_annotations, train_path),
            ("valid", val_annotations, val_path),
        ]
    )

    train_image_paths = [
        f"{train_path}/{image['file_name']}" for image in train_annotations["images"]
    ]
    val_image_paths = [
        f"{val_path}/{image['file_name']}" for image in val_annotations["images"]
    ]

    train_mask_image_paths = [
        f"{train_mask_path}/{image['file_name']}"
        for image in train_annotations["images"]
    ]
    val_mask_image_paths = [
        f"{val_mask_path}/{image['file_name']}" for image in val_annotations["images"]
    ]

    train_dataset_X, train_dataset_y = load_data(
        train_image_paths, train_mask_image_paths
    )
    val_dataset = load_data(val_image_paths, val_mask_image_paths)

    for model_name, model in models.items():
        # Add callbacks for memory management
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"/app/weights/{model_name}_checkpoint.keras",
                save_best_only=True,
                monitor="val_loss",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ]

        model.fit(
            x=train_dataset_X,
            y=train_dataset_y,
            epochs=10,
            validation_data=val_dataset,
            batch_size=1,  # Reduced batch size
            verbose=1,
            callbacks=callbacks,
        )

        # Save model and clear memory
        model.save(f"/app/weights/{model_name}.weights.h5")
        print(f"Model {model_name} saved")

        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
