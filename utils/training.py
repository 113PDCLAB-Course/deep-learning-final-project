from utils.dataset import create_dataset, load_annotations
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

    train_dataset = create_dataset(train_image_paths, train_mask_image_paths)
    val_dataset = create_dataset(val_image_paths, val_mask_image_paths)

    for model_name, model in models.items():
        model.fit(
            train_dataset,
            epochs=5,
            validation_data=val_dataset,
            verbose=1,
        )
        model.save(f"/app/weights/{model_name}.weights.h5")
        print(f"Model {model_name} saved")
