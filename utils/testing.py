from utils.dataset import create_dataset, load_annotations
from utils.mask import run_mask_tasks
from utils.metrics import cal_accuracy
from utils.parameters import test_mask_path, test_path


def testing_models(models):
    test_annotations = load_annotations(test_path)

    run_mask_tasks([("test", test_annotations, test_path)])

    test_image_paths = [
        f"{test_path}/{image['file_name']}" for image in test_annotations["images"]
    ]

    test_mask_image_paths = [
        f"{test_mask_path}/{image['file_name']}" for image in test_annotations["images"]
    ]

    test_dataset = create_dataset(test_image_paths, test_mask_image_paths)

    for model_name, model in models.items():
        print(f"Model {model_name}")
        cal_accuracy(model, test_dataset)
