import os

from tensorflow.keras.optimizers import Adam

from models.resnext50 import ResNext50
from models.resunet_plus_plus import ResUNetPlusPlus
from models.unet import uNet
from utils.metrics import dice_coefficient, dice_loss


class ModelZoo:
    models_map = {
        "unet": uNet,
        "resnext50": ResNext50,
        "resunet-plus-plus": ResUNetPlusPlus,
    }

    def get_model(self, model_name, initial_lr=1e-4, input_shape=(512, 512, 3)):
        weights_path = f"/app/weights/{model_name}.weights.h5"

        model = self.models_map.get(model_name)(input_shape)

        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        model.compile(
            optimizer=Adam(learning_rate=initial_lr),
            loss=dice_loss,
            metrics=["accuracy", dice_coefficient],
        )

        return model
