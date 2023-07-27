from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from architectures.fornet import FeatureExtractor
from evaluation.util import read_models_information, preprocess_image, read_image


class ModelEvaluator:
    def __init__(self, loaded_model: FeatureExtractor, model_path: Path) -> None:
        super().__init__()
        self.__model = loaded_model
        self.__face_policy, self.__patch_size, self.__net_name, self.__model_name = read_models_information(model_path)

    def evaluate(self, image_path_or_base64: Union[Path, str]):
        image = read_image(image_path_or_base64)
        image_array = np.array(image)
        image_preprocessed = preprocess_image(image_array, self.__model, self.__face_policy, self.__patch_size)

        with torch.no_grad():
            yhat = self.__model(image_preprocessed)
            soft_yhat_predict_class = nn.functional.sigmoid(yhat).cpu().numpy()[0][0]
            soft_yhat_opposite_class = 1 - soft_yhat_predict_class
        return soft_yhat_predict_class, soft_yhat_opposite_class



