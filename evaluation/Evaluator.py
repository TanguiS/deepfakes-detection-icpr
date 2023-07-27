from pathlib import Path
from typing import Union

import torch

from architectures.fornet import FeatureExtractor
from evaluation.util import read_models_information, preprocess_image, read_image


class ModelEvaluator:
    def __init__(self, loaded_model: FeatureExtractor, model_path: Path) -> None:
        super().__init__()
        self.__model = loaded_model
        self.__face_policy, self.__patch_size, self.__net_name, self.__model_name = read_models_information(model_path)

    def evaluate(self, image_path_or_base64: Union[Path, str]):
        image = read_image(image_path_or_base64)
        image_preprocessed = preprocess_image(image, self.__model, self.__face_policy, self.__patch_size)

        with torch.no_grad():
            yhat = self.__model(image_preprocessed)
        return yhat


