import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from numpy import ndarray

from architectures import fornet
from architectures.fornet import FeatureExtractor
from isplutils import utils


def read_models_information(model_path: Path) -> Tuple[str, int, str, str]:
    face_policy = str(model_path).split('face-')[1].split('_')[0]
    patch_size = int(str(model_path).split('size-')[1].split('_')[0])
    net_name = str(model_path).split('net-')[1].split('_')[0]
    model_name = '_'.join(model_path.with_suffix('').parts[-2:])
    return face_policy, patch_size, net_name, model_name


def preprocess_image(image: ndarray, model: FeatureExtractor, face_policy: str, patch_size: int) -> ndarray:
    test_transformer: A.BasicTransform = utils.get_transformer(
        face_policy, patch_size, model.get_normalizer(), train=False
    )
    face = test_transformer(image=image)['image']
    face_batch = face.unsqueeze(0)
    return face_batch


def read_image(image_path_or_base64: Union[str, Path]) -> Image:
    if isinstance(image_path_or_base64, Path):
        return Image.open(str(image_path_or_base64))
    image_bytes = base64.b64decode(image_path_or_base64)
    image_stream = BytesIO(image_bytes)
    return Image.open(image_stream)


def load_model(model_path: Path, net_name: str) -> FeatureExtractor:
    state_tmp = torch.load(model_path, map_location='cpu')
    if 'net' not in state_tmp.keys():
        raise ValueError("Don't know why.")
    else:
        state = state_tmp
    net_class = getattr(fornet, net_name)
    net: FeatureExtractor = net_class().eval().to('cpu')
    missing = net.load_state_dict(state['net'], strict=True)
    print(missing)
    return net
