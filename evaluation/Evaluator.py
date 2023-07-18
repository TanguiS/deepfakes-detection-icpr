from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from architectures import fornet
from architectures.fornet import FeatureExtractor


class Evaluator:
    def __init__(self, model_path: Path, net_name: str) -> None:
        super().__init__()
        self.__model_path = model_path

    def __load_model(self, net_name: str):
        state_tmp = torch.load(self.__model_path, map_location='cpu')
        if 'net' not in state_tmp.keys():
            state = OrderedDict({'net': OrderedDict()})
            [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
        else:
            state = state_tmp
        net_class = getattr(fornet, net_name)
        self.__net: FeatureExtractor = net_class().eval().to('cpu')
        self.__model = self.__net.load_state_dict(state['net'], strict=True)


def preprocess(size: int, frame_path: Optional[Path] = None, frame_base64: Optional[str] = None):
    if frame_path is None and frame_base64 is None:
        raise ValueError
    opener = cv2.imread
    if frame_path is None:
        opener = None
    face = np.zeros((size, size, 3), dtype=np.uint8)
    face = cv2.imread()
    face = np.array(face)
