from functools import lru_cache
from pathlib import Path
from typing import Iterable

import dlib
import numpy as np

from facerec.recognizer import ImageType, Descriptor, Location


@lru_cache(maxsize=1)
class DlibFaceDescriptorComputer:
    def __init__(self, shape_weights: Path, face_weights: Path) -> None:
        self._shape_weights = shape_weights
        self._face_weights = face_weights
        self._frontal_face_detector = dlib.get_frontal_face_detector()
        self._shape_predictor = dlib.shape_predictor(str(shape_weights))
        self._descriptor_computer = dlib.face_recognition_model_v1(str(face_weights))

    def __call__(self, image: ImageType) -> Iterable[tuple[Descriptor, Location]]:
        detections = self._frontal_face_detector(image, 1)
        for d in detections:
            shape = self._shape_predictor(image, d)
            face_chip = dlib.get_face_chip(image, shape)
            descriptor = self._descriptor_computer.compute_face_descriptor(face_chip)
            yield (
                np.array(descriptor),
                Location(xmin=d.left(), ymin=d.top(), xmax=d.right(), ymax=d.bottom()),
            )
