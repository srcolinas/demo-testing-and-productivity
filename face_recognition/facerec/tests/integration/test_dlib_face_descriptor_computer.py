from pathlib import Path

import cv2

from facerec.face_descriptor import DlibFaceDescriptorComputer


def test(shape_weights: Path, face_weights: Path, image: Path, descriptor_length: int):
    computer = DlibFaceDescriptorComputer(shape_weights, face_weights)
    image_arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    height, width, _ = image_arr.shape
    for desc, loc in computer(image_arr):
        assert desc.shape == (descriptor_length,)
        assert 0 <= loc.xmin < loc.xmax <= width
        assert 0 <= loc.ymin < loc.ymax <= height
