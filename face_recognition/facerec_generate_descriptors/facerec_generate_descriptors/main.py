import sys

import cv2
import numpy as np

from facerec.finder import find_weights
from facerec.recognizer import FaceDescriptorComputer, Descriptor
from facerec.face_descriptor import DlibFaceDescriptorComputer
from facerec.settings import Settings

from facerec_client.image_producer import produce_images, ImageType
from facerec_client.keyboard_handler import read_signal


def _main():
    settings = Settings()
    face, shape = find_weights(settings.models_dir)
    computer = DlibFaceDescriptorComputer(shape, face)
    list(
        produce_images(
            cv2.VideoCapture(0),
            VisualizerWithComputerAndStorage(computer),
            read_signal
        )
    )



class VisualizerWithComputerAndStorage:
    def __init__(self, computer: FaceDescriptorComputer) -> None:
        self._name = "I see you"
        self._computer = computer
        self._all_descriptions: list[Descriptor] = []

    def show(self, image: ImageType) -> None:
        results = list(self._computer(image))
        assert len(results) == 1, "there should be only one face"
        description, location = results[0]
        cv2.rectangle(
            image,
            (location.xmin, location.ymin),
            (location.xmax, location.xmin),
            (0,255,0), 2
        )
        cv2.imshow(self._name, image)

        while True:
            k = cv2.waitKey(1)
            if k == 32:
                # accumulate (Space)
                self._all_descriptions.append(description.reshape((1, -1)))
                break
            elif k == 2555904:
                # ignore (Right Arrow)
                break
            elif k % 256 == 27:
                # Exit (ESC)
                np.save("descriptor.npy", self._all_descriptions)
                sys.exit()
            

    def destroy(self) -> None:
        cv2.destroyAllWindows()

