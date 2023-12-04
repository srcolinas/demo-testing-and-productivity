import cv2

from facerec_client.image_producer import ImageType


class OpenCVVisualizer:
    def __init__(self) -> None:
        self._name = "I see you"

    def show(self, image: ImageType) -> None:
        cv2.imshow(self._name, image)

    def destroy(self) -> None:
        cv2.destroyAllWindows()
