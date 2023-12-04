import cv2

from facerec_client.image_producer import Signals


def read_signal() -> Signals:
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        return Signals.STOP
    return Signals.CONTINUE
