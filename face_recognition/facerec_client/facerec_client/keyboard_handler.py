import cv2

from facerec_client.image_producer import Signal


def read_signal() -> Signal:
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        return Signal.STOP
    return Signal.CONTINUE
