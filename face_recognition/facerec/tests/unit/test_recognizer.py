from typing import Iterable

import cv2
import pytest
import numpy as np

from facerec.recognizer import (
    Recognizer,
    Location,
    Detection,
    Descriptor,
    FaceDescriptorComputer,
    ImageType,
)


def test_detection_if_distance_below_05(image: bytes):
    rec = Recognizer(
        known_people={"Dana": np.array([1.0, 0.8]).reshape((1, 2))},
        descriptor_computer=_get_dummy_computer(
            [(np.array([0.9, 0.7]), Location(xmin=1, ymin=2, xmax=3, ymax=4))]
        ),
    )
    detections = list(rec(image))
    assert len(detections)
    assert detections[0] == Detection(
        location=Location(xmin=1, ymin=2, xmax=3, ymax=4), name="Dana", known=True
    )


def test_detection_if_distance_above_05(image: bytes):
    rec = Recognizer(
        known_people={"Sebas": np.array([1.0, 0.8]).reshape((1, 2))},
        descriptor_computer=_get_dummy_computer(
            [(np.array([0.1, 0.2]), Location(xmin=1, ymin=2, xmax=3, ymax=4))]
        ),
    )
    detections = list(rec(image))
    assert len(detections)
    assert detections[0] == Detection(
        location=Location(xmin=1, ymin=2, xmax=3, ymax=4), name="", known=False
    )


@pytest.fixture
def image() -> bytes:
    arr = np.zeros((10, 10, 3), np.uint8)
    return cv2.imencode(".png", arr)[1].tobytes()


def _get_dummy_computer(
    result: Iterable[tuple[Descriptor, Location]]
) -> FaceDescriptorComputer:
    class Helper:
        def __call__(self, image: ImageType) -> Iterable[tuple[Descriptor, Location]]:
            return result

    return Helper()
