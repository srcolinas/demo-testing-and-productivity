from dataclasses import dataclass
from typing import Annotated, Iterable, Mapping, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Location:
    xmin: int
    xmax: int
    ymin: int
    ymax: int


@dataclass(frozen=True)
class Detection:
    location: Location
    name: str
    known: bool


ImageType = Annotated[np.ndarray, ("hight", "width", 3, np.uint8)]
Descriptor = Annotated[np.ndarray, ("length",)]
DescriptorBatch = Annotated[np.ndarray, ("N", "length")]


class FaceDescriptorComputer(Protocol):
    def __call__(self, image: ImageType) -> Iterable[tuple[Descriptor, Location]]:
        ...


class Recognizer:
    def __init__(
        self,
        *,
        known_people: Mapping[str, DescriptorBatch],
        descriptor_computer: FaceDescriptorComputer,
    ) -> None:
        """
        Args:
            known_people:
                Mapping from names to descriptors of the known people.
            descriptor_computer:
                A function to get descriptors from new images.
        """
        self.known_peole = known_people
        self.descriptor_computer = descriptor_computer

    def __call__(self, image_: bytes, /) -> Iterable[Detection]:
        """
        For every detected person in the image, we compute the descriptor of
        such person and compar it with the set of descriptors available for
        each known person.

        We say a person is known if the average distance from its descriptor
        with the set of known descriptors is below 0.5.
        """
        image = np.frombuffer(image_, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        for desc, loc in self.descriptor_computer(image):
            for name, reff in self.known_peole.items():
                distance = np.linalg.norm(reff - desc).mean()
                if distance < 0.5:
                    yield Detection(location=loc, name=name, known=True)
                else:
                    yield Detection(location=loc, name="", known=False)
