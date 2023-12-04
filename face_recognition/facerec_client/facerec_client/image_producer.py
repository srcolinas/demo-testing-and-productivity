from enum import Enum, auto
from typing import Annotated, Callable, Iterable, Protocol

import numpy as np

ImageType = Annotated[np.ndarray, ("hight", "width", 3, np.uint8)]


class Camera(Protocol):
    def read(self) -> tuple[bool, ImageType]:
        ...

    def release(self) -> None:
        ...


class Visualizer(Protocol):
    def show(self, image: ImageType, /) -> None:
        ...

    def destroy(self) -> None:
        ...


class Signal(Enum):
    STOP = auto()
    CONTINUE = auto()


def produce_images(
    camera: Camera, visualizer: Visualizer, signal_reader: Callable[[], Signal]
) -> Iterable[ImageType]:
    while True:
        if signal_reader() is Signal.STOP:
            break

        read, cap = camera.read()
        if not read:
            break
        visualizer.show(cap)
        yield cap

    camera.release()
    visualizer.destroy()
