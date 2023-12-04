import pytest
import numpy as np

from facerec_client.image_producer import produce_images, ImageType, Signal


def test_yield_what_visualizes(image: ImageType):
    camera = _Camera([(True, image), (True, image * 2), (False, image)])
    visualizer = _Visualizer()
    for i, img in enumerate(
        produce_images(
            camera,
            visualizer,
            _SignalReader(2000),
        ),
        start=1,
    ):
        np.testing.assert_array_equal(img, image * i)
        np.testing.assert_array_equal(visualizer.history[i - 1], image * i)
    assert len(visualizer.history) == 2

    assert camera.released
    assert visualizer.destroyed


def test_runs_unitl_stop(image: ImageType):
    camera = _Camera([(True, image)] * 3)
    visualizer = _Visualizer()
    list(
        produce_images(
            camera,
            visualizer,
            _SignalReader(1),
        )
    )

    assert len(visualizer.history) == 1
    np.testing.assert_array_equal(visualizer.history[0], image)

    assert camera.released
    assert visualizer.destroyed


def test_runs_unitl_not_read(image: ImageType):
    camera = _Camera([(True, image), (False, image * 2)])
    visualizer = _Visualizer()
    list(
        produce_images(
            camera,
            visualizer,
            _SignalReader(2000),
        )
    )
    assert len(visualizer.history) == 1
    np.testing.assert_array_equal(visualizer.history[0], image)

    assert camera.released
    assert visualizer.destroyed


@pytest.fixture
def image():
    return np.array([[[7, 8, 6], [7, 4, 3]], [[6, 3, 0], [1, 5, 8]]], dtype=np.uint8)


class _Visualizer:
    def __init__(self) -> None:
        self.history: list[ImageType] = []
        self.destroyed = False

    def show(self, image: ImageType, /) -> None:
        self.history.append(image)

    def destroy(self) -> None:
        self.destroyed = True


class _Camera:
    def __init__(self, reads: list[tuple[bool, ImageType]]) -> None:
        self.released = False
        self._reads = reads

    def read(self) -> tuple[bool, ImageType]:
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True


class _SignalReader:
    def __init__(self, calls_before_stop: int) -> None:
        self._counter = 0
        self._calls_before_stop = calls_before_stop

    def __call__(self) -> Signal:
        if self._counter < self._calls_before_stop:
            self._counter += 1
            return Signal.CONTINUE
        else:
            return Signal.STOP
