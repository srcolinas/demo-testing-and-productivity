from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--shapeweights", action="store", type=Path)
    parser.addoption("--faceweights", action="store", type=Path)
    parser.addoption("--image", action="store", type=Path)
    parser.addoption("--descriptorlength", action="store", type=int)
    parser.addoption("--descriptor", action="store", type=Path)


@pytest.fixture(scope="session")
def shape_weights(pytestconfig: pytest.Config):
    value: Path = pytestconfig.getoption("shapeweights")
    assert value.is_file(), value.absolute()
    return value


@pytest.fixture(scope="session")
def face_weights(pytestconfig: pytest.Config):
    value: Path = pytestconfig.getoption("faceweights")
    assert value.is_file(), value.absolute()
    return value


@pytest.fixture(scope="session")
def image(pytestconfig: pytest.Config):
    value: Path = pytestconfig.getoption("image")
    assert value.is_file(), value.absolute()
    return value


@pytest.fixture(scope="session")
def descriptor_length(pytestconfig: pytest.Config):
    value: int = pytestconfig.getoption("descriptorlength")
    assert value > 0
    return value
