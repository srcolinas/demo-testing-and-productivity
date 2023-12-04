import shutil
import threading
import time
from pathlib import Path

import grpc
import pytest

from facerec_pb2s import facerec_pb2
from facerec_pb2s import facerec_pb2_grpc

from facerec.settings import Settings
from facerec.main import serve

def test(models_dir: Path, image: Path):
    threading.Thread(serve, [Settings(models_dir=models_dir)], daemon=True)
    time.sleep(5)
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = facerec_pb2_grpc.FaceRecognitionStub(channel)
        response = stub.run(facerec_pb2.Payload(image=image.read_bytes()))
        print(f"{response}")

@pytest.fixture
def models_dir(shape_weights: Path, face_weights: Path, descriptor: Path, tmp_path: Path):
    shutil.copyfile(shape_weights, tmp_path / "shape_predictor_1_.dat")
    shutil.copyfile(face_weights, tmp_path / "face_recognition_1_.dat")
    shutil.copyfile(descriptor, tmp_path / "descriptor_python_1_.npz")

@pytest.fixture
def descriptor(pytestconfig: pytest.Config):
    value: Path = pytestconfig.getoption("descriptor")
    assert value.is_file(), value.absolute()
    return value
