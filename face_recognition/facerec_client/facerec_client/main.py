from typing import Iterable

import cv2
import grpc

from facerec_pb2s import facerec_pb2
from facerec_pb2s import facerec_pb2_grpc

from facerec_client.image_producer import produce_images
from facerec_client.keyboard_handler import read_signal
from facerec_client.opencv_visualizer import OpenCVVisualizer


def run(producer: Iterable[bytes], address: str = "localhost:50051"):
    with grpc.insecure_channel(address) as channel:
        stub = facerec_pb2_grpc.FaceRecognitionStub(channel)
        for img in producer:
            response = stub.run(facerec_pb2.Payload(image=img))
            print(f"{response}")


def _main():
    run(produce_images(cv2.VideoCapture(0), OpenCVVisualizer(), read_signal))


if __name__ == "__main__":
    _main()
