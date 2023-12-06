from concurrent import futures

import grpc

from facerec_pb2s import facerec_pb2
from facerec_pb2s import facerec_pb2_grpc

from facerec.recognizer import Recognizer
from facerec.finder import find_weights, find_descriptors
from facerec.load_descriptors import load_descriptors
from facerec.face_descriptor import DlibFaceDescriptorComputer
from facerec.settings import Settings


class Service(facerec_pb2_grpc.FaceRecognitionServicer):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def run(self, request: facerec_pb2.Payload, context):
        face, shape = find_weights(self._settings.models_dir)
        descriptors = find_descriptors(self._settings.descriptors_dir)
        rec = Recognizer(
            known_people=load_descriptors(**descriptors),
            descriptor_computer=DlibFaceDescriptorComputer(shape, face),
        )
        detections = [
            facerec_pb2.Response.Detection(
                location=facerec_pb2.Response.Location(
                    xmin=det.location.xmin,
                    ymin=det.location.ymin,
                    xmax=det.location.xmax,
                    ymax=det.location.ymax,
                ),
                name=det.name,
                known=det.known,
            )
            for det in rec(request.image)
        ]
        return facerec_pb2.Response(detections=detections)


def serve(settings: Settings):
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    facerec_pb2_grpc.add_FaceRecognitionServicer_to_server(Service(settings), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def _main():
    serve(Settings())


if __name__ == "__main__":
    _main()
