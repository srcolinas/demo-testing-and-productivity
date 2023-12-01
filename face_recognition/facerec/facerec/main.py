from concurrent import futures

import grpc

from facerec_pb2s import facerec_pb2
from facerec_pb2s import facerec_pb2_grpc

from facerec.detector import Detector
from facerec.settings import Settings


class Service(facerec_pb2_grpc.FaceRecognitionServicer):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def run(self, request: facerec_pb2.Payload, context):
        detector = Detector.load(self._settings.models_dir)
        detection = detector.detect(request.image)
        return facerec_pb2.Response(
            detections=[
                facerec_pb2.Response.Detection(
                    location=facerec_pb2.Response.Location(
                        xmin=detection.location.xmin,
                        ymin=detection.location.ymin,
                        xmax=detection.location.xmax,
                        ymax=detection.location.ymax,
                    ),
                    name=detection.name,
                    known=detection.known,
                )
            ]
        )


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
