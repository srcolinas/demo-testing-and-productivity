from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Payload(_message.Message):
    __slots__ = ["image"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    def __init__(self, image: _Optional[bytes] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["detections"]
    class Location(_message.Message):
        __slots__ = ["xmin", "ymin", "xmax", "ymax"]
        XMIN_FIELD_NUMBER: _ClassVar[int]
        YMIN_FIELD_NUMBER: _ClassVar[int]
        XMAX_FIELD_NUMBER: _ClassVar[int]
        YMAX_FIELD_NUMBER: _ClassVar[int]
        xmin: int
        ymin: int
        xmax: int
        ymax: int
        def __init__(self, xmin: _Optional[int] = ..., ymin: _Optional[int] = ..., xmax: _Optional[int] = ..., ymax: _Optional[int] = ...) -> None: ...
    class Detection(_message.Message):
        __slots__ = ["location", "name", "known"]
        LOCATION_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        KNOWN_FIELD_NUMBER: _ClassVar[int]
        location: Response.Location
        name: str
        known: bool
        def __init__(self, location: _Optional[_Union[Response.Location, _Mapping]] = ..., name: _Optional[str] = ..., known: bool = ...) -> None: ...
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    detections: _containers.RepeatedCompositeFieldContainer[Response.Detection]
    def __init__(self, detections: _Optional[_Iterable[_Union[Response.Detection, _Mapping]]] = ...) -> None: ...
