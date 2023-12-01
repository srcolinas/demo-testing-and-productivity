from dataclasses import dataclass
from pathlib import Path


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


class Detector:
    @classmethod
    def load(cls, location: Path, /) -> "Detector":
        """Loads the latest version of the model.

        The following contents need to be in the location provided:
            1. `shape_predictor_{version}_.dat`
            2. `face_recognition_{version}_.dat`

        Where  _{version}_ should be replaced by an integer indicating the
        version of the models weights. This function contains a cache that
        makes sure the model is only loaded a newer version of the weights
        is available.
        """
        return cls()

    def detect(self, image: bytes) -> Detection:
        return Detection(
            location=Location(xmin=1, xmax=2, ymin=3, ymax=4), name="", known=False
        )
