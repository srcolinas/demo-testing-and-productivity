from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


class _Findings(NamedTuple):
    face_weights: Path
    shape_weights: Path


def find_weights(location: Path, /) -> _Findings:
    """Finds the latest version of the models' weights.

    The following contents need to be in the location provided:
        1. shape_predictor_{version}_.dat
        2. face_recognition_{version}_.dat

    Where  {version} should be replaced by an integer indicating the
    version of the data file.
    """
    faces: list[Path] = []
    shapes: list[Path] = []
    for f in location.glob("*_.dat"):
        if f.name.startswith("face_recognition"):
            faces.append(f)
        elif f.name.startswith("shape_predictor"):
            shapes.append(f)

    face = sorted(faces, key=lambda x: x.name.split("_")[-2])[-1]
    shape = sorted(shapes, key=lambda x: x.name.split("_")[-2])[-1]
    return _Findings(face, shape)


def find_descriptors(location: Path, /) -> dict[str, Path]:
    """Finds the latest version of the known descriptors.

    The location should contain files named like
    descriptor_{name}_{version}_.npz

    Where  {version} should be replaced by an integer indicating the
    version of the descriptor file and insteade of {name} you include the name
    of each known person (the name associated to such descriptor).
    """

    versions: dict[str, list[str]] = defaultdict(list)
    descriptors = list(location.glob("*_.npy"))
    for d in descriptors:
        _, person, version, _ = d.name.split("_")
        versions[person].append(version)

    latest_versions = {k: sorted(v)[-1] for k, v in versions.items()}

    def helper():
        for k, v in latest_versions.items():
            for d in descriptors:
                if d.name.endswith(f"_{v}_.npy") and d.name.startswith(
                    f"descriptor_{k}"
                ):
                    yield (k, d)

    return dict(helper())
