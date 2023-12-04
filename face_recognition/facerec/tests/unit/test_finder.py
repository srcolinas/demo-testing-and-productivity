from pathlib import Path

from facerec.finder import find_weights_and_descriptors


def test_finds_latest_version(tmp_path: Path):
    location = tmp_path / "test"
    location.mkdir()

    (location / "shape_predictor_1_.dat").write_text("")
    (location / "shape_predictor_2_.dat").write_text("")
    (location / "face_recognition_1_.dat").write_text("")
    (location / "face_recognition_2_.dat").write_text("")
    (location / "face_recognition_3_.dat").write_text("")
    (location / "descriptor_dana_2_.npz").write_text("")
    (location / "descriptor_dana_3_.npz").write_text("")
    (location / "descriptor_dana_4_.npz").write_text("")
    (location / "descriptor_sebas_4_.npz").write_text("")
    (location / "descriptor_sebas_5_.npz").write_text("")
    (location / "descriptor_sebas_6_.npz").write_text("")

    face, shape, descriptors = find_weights_and_descriptors(location)
    assert face == location / "face_recognition_3_.dat"
    assert shape == location / "shape_predictor_2_.dat"
    assert descriptors == {
        "dana": location / "descriptor_dana_4_.npz",
        "sebas": location / "descriptor_sebas_6_.npz",
    }
