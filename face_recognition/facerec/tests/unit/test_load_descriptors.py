from pathlib import Path

import numpy as np

from facerec.load_descriptors import load_descriptors


def test(tmp_path: Path):
    filepath_1 = tmp_path / "descriptor_dana_1_.npy"
    np.save(filepath_1, np.array([[1, 2, 3], [4, 5, 6]]))
    filepath_2 = tmp_path / "descriptor_sebas_1_.npy"
    np.save(filepath_2, np.array([[7, 8, 9], [10, 11, 12]]))
    descriptors = load_descriptors(**{"dana": filepath_1, "sebas": filepath_2})
    assert set(descriptors.keys()) == {"dana", "sebas"}
    np.testing.assert_array_equal(descriptors["dana"], np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_array_equal(
        descriptors["sebas"], np.array([[7, 8, 9], [10, 11, 12]])
    )
