from pathlib import Path
from functools import lru_cache

import numpy as np

from facerec.recognizer import DescriptorBatch


@lru_cache(maxsize=1)
def load_descriptors(**ref: Path) -> dict[str, DescriptorBatch]:
    result: dict[str, DescriptorBatch] = {}
    for k, f in ref.items():
        result[k] = np.load(f, allow_pickle=False)
    return result
