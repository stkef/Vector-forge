import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vectorforge import VectorForge  # noqa: E402


@pytest.fixture
def db(tmp_path):
    with VectorForge(tmp_path / "data") as d:
        yield d


@pytest.fixture
def rng():
    return np.random.default_rng(42)
