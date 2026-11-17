"""Tests for the pair features and the compatibility model."""
from pathlib import Path

import numpy as np
import pytest

from ml.features import pair_matrix


def test_pair_matrix_shape_and_symmetry():
    """pair_matrix makes 1,024 numbers per pair, and item order doesn't matter."""
    rng = np.random.default_rng(0)                     # fixed seed: same random numbers every run
    ia, ib = rng.random((3, 256)), rng.random((3, 256))   # 3 fake pairs of image vectors
    ta, tb = rng.random((3, 256)), rng.random((3, 256))   # and of text vectors
    X = pair_matrix(ia, ta, ib, tb)
    assert X.shape == (3, 1024)                        # 3 rows, 256 * 4 columns
    assert np.allclose(X, pair_matrix(ib, tb, ia, ta))   # swapping A and B gives the same rows


# Where the trained model should be; it's gitignored, so it may be missing
MODEL = Path(__file__).resolve().parents[1] / "ml" / "models" / "compat_model.joblib"


# Skip (not fail) on computers that haven't copied the model files from Drive
@pytest.mark.skipif(not MODEL.exists(), reason="model files not downloaded from Drive")
def test_predict_compatibility_range_and_symmetry():
    """Real model: scores are between 0 and 1 and don't depend on item order."""
    from ml.compatibility import predict_compatibility   # imported here so the skip happens first
    rng = np.random.default_rng(1)
    # Two fake items in Sam's raw format: 2,048 image numbers, 256 text numbers
    a = {"image": rng.random(2048), "text": rng.random(256)}
    b = {"image": rng.random(2048), "text": rng.random(256)}
    s = predict_compatibility(a, b)
    assert 0.0 <= s <= 1.0                                    # a probability
    assert abs(s - predict_compatibility(b, a)) < 1e-6        # same score either way