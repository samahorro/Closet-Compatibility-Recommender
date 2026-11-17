"""Pairwise compatibility model: score how well two items go together.

Other parts of the app (Jose's outfit search, the Streamlit pages) call
predict_compatibility() or predict_many() and don't need to know how the
model was trained.
"""
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib          # loads the saved scikit-learn model and PCA
import numpy as np

from ml.features import pair_matrix   # the same row-building function used in training

# The model files live in ml/models/ (gitignored, so copy them from the shared Drive)
MODELS = Path(__file__).parent / "models"

# Start empty; _load() fills these the first time a score is needed
_model = None
_pca = None


def _load():
    """Load the model and the image PCA once, then reuse them on every later call."""
    global _model, _pca                       # change the module-level variables above
    if _model is None:                        # only true on the very first call
        _model = joblib.load(MODELS / "compat_model.joblib")
        pca_path = MODELS / "image_pca.joblib"
        # The PCA file only exists when image vectors are bigger than 256 numbers
        _pca = joblib.load(pca_path) if pca_path.exists() else None
    return _model, _pca


def _unit(v: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (the tiny 1e-8 avoids dividing by zero)."""
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def _vectors(items: Sequence[Mapping[str, Any]]):
    """Prepare image and text vectors exactly as in training: PCA, then length 1."""
    # Stack each item's vector into one array with a row per item
    img = np.stack([np.asarray(it["image"], dtype="float32") for it in items])
    txt = np.stack([np.asarray(it["text"], dtype="float32") for it in items])
    _, pca = _load()
    # Shrink Sam's 2,048-number image vectors to 256, using the PCA fit in training
    if pca is not None and img.shape[1] == pca.n_features_in_:
        img = pca.transform(img).astype("float32")
    return _unit(img), _unit(txt)


def predict_many(items_a: Sequence[Mapping[str, Any]],
                 items_b: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Score many pairs at once: items_a[k] with items_b[k].

    Much faster than calling predict_compatibility() in a loop,
    which matters for the outfit search. Returns values from 0 to 1.
    """
    model, _ = _load()
    img_a, txt_a = _vectors(items_a)
    img_b, txt_b = _vectors(items_b)
    X = pair_matrix(img_a, txt_a, img_b, txt_b)   # one row per pair
    return model.predict_proba(X)[:, 1]           # column 1 = probability of "goes together"


def predict_compatibility(item1: Mapping[str, Any], item2: Mapping[str, Any]) -> float:
    """Return a score from 0 to 1 for how well two items go together.

    Each item is a dict with "image" (2,048 numbers) and "text" (256 numbers),
    the format of Sam's features.npz and featurize_photo().
    """
    return float(predict_many([item1], [item2])[0])   # a batch of one pair