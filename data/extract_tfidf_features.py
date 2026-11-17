"""Extract TF-IDF title features for every required Polyvore item.

Fit the vocabulary/IDF on training items only, then transform *all*
required items (including held-out FITB candidates) in a shared order.
Run from the repository root: python data/extract_tfidf_features.py
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent
FEATURE_DIR = DATA_DIR / "processed" / "features"
METADATA_FILE = DATA_DIR / "raw" / "polyvore" / "polyvore_item_metadata.json"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def title_for_item(record):
    """Match the same text fallbacks used by preprocess_dataset.py."""
    if not isinstance(record, dict):
        return ""
    for key in ("title", "url_name", "description"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_features(metadata_file=METADATA_FILE, feature_dir=FEATURE_DIR, max_features=256):
    required_ids = sorted(set(map(str, read_json(feature_dir / "required_item_ids.json"))))
    train_ids = sorted(set(map(str, read_json(feature_dir / "train_item_ids.json"))))
    metadata = read_json(metadata_file)

    if not required_ids or not train_ids:
        raise ValueError("Required and training item ID lists must not be empty")
    if not set(train_ids).issubset(required_ids):
        raise ValueError("Every training item must occur in required_item_ids.json")

    missing_metadata = [item_id for item_id in required_ids if item_id not in metadata]
    if missing_metadata:
        raise ValueError(f"Missing metadata for {len(missing_metadata)} items, examples: {missing_metadata[:5]}")

    all_titles = [title_for_item(metadata[item_id]) for item_id in required_ids]
    title_by_id = dict(zip(required_ids, all_titles))
    train_titles = [title_by_id[item_id] for item_id in train_ids]
    if not any(train_titles):
        raise ValueError("No usable titles found among training items")

    # Fitting on held-out titles would leak test-set vocabulary/IDF statistics.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
        dtype=np.float32,
        norm="l2",
    )
    vectorizer.fit(train_titles)
    text = vectorizer.transform(all_titles).toarray().astype(np.float32, copy=False)
    item_ids = np.asarray(required_ids, dtype=str)

    if text.shape[0] != len(item_ids) or not np.isfinite(text).all():
        raise RuntimeError("Invalid text feature matrix")

    feature_dir.mkdir(parents=True, exist_ok=True)
    destination = feature_dir / "text_features.npz"
    temp = feature_dir / "text_features.tmp.npz"
    np.savez_compressed(temp, item_ids=item_ids, text=text)
    temp.replace(destination)
    joblib.dump(vectorizer, feature_dir / "tfidf_vectorizer.joblib")

    print("=== TF-IDF title feature extraction ===")
    print(f"Training items used to fit TF-IDF: {len(train_ids)}")
    print(f"All required items transformed: {len(item_ids)}")
    print(f"Empty training titles: {sum(not t for t in train_titles)}")
    print(f"Empty titles across all items: {sum(not t for t in all_titles)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Text feature shape: {text.shape}")
    print(f"Saved: {destination}")
    print("Note: all validation/test/FITB IDs are included if listed in required_item_ids.json")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=256)
    args = parser.parse_args()
    if args.max_features < 1:
        parser.error("--max-features must be >= 1")
    extract_features(max_features=args.max_features)


if __name__ == "__main__":
    main()