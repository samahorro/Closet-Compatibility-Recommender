
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "processed" / "features"

RESNET_DIR = FEATURE_DIR / "resnet_checkpoints"
TEXT_FILE = FEATURE_DIR / "text_features.npz"
REQUIRED_IDS_FILE = FEATURE_DIR / "required_item_ids.json"
OUTPUT_FILE = FEATURE_DIR / "features.npz"

CHUNK_SIZE = 512
IMAGE_DIM = 2048
TEXT_DIM = 256


def main():
    print("=== Combining ResNet and TF-IDF Features ===")

    # --------------------------------------------------
    # 1. Load required item IDs
    # --------------------------------------------------

    with open(REQUIRED_IDS_FILE, "r", encoding="utf-8") as f:
        required_ids = sorted(set(map(str, json.load(f))))

    total_items = len(required_ids)
    print(f"Required items: {total_items}")

    # --------------------------------------------------
    # 2. Load TF-IDF features
    # --------------------------------------------------

    print("\nLoading TF-IDF features...")

    with np.load(TEXT_FILE, allow_pickle=False) as data:
        text_ids = data["item_ids"].astype(str)
        text_features = data["text"].astype(np.float32)

    if text_features.shape != (total_items, TEXT_DIM):
        raise ValueError(
            f"Unexpected text shape: {text_features.shape}"
        )

    if len(set(text_ids.tolist())) != len(text_ids):
        raise ValueError("Duplicate text item IDs detected")

    text_lookup = {
        item_id: i for i, item_id in enumerate(text_ids)
    }

    missing_text = set(required_ids) - set(text_lookup)

    if missing_text:
        raise ValueError(
            f"Missing TF-IDF features: {len(missing_text)}"
        )

    # Align text features with the required item ID order.
    text_indices = [
        text_lookup[item_id] for item_id in required_ids
    ]

    if text_indices != list(range(total_items)):
        text_features = text_features[text_indices]

    del text_indices, text_lookup

    print(f"Text shape: {text_features.shape}")

    # --------------------------------------------------
    # 3. Load ResNet feature chunks
    # --------------------------------------------------

    print("\nLoading ResNet checkpoint chunks...")

    chunks = sorted(RESNET_DIR.glob("chunk_*.npz"))

    expected_chunks = (
        total_items + CHUNK_SIZE - 1
    ) // CHUNK_SIZE

    if len(chunks) != expected_chunks:
        raise ValueError(
            f"Expected {expected_chunks} chunks, "
            f"but found {len(chunks)}"
        )

    image_features = np.empty(
        (total_items, IMAGE_DIM),
        dtype=np.float32
    )

    for chunk_index in range(expected_chunks):
        chunk_path = (
            RESNET_DIR / f"chunk_{chunk_index:05d}.npz"
        )

        if not chunk_path.exists():
            raise FileNotFoundError(chunk_path)

        start = chunk_index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_items)

        with np.load(chunk_path, allow_pickle=False) as data:
            chunk_ids = data["item_ids"].astype(str)
            chunk_images = data["image"]

            expected_ids = required_ids[start:end]

            if chunk_ids.tolist() != expected_ids:
                raise ValueError(
                    f"Item ID mismatch: {chunk_path.name}"
                )

            if chunk_images.shape != (
                len(expected_ids), IMAGE_DIM
            ):
                raise ValueError(
                    f"Invalid shape: {chunk_path.name}"
                )

            if not np.isfinite(chunk_images).all():
                raise ValueError(
                    f"Non-finite image features: "
                    f"{chunk_path.name}"
                )

            image_features[start:end] = chunk_images

        if (
            (chunk_index + 1) % 25 == 0
            or chunk_index + 1 == expected_chunks
        ):
            print(
                f"Loaded {chunk_index + 1}/"
                f"{expected_chunks} chunks"
            )

    print(f"Image shape: {image_features.shape}")

    # --------------------------------------------------
    # 4. Validate combined features
    # --------------------------------------------------

    print("\nValidating combined features...")

    if not np.isfinite(text_features).all():
        raise ValueError("Non-finite text features found")

    assert image_features.shape[0] == total_items
    assert text_features.shape[0] == total_items

    print("All feature rows are aligned.")
    print("No missing required items.")
    print("No duplicate item IDs.")

    # --------------------------------------------------
    # 5. Save combined feature file
    # --------------------------------------------------

    print("\nSaving features.npz...")

    np.savez_compressed(
        OUTPUT_FILE,
        item_ids=np.asarray(required_ids, dtype=str),
        image=image_features,
        text=text_features,
    )

    print("\n=== Final Feature Export ===")
    print(f"Item IDs: ({total_items},)")
    print(f"Image: {image_features.shape}")
    print(f"Text: {text_features.shape}")
    print(f"Saved: {OUTPUT_FILE}")
    print("SUCCESS: Combined features exported!")


if __name__ == "__main__":
    main()
