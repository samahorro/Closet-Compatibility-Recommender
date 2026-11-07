
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DISJOINT_DIR = DATA_DIR / "raw" / "polyvore" / "disjoint"
PROCESSED_DIR = DATA_DIR / "processed"


def load_item_ids(path):
    with open(path, "r", encoding="utf-8") as file:
        outfits = json.load(file)

    return {
        str(item["item_id"])
        for outfit in outfits
        for item in outfit["items"]
    }


def check_overlaps(label, paths):
    print(f"\n=== {label} ===")

    splits = {
        name: load_item_ids(path)
        for name, path in paths.items()
    }

    for name, ids in splits.items():
        print(f"{name}: {len(ids)} unique items")

    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]

    for first, second in pairs:
        overlap = splits[first] & splits[second]

        print(
            f"{first} / {second} overlap: "
            f"{len(overlap)}"
        )

        if overlap:
            print(
                "Example overlapping IDs:",
                sorted(overlap)[:5]
            )


def main():
    check_overlaps(
        "ORIGINAL DISJOINT DATASET",
        {
            "train": DISJOINT_DIR / "train.json",
            "valid": DISJOINT_DIR / "valid.json",
            "test": DISJOINT_DIR / "test.json",
        },
    )

    check_overlaps(
        "PROCESSED DATASET",
        {
            "train": PROCESSED_DIR / "train_cleaned.json",
            "valid": PROCESSED_DIR / "valid_cleaned.json",
            "test": PROCESSED_DIR / "test_cleaned.json",
        },
    )


if __name__ == "__main__":
    main()

