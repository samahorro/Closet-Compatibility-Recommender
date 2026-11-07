
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "polyvore"
DISJOINT_DIR = RAW_DIR / "disjoint"
OUTPUT_DIR = DATA_DIR / "processed"

RANDOM_SEED = 42
TRAIN_SUBSET_SIZE = 8000


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_outfits(outfits, metadata):
    cleaned = []
    missing_items = 0

    for outfit in outfits:
        items = []
        seen_ids = set()

        for item in outfit.get("items", []):
            item_id = str(item.get("item_id", ""))

            if not item_id or item_id in seen_ids:
                continue

            item_metadata = metadata.get(item_id)

            if not isinstance(item_metadata, dict):
                missing_items += 1
                continue

            category = item_metadata.get(
                "semantic_category", ""
            )

            if not category:
                continue

            title = (
                item_metadata.get("title")
                or item_metadata.get("url_name")
                or item_metadata.get("description")
                or ""
            )

            items.append({
                "item_id": item_id,
                "index": item.get("index"),
                "title": title.strip(),
                "category": category,
                "category_id": item_metadata.get(
                    "category_id"
                ),
            })

            seen_ids.add(item_id)

        # Require at least two items for compatibility learning
        if len(items) >= 2:
            cleaned.append({
                "set_id": str(outfit.get("set_id", "")),
                "items": items,
            })

    return cleaned, missing_items


def save_json(data, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Saved: {output_path}")


def main():
    random.seed(RANDOM_SEED)

    print("=== Polyvore Data Preprocessing ===")

    metadata = load_json(
        RAW_DIR / "polyvore_item_metadata.json"
    )

    for split in ["train", "valid", "test"]:
        outfits = load_json(
            DISJOINT_DIR / f"{split}.json"
        )

        print(f"\nProcessing {split}...")
        print(f"Original outfits: {len(outfits)}")

        cleaned, missing = clean_outfits(
            outfits, metadata
        )

        print(f"Valid outfits: {len(cleaned)}")
        print(f"Missing metadata items: {missing}")

        if split == "train":
            random.shuffle(cleaned)
            cleaned = cleaned[:TRAIN_SUBSET_SIZE]

        print(f"Selected outfits: {len(cleaned)}")

        save_json(
            cleaned,
            f"{split}_cleaned.json"
        )

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
