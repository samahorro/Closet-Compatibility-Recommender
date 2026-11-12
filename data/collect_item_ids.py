
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "polyvore" / "disjoint"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "features"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=== Collecting Required Clothing Item IDs ===")

    # Map outfit-position references to actual clothing IDs.
    item_lookup = {}

    for split in ["train", "valid", "test"]:
        outfits = load_json(RAW_DIR / f"{split}.json")

        for outfit in outfits:
            set_id = str(outfit["set_id"])

            for item in outfit["items"]:
                reference = f"{set_id}_{item['index']}"
                item_lookup[reference] = str(item["item_id"])

    print(f"Indexed item references: {len(item_lookup)}")

    required_ids = set()
    train_ids = set()

    # Collect items from the cleaned outfit datasets.
    for split in ["train", "valid", "test"]:
        outfits = load_json(
            PROCESSED_DIR / f"{split}_cleaned.json"
        )

        split_ids = {
            str(item["item_id"])
            for outfit in outfits
            for item in outfit["items"]
        }

        required_ids.update(split_ids)

        if split == "train":
            train_ids = split_ids

        print(f"{split}: {len(split_ids)} unique items")

    # Add all items referenced by FITB evaluation.
    fitb = load_json(RAW_DIR / "fill_in_blank_test.json")

    missing_references = set()
    fitb_ids = set()

    for question in fitb:
        references = (
            question.get("question", [])
            + question.get("answers", [])
        )

        for reference in references:
            item_id = item_lookup.get(str(reference))

            if item_id is None:
                missing_references.add(str(reference))
            else:
                fitb_ids.add(item_id)

    required_ids.update(fitb_ids)

    print(f"FITB unique items: {len(fitb_ids)}")
    print(f"Unresolved FITB references: {len(missing_references)}")

    # Verify metadata coverage.
    metadata = load_json(
        BASE_DIR / "raw" / "polyvore"
        / "polyvore_item_metadata.json"
    )

    missing_metadata = required_ids - set(metadata)

    print(f"Total required items: {len(required_ids)}")
    print(f"Missing metadata items: {len(missing_metadata)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "required_item_ids.json": sorted(required_ids),
        "train_item_ids.json": sorted(train_ids),
        "unresolved_fitb.json": sorted(missing_references),
    }

    for filename, data in outputs.items():
        with open(
            OUTPUT_DIR / filename,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(data, file, indent=2)

        print(f"Saved: {filename}")

    if missing_references or missing_metadata:
        print("\nWARNING: Some items require investigation.")
    else:
        print("\nAll item references resolved successfully!")


if __name__ == "__main__":
    main()
