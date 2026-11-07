
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR = DATA_DIR / "item_disjoint"


def load_split(name):
    with open(DATA_DIR / f"{name}_cleaned.json", encoding="utf-8") as f:
        return json.load(f)


def get_item_ids(outfit):
    return {str(item["item_id"]) for item in outfit["items"]}


def save_split(name, outfits):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"{name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(outfits, f)

    print(f"Saved {name}: {len(outfits)} outfits")


def main():
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    train_items = set()
    for outfit in train:
        train_items.update(get_item_ids(outfit))

    # Only retain validation outfits with unseen items.
    valid_filtered = [
        outfit for outfit in valid
        if get_item_ids(outfit).isdisjoint(train_items)
    ]

    valid_items = set()
    for outfit in valid_filtered:
        valid_items.update(get_item_ids(outfit))

    # Exclude training and retained validation items.
    excluded_items = train_items | valid_items

    test_filtered = [
        outfit for outfit in test
        if get_item_ids(outfit).isdisjoint(excluded_items)
    ]

    save_split("train", train)
    save_split("valid", valid_filtered)
    save_split("test", test_filtered)

    print("\n=== FINAL DATASET ===")
    print(f"Training: {len(train)}")
    print(f"Validation: {len(valid_filtered)}")
    print(f"Testing: {len(test_filtered)}")


if __name__ == "__main__":
    main()
