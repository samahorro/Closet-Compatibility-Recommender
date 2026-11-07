
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent / "processed"


def validate_split(split):
    path = DATA_DIR / f"{split}_cleaned.json"

    with open(path, "r", encoding="utf-8") as file:
        outfits = json.load(file)

    categories = Counter()
    unique_items = set()
    empty_titles = 0
    duplicate_items = 0
    total_items = 0
    outfit_ids = set()

    for outfit in outfits:
        outfit_ids.add(outfit["set_id"])
        seen = set()

        for item in outfit["items"]:
            item_id = str(item["item_id"])
            total_items += 1

            if item_id in seen:
                duplicate_items += 1

            seen.add(item_id)
            unique_items.add(item_id)

            categories[item["category"]] += 1

            if not item.get("title", "").strip():
                empty_titles += 1

    print(f"\n=== {split.upper()} VALIDATION ===")
    print(f"Outfits: {len(outfits)}")
    print(f"Total item occurrences: {total_items}")
    print(f"Unique clothing items: {len(unique_items)}")
    print(f"Empty titles: {empty_titles}")
    print(f"Duplicate items within outfits: {duplicate_items}")

    print("\nClothing categories:")
    for category, count in categories.most_common():
        print(f"  {category}: {count}")

    return unique_items, outfit_ids


def main():
    train_items, train_outfits = validate_split("train")
    valid_items, valid_outfits = validate_split("valid")
    test_items, test_outfits = validate_split("test")

    print("\n=== ITEM LEAKAGE CHECK ===")
    print(f"Train / Valid overlap: {len(train_items & valid_items)}")
    print(f"Train / Test overlap: {len(train_items & test_items)}")
    print(f"Valid / Test overlap: {len(valid_items & test_items)}")

    print("\n=== OUTFIT LEAKAGE CHECK ===")
    print(f"Train / Valid overlap: {len(train_outfits & valid_outfits)}")
    print(f"Train / Test overlap: {len(train_outfits & test_outfits)}")
    print(f"Valid / Test overlap: {len(valid_outfits & test_outfits)}")


if __name__ == "__main__":
    main()
