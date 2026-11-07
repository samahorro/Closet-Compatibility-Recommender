
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent / "raw" / "polyvore" / "disjoint"


def load_items(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        outfits = json.load(f)

    locations = defaultdict(set)

    for outfit in outfits:
        for item in outfit["items"]:
            locations[str(item["item_id"])].add(
                str(outfit["set_id"])
            )

    return locations


train = load_items("train.json")
valid = load_items("valid.json")

shared = sorted(set(train) & set(valid))

print("=== TRAIN / VALID OVERLAP ===")
print(f"Shared item IDs: {len(shared)}")

for item_id in shared[:10]:
    print(f"\nItem ID: {item_id}")
    print(f"Train outfits: {sorted(train[item_id])[:3]}")
    print(f"Valid outfits: {sorted(valid[item_id])[:3]}")
