
import json
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "raw" / "polyvore"


def main():
    print("=== Polyvore Metadata Inspection ===")

    # Load metadata
    metadata_path = DATA_DIR / "polyvore_item_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print(f"\nMetadata type: {type(metadata).__name__}")
    print(f"Total metadata records: {len(metadata)}")

    # Display an example without assuming its structure
    if isinstance(metadata, dict):
        first_key = next(iter(metadata))
        print(f"\nExample item ID: {first_key}")
        print(json.dumps(metadata[first_key], indent=2)[:2000])

    elif isinstance(metadata, list):
        print("\nExample metadata record:")
        print(json.dumps(metadata[0], indent=2)[:2000])

    # Inspect categories
    categories_path = DATA_DIR / "categories.csv"

    print("\n=== Clothing Categories ===")

    with open(
        categories_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file)

        for index, row in enumerate(reader):
            print(row)

            if index >= 9:
                break


if __name__ == "__main__":
    main()
