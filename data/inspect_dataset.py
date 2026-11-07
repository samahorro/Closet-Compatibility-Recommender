
import json
from pathlib import Path

# Directory containing the downloaded dataset
DATA_DIR = Path(__file__).resolve().parent / "raw" / "polyvore" / "disjoint"


def load_dataset(filename):
    file_path = DATA_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=== Polyvore Dataset Inspection ===")

    for filename in ["train.json", "valid.json", "test.json"]:
        dataset = load_dataset(filename)

        print(f"\nFile: {filename}")
        print(f"Total outfits: {len(dataset)}")

        if dataset:
            example = dataset[0]

            print("\nExample outfit:")
            print(json.dumps(example, indent=2)[:2500])

            print("\nAvailable fields:")
            print(list(example.keys()))


if __name__ == "__main__":
    main()
