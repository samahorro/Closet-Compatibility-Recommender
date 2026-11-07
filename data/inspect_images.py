
from datasets import load_dataset

print("=== Polyvore Image Dataset Inspection ===")

dataset = load_dataset(
    "mvasil/polyvore-outfits",
    "disjoint",
    split="train",
    streaming=True,
)

print("\nAvailable columns:")
print(dataset.column_names)

print("\nDataset features:")
print(dataset.features)

print("\nFirst record:")
record = next(iter(dataset))

for key, value in record.items():
    if key == "image":
        print(f"{key}: {type(value).__name__}")
    else:
        print(f"{key}: {str(value)[:200]}")
