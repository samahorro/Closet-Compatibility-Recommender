
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "processed" / "images"
FEATURE_DIR = BASE_DIR / "processed" / "features"

IDS_FILE = FEATURE_DIR / "required_item_ids.json"


class ClothingDataset(Dataset):
    def __init__(self, item_ids, transform):
        self.item_ids = item_ids
        self.transform = transform

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        image_path = IMAGE_DIR / f"{item_id}.jpg"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return item_id, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=== ResNet-50 Feature Extraction ===")

    with open(IDS_FILE, encoding="utf-8") as file:
        item_ids = json.load(file)

    item_ids = sorted(map(str, item_ids))

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    item_ids = item_ids[:args.limit]

    missing = [
        item_id for item_id in item_ids
        if not (IMAGE_DIR / f"{item_id}.jpg").is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files are missing"
        )

    # Use CPU; pretrained weights are not trained here.
    device = torch.device("cpu")
    print(f"Device: {device}")

    weights = ResNet50_Weights.DEFAULT

    # Includes resizing, cropping and normalization
    # expected by pretrained ResNet-50.
    transform = weights.transforms()

    model = resnet50(weights=weights)

    # Remove the final classification layer.
    # Output becomes a 2048-dimensional vector.
    model.fc = torch.nn.Identity()

    model = model.to(device)
    model.eval()

    dataset = ClothingDataset(item_ids, transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_ids = []
    all_features = []

    with torch.inference_mode():
        for batch_index, (ids, images) in enumerate(loader):
            images = images.to(device)

            features = model(images)

            all_features.append(
                features.cpu().numpy()
            )
            all_ids.extend(ids)

            print(
                f"Processed batch {batch_index + 1}/"
                f"{len(loader)}"
            )

    feature_matrix = np.concatenate(
        all_features, axis=0
    ).astype(np.float32)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURE_DIR / "resnet_sample.npz"

    np.savez_compressed(
        output_path,
        item_ids=np.array(all_ids),
        image=feature_matrix,
    )

    print("\n=== Extraction Complete ===")
    print(f"Items processed: {len(all_ids)}")
    print(f"Feature shape: {feature_matrix.shape}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
