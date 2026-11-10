
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "processed" / "images"
FEATURE_DIR = BASE_DIR / "processed" / "features"
CHECKPOINT_DIR = FEATURE_DIR / "resnet_checkpoints"
IDS_FILE = FEATURE_DIR / "required_item_ids.json"
MANIFEST_FILE = CHECKPOINT_DIR / "manifest.json"


class ClothingDataset(Dataset):
    def __init__(self, item_ids, transform):
        self.item_ids = item_ids
        self.transform = transform

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, index):
        item_id = self.item_ids[index]
        path = IMAGE_DIR / f"{item_id}.jpg"

        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))

        return item_id, tensor


def checkpoint_valid(path, expected_ids):
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["item_ids"].tolist()
            features = saved["image"]

            return (
                ids == expected_ids
                and features.shape == (len(expected_ids), 2048)
                and np.isfinite(features).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit-chunks", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("Batch and chunk sizes must be positive")

    print("=== Full ResNet-50 Feature Extraction ===", flush=True)

    with open(IDS_FILE, encoding="utf-8") as file:
        item_ids = sorted(set(map(str, json.load(file))))

    total_items = len(item_ids)
    total_chunks = (
        total_items + args.chunk_size - 1
    ) // args.chunk_size

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "item_ids_sha256": hashlib.sha256(
            json.dumps(item_ids).encode("utf-8")
        ).hexdigest(),
        "chunk_size": args.chunk_size,
        "model": "resnet50",
        "weights": "IMAGENET1K_V2",
        "feature_dimension": 2048,
    }

    # Prevent incompatible checkpoint configurations.
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as file:
            previous = json.load(file)

        if previous != manifest:
            raise RuntimeError(
                "Checkpoint configuration changed. "
                "Use the original chunk size and item list, "
                "or a separate checkpoint directory."
            )
    else:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    print(f"Required images: {total_items}", flush=True)
    print(f"Total chunks: {total_chunks}", flush=True)
    print(f"Device: CPU", flush=True)

    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()

    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to("cpu")
    model.eval()

    processed_this_run = 0
    skipped = 0

    for chunk_index in range(total_chunks):
        if (
            args.limit_chunks is not None
            and processed_this_run >= args.limit_chunks
        ):
            break

        start = chunk_index * args.chunk_size
        end = min(start + args.chunk_size, total_items)
        chunk_ids = item_ids[start:end]

        output_path = (
            CHECKPOINT_DIR / f"chunk_{chunk_index:05d}.npz"
        )

        if checkpoint_valid(output_path, chunk_ids):
            skipped += 1
            continue

        print(
            f"\nProcessing chunk {chunk_index + 1}/{total_chunks} "
            f"(items {start + 1}-{end})",
            flush=True,
        )

        dataset = ClothingDataset(chunk_ids, transform)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_features = []
        all_ids = []

        with torch.inference_mode():
            for ids, images in loader:
                features = model(images.to("cpu"))

                all_features.append(
                    features.cpu().numpy().astype(np.float32)
                )
                all_ids.extend(ids)

        feature_matrix = np.concatenate(
            all_features, axis=0
        )

        # Save atomically to protect against interruption.
        temp_path = output_path.with_suffix(".tmp.npz")

        np.savez_compressed(
            temp_path,
            item_ids=np.asarray(all_ids, dtype=str),
            image=feature_matrix,
        )

        temp_path.replace(output_path)

        processed_this_run += 1

        print(
            f"Saved {output_path.name} "
            f"| Shape: {feature_matrix.shape}",
            flush=True,
        )

    completed = 0
    saved_items = 0

    for chunk_index in range(total_chunks):
        start = chunk_index * args.chunk_size
        end = min(start + args.chunk_size, total_items)
        path = CHECKPOINT_DIR / f"chunk_{chunk_index:05d}.npz"

        if checkpoint_valid(path, item_ids[start:end]):
            completed += 1
            saved_items += end - start

    print("\n=== Extraction Status ===")
    print(f"Total required items: {total_items}")
    print(f"Processed this run: {processed_this_run} chunks")
    print(f"Skipped existing chunks: {skipped}")
    print(f"Completed chunks: {completed}/{total_chunks}")
    print(f"Saved image features: {saved_items}/{total_items}")

    if completed == total_chunks:
        print("SUCCESS: All ResNet image features extracted.")
    else:
        print("IN PROGRESS: Rerun to continue extraction.")


if __name__ == "__main__":
    main()
