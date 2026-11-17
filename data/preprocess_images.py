
import argparse
import json
from io import BytesIO
from pathlib import Path

from datasets import load_dataset, Image
from PIL import Image as PILImage, ImageOps

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FILE = BASE_DIR / "processed/features/required_item_ids.json"
OUTPUT_DIR = BASE_DIR / "processed/images"

IMAGE_SIZE = (224, 224)


def preprocess_image(image_bytes, output_path):
    with PILImage.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

        # Maintain aspect ratio.
        image.thumbnail(IMAGE_SIZE, PILImage.Resampling.LANCZOS)

        # Place the clothing image on a white canvas.
        canvas = PILImage.new("RGB", IMAGE_SIZE, (255, 255, 255))

        x = (224 - image.width) // 2
        y = (224 - image.height) // 2

        canvas.paste(image, (x, y))

        temporary = output_path.with_suffix(".tmp.jpg")
        canvas.save(temporary, format="JPEG", quality=95)
        temporary.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum images to save during this run",
    )
    args = parser.parse_args()

    with open(REQUIRED_FILE, encoding="utf-8") as file:
        required_ids = set(map(str, json.load(file)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remaining = {
        item_id
        for item_id in required_ids
        if not (OUTPUT_DIR / f"{item_id}.jpg").exists()
    }

    print("=== Polyvore Image Preprocessing ===")
    print(f"Required items: {len(required_ids)}")
    print(f"Already saved: {len(required_ids) - len(remaining)}")
    print(f"Remaining: {len(remaining)}")

    saved = 0
    failures = {}

    for split in ["train", "validation", "test"]:
        if not remaining:
            break

        if args.limit is not None and saved >= args.limit:
            break

        print(f"\nReading {split} images...")

        dataset = load_dataset(
            "mvasil/polyvore-outfits",
            "disjoint",
            split=split,
            streaming=True,
        )

        dataset = dataset.cast_column(
            "image", Image(decode=False)
        )

        for record in dataset:
            item_id = str(record["item_id"])

            if item_id not in remaining:
                continue

            try:
                image_bytes = record["image"].get("bytes")

                if not image_bytes:
                    raise ValueError("Image bytes unavailable")

                output_path = OUTPUT_DIR / f"{item_id}.jpg"
                preprocess_image(image_bytes, output_path)

                remaining.remove(item_id)
                saved += 1

                if saved % 10 == 0:
                    print(f"Saved {saved} images this run")

            except (OSError, ValueError) as error:
                failures[item_id] = str(error)

            if args.limit is not None and saved >= args.limit:
                break

    report = {
        "required": len(required_ids),
        "saved_total": len(required_ids) - len(remaining),
        "missing": len(remaining),
        "failed_images": failures,
    }

    report_path = BASE_DIR / "processed/features/image_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n=== Results ===")
    print(f"Saved this run: {saved}")
    print(f"Total saved: {report['saved_total']}")
    print(f"Remaining: {report['missing']}")


if __name__ == "__main__":
    main()
