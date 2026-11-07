
from datasets import load_dataset, Image
from pathlib import Path
from PIL import Image as PILImage
from io import BytesIO

OUTPUT_DIR = Path(__file__).resolve().parent / "processed" / "sample_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Polyvore Image Extraction Test ===")

dataset = load_dataset(
    "mvasil/polyvore-outfits",
    "disjoint",
    split="train",
    streaming=True,
)

# Keep raw image bytes instead of automatically decoding.
dataset = dataset.cast_column("image", Image(decode=False))

record = next(iter(dataset))

item_id = record["item_id"]
image_data = record["image"]

print(f"Item ID: {item_id}")

if image_data["bytes"] is not None:
    image = PILImage.open(
        BytesIO(image_data["bytes"])
    ).convert("RGB")

    print(f"Image dimensions: {image.size}")

    output_path = OUTPUT_DIR / f"{item_id}.jpg"
    image.save(output_path, format="JPEG")

    print(f"Saved image: {output_path}")

else:
    print("No image bytes found in this record.")
