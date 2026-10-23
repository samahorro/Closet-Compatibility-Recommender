# Closet Compatibility Recommender

An AI outfit builder for the clothes you already own. The planned application recommends combinations from a personal wardrobe and evaluates how versatile a potential purchase would be.

## Team
- James Ubiadas — dataset pipeline and feature extraction
- Sam Ahorro — compatibility models, baselines, evaluation
- Jose Polanco — constrained outfit search, versatility score
- Andrew Nguyen — Streamlit UI and photo processing

## Project status
Initial repository scaffold. The Streamlit interface is a placeholder; no trained model or Polyvore dataset is included.

## Setup (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

For ResNet feature extraction, PyTorch, torchvision and rembg, install the optional dependencies in `requirements-ml.txt` with versions appropriate to your machine/Colab runtime.

## Data
Use the Polyvore Outfits **disjoint** split, via the dataset instructions at https://github.com/mvasil/fashion-compatibility. Do not commit its large images or extracted vectors to Git. Store raw downloads under `data/raw/` and generated artifacts under `data/processed/`; these are gitignored.

## Model design
For each item, combine ResNet image features, TF-IDF title features, and category encodings. Generate positives from co-occurring items in training outfits and sample negatives from the training partition. Preserve disjoint evaluation boundaries. Compare logistic regression, MLP, and simple baselines, using ROC-AUC, pair accuracy and fill-in-the-blank ranking accuracy.

## Team workflow
Create feature branches, open pull requests, and review changes before merging into `main`. Agree on a single item-feature schema and compatibility prediction API before integration.

## Directory ownership
- `data/`, `notebooks/` — James
- `ml/` — Sam
- `recommender/` — Jose
- `app/` — Andrew
- `tests/`, `docs/` — shared
