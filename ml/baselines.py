"""Fill-in-the-blank baselines: random, category co-occurrence, visual similarity."""
from collections import Counter
import numpy as np


def random_scorer(seed=42):
    rng = np.random.default_rng(seed)
    return lambda question, cand: [rng.random()]


def category_cooccurrence_scorer(train_outfits, cat):
    """Score by how often the candidate's category appears with each question item's category."""
    co = Counter()
    for o in train_outfits:
        cs = [cat[i] for i in o["items"]]
        for x in cs:
            for y in cs:
                if x != y:
                    co[(x, y)] += 1
    return lambda question, cand: [co[(cat[i], cat[cand])] for i in question]


def visual_similarity_scorer(image_vectors, row):
    """Score by cosine similarity of image vectors (vectors have length 1)."""
    return lambda question, cand: [image_vectors[row[i]] @ image_vectors[row[cand]] for i in question]