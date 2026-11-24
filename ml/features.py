"""Pair features shared by training and predict_compatibility().

Training (in the notebook) and the app both turn two items into one row
of numbers with this same function, so the model always sees rows built
the same way it learned from.
"""
import numpy as np


def pair_matrix(img_a, txt_a, img_b, txt_b):
    """Build one row of features per pair of items.

    Each argument is an array with one row per pair:
      img_a, img_b: image vectors of the first and second items (256 numbers each)
      txt_a, txt_b: text vectors of the first and second items (256 numbers each)
    Returns an array with one row of 1,024 numbers per pair.
    """
    return np.hstack([          # hstack glues the four pieces side by side into one row
        img_a * img_b,          # where the two images agree; summed, this is their cosine similarity
        np.abs(img_a - img_b),  # how different the two images are, trait by trait
        txt_a * txt_b,          # the same two ideas for the titles
        np.abs(txt_a - txt_b),
    ]).astype("float32")        # float32 uses half the memory of the default float64