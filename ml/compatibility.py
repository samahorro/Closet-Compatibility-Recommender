"""Integration contract for trained pairwise compatibility model."""
from typing import Mapping, Any


def predict_compatibility(item1: Mapping[str, Any], item2: Mapping[str, Any]) -> float:
    """Return a score in [0, 1]. To be implemented after feature schema agreement.

    Both items must use the same item-feature schema agreed with the data team.
    """
    raise NotImplementedError("Train and load a compatibility model first")
