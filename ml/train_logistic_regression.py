"""Sam: implement training after agreeing on a feature vector schema."""
from sklearn.linear_model import LogisticRegression


def build_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)
