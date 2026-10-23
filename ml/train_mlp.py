"""Sam: MLP experiment scaffold."""
from sklearn.neural_network import MLPClassifier


def build_model() -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=300)
