from ml.train_logistic_regression import build_model as lr_model
from ml.train_mlp import build_model as mlp_model


def test_models_construct():
    assert lr_model().max_iter == 1000
    assert mlp_model().hidden_layer_sizes == (128, 64)
