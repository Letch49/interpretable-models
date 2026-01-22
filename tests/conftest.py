import numpy as np
import pytest

from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def datasets():
    """
    Набор датасетов под разные режимы:
      - linear: хорошо для линейки
      - nonlinear: нелинейность/взаимодействия (деревья/бустинги лучше)
      - noisy: шум + лишние фичи
    """
    out = {}

    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.4,
        flip_y=0.01,
        random_state=1,
    )
    out["linearish"] = (X, y)

    X, y = make_classification(
        n_samples=6000,
        n_features=30,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=0.8,
        flip_y=0.03,
        random_state=2,
    )
    # добавим “нелинейность” вручную: XOR-подобный признак
    z = (X[:, 0] > 0).astype(int) ^ (X[:, 1] > 0).astype(int)
    X = X.copy()
    X[:, 0] = z  # заменим один признак на дискретный нелинейный
    out["nonlinearish"] = (X, y)

    X, y = make_classification(
        n_samples=8000,
        n_features=80,
        n_informative=8,
        n_redundant=4,
        class_sep=1.0,
        flip_y=0.05,
        random_state=3,
    )
    out["noisy_highdim"] = (X, y)

    return out
