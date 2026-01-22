import numpy as np

from sklearn.datasets import make_classification


def make_case(name: str, seed: int = 0):
    rng = np.random.default_rng(seed)

    if name == "linear":
        X, y = make_classification(
            n_samples=8000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            class_sep=1.6,
            flip_y=0.01,
            random_state=seed,
        )
        return X, y

    if name == "xor":
        X, y = make_classification(
            n_samples=8000, n_features=10, n_informative=4, n_redundant=0, class_sep=0.8, flip_y=0.02, random_state=seed
        )
        z = ((X[:, 0] > 0).astype(int) ^ (X[:, 1] > 0).astype(int)).astype(float)
        X = X.copy()
        X[:, 0] = z
        return X, y

    if name == "thresholds":
        n = 9000
        X = rng.normal(size=(n, 12))
        y = (((X[:, 0] > 0.5) & (X[:, 1] < -0.2)) | (X[:, 2] > 1.0)).astype(int)
        flip = rng.random(n) < 0.05
        y[flip] = 1 - y[flip]
        return X, y

    if name == "noisy_highdim":
        X, y = make_classification(
            n_samples=10000,
            n_features=80,
            n_informative=8,
            n_redundant=4,
            class_sep=1.0,
            flip_y=0.05,
            random_state=seed,
        )
        return X, y

    if name == "imbalanced":
        X, y = make_classification(
            n_samples=12000,
            n_features=30,
            n_informative=8,
            n_redundant=4,
            weights=[0.9, 0.1],
            class_sep=1.1,
            flip_y=0.03,
            random_state=seed,
        )
        return X, y

    raise ValueError(f"Unknown case: {name}")
