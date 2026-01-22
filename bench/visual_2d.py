import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from rulefit_lite import SimpleRuleFitClassifier


def make_2d_xor(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    # шум
    flip = rng.random(n) < 0.03
    y[flip] = 1 - y[flip]
    return X, y


def plot_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, proba, levels=20, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=8)
    ax.set_title(title)


def main():
    X, y = make_2d_xor()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    models = {
        "LogReg": LogisticRegression(max_iter=5000).fit(Xtr, ytr),
        "Tree(d=3)": DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr),
        "HGB": HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr),
        "RuleFit-Lite": SimpleRuleFitClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, max_rules=1000, C=0.6, random_state=0
        ).fit(Xtr, ytr),
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for ax, (name, m) in zip(axs, models.items()):
        yhat = m.predict(Xte)
        f1 = f1_score(yte, yhat)
        plot_boundary(ax, m, Xte, yte, f"{name} | F1={f1:.3f}")

    plt.tight_layout()
    plt.savefig("bench_2d_xor.png", dpi=200)
    print("Saved: bench_2d_xor.png")

    # Пример правил RuleFit
    rf_rules = models["RuleFit-Lite"].interpret(use_simplify=True)
    print("Top rules:", rf_rules[:8])


if __name__ == "__main__":
    main()
