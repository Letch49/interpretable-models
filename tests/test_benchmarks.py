from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rulefit_lite import SimpleRuleFitClassifier


def _auc(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def test_models_quality_ordering_smoke(datasets):
    """
    Smoke test: проверяем, что всё обучается/предсказывает и
    что на "nonlinearish" бустинги обычно не хуже линейки.

    Важно: это не доказательство, а регресс-тест от поломок.
    """
    X, y = datasets["nonlinearish"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    dt = DecisionTreeClassifier(max_depth=4, random_state=0)
    gb = GradientBoostingClassifier(random_state=0)
    hgb = HistGradientBoostingClassifier(random_state=0)
    rf = SimpleRuleFitClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        max_rules=1500,
        max_rule_depth=3,
        min_rule_support=0.01,
        C=0.5,
        random_state=0,
    )

    auc_lr = _auc(lr, Xtr, Xte, ytr, yte)
    auc_dt = _auc(dt, Xtr, Xte, ytr, yte)
    auc_gb = _auc(gb, Xtr, Xte, ytr, yte)
    auc_hgb = _auc(hgb, Xtr, Xte, ytr, yte)
    auc_rf = _auc(rf, Xtr, Xte, ytr, yte)

    # sanity: должны быть в разумных пределах
    for v in [auc_lr, auc_dt, auc_gb, auc_hgb, auc_rf]:
        assert 0.5 <= v <= 1.0

    # ожидаем, что бустинг часто >= линейки на нелинейности (допускаем флуктуации)
    assert max(auc_gb, auc_hgb, auc_rf) >= auc_lr - 0.02
