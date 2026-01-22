from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rulefit_lite import SimpleRuleFitClassifier
from rulefit_lite.linear_rules import extract_linear_threshold_rules


def test_rulefit_interpret_returns_rules(datasets):
    X, y = datasets["linearish"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    clf = SimpleRuleFitClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        max_rules=800,
        max_rule_depth=3,
        min_rule_support=0.01,
        C=0.8,
        simplify_top_k=15,
        random_state=0,
    )
    clf.fit(Xtr, ytr)

    rules_simple = clf.interpret(use_simplify=True)
    rules_full = clf.interpret(use_simplify=False)

    assert isinstance(rules_simple, list)
    assert isinstance(rules_full, list)

    # simplify должен не увеличивать количество
    assert len(rules_simple) <= len(rules_full)

    # каждое правило должно иметь поля
    for r in rules_simple[:5]:
        assert "rule" in r and isinstance(r["rule"], str)
        assert "weight" in r
        assert "support" in r  # может быть None, но ключ должен быть


def test_linear_weight_rules_extraction(datasets):
    X, y = datasets["noisy_highdim"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    lr.fit(Xtr, ytr)

    # extract rules from the *pipeline* -> передаём последний estimator, чтобы были coef_
    rules = extract_linear_threshold_rules(
        lr.named_steps["clf"],
        X_ref=lr.named_steps["scaler"].transform(Xtr),  # важно: если coef_ обучались на scaled
        top_k=20,
        use_simplify=True,
    )

    assert isinstance(rules, list)
    assert len(rules) <= 20
    if len(rules) > 0:
        assert "rule" in rules[0]
        assert "weight" in rules[0]


def test_simplify_reduces_or_equal_rules(datasets):
    X, y = datasets["nonlinearish"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    clf = SimpleRuleFitClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        max_rules=1200,
        max_rule_depth=3,
        min_rule_support=0.005,
        C=0.7,
        simplify_top_k=10,
        simplify_min_support=0.02,
        random_state=0,
    ).fit(Xtr, ytr)

    full_rules = clf.interpret(use_simplify=False)
    simp_rules = clf.interpret(use_simplify=True)

    assert len(simp_rules) <= len(full_rules)
    assert len(simp_rules) <= 10
