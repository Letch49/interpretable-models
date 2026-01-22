import argparse
import json

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bench.generate_cases import make_case
from rulefit_lite import SimpleRuleFitClassifier
from rulefit_lite.linear_rules import extract_linear_threshold_rules


def eval_model(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    return {
        "f1": float(f1_score(yte, yhat)),
        "precision": float(precision_score(yte, yhat, zero_division=0)),
        "recall": float(recall_score(yte, yhat, zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="thresholds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="bench_results.json")
    ap.add_argument("--dump_rules_dir", default="bench_rules")
    args = ap.parse_args()

    X, y = make_case(args.case, seed=args.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)

    models = {}

    # A) weights-only baseline (LogReg)
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000, solver="lbfgs", class_weight="balanced" if args.case == "imbalanced" else None
                ),
            ),
        ]
    )
    models["A_logreg"] = lr

    # B) rules-only (DecisionTree)
    dt = DecisionTreeClassifier(
        max_depth=4, random_state=args.seed, class_weight="balanced" if args.case == "imbalanced" else None
    )
    models["B_tree"] = dt

    # Baseline black-box (HGB)
    hgb = HistGradientBoostingClassifier(random_state=args.seed)
    models["baseline_hgb"] = hgb

    # C) rules+weights (RuleFit-Lite)
    rf = SimpleRuleFitClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        max_rules=2000,
        max_rule_depth=3,
        min_rule_support=0.01,
        C=0.6,
        random_state=args.seed,
        simplify_top_k=20,
    )
    models["C_rulefit"] = rf

    results = {"case": args.case, "seed": args.seed, "models": {}}

    # Evaluate
    for name, model in models.items():
        m = eval_model(model, Xtr, Xte, ytr, yte)
        results["models"][name] = m

    # Dump explanations
    import os

    os.makedirs(args.dump_rules_dir, exist_ok=True)

    # A: convert weights -> threshold rules (surrogate)
    lr.fit(Xtr, ytr)
    Xtr_scaled = lr.named_steps["scaler"].transform(Xtr)
    lr_rules = extract_linear_threshold_rules(lr.named_steps["clf"], X_ref=Xtr_scaled, top_k=20, use_simplify=True)
    with open(os.path.join(args.dump_rules_dir, f"{args.case}_A_logreg_rules.json"), "w", encoding="utf-8") as f:
        json.dump(lr_rules, f, ensure_ascii=False, indent=2)

    # C: real rules
    rf.fit(Xtr, ytr)
    rf_rules = rf.interpret(use_simplify=True)
    with open(os.path.join(args.dump_rules_dir, f"{args.case}_C_rulefit_rules.json"), "w", encoding="utf-8") as f:
        json.dump(rf_rules, f, ensure_ascii=False, indent=2)

    # B: tree rules (как текст) — минимально: экспортируем sklearn tree
    from sklearn.tree import export_text

    tree_txt = export_text(dt.fit(Xtr, ytr), feature_names=[f"x{i}" for i in range(Xtr.shape[1])])
    with open(os.path.join(args.dump_rules_dir, f"{args.case}_B_tree_rules.txt"), "w", encoding="utf-8") as f:
        f.write(tree_txt)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
