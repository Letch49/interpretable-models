import argparse
import json
import os

import numpy as np

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


def best_thr_f1(y_true, p, grid=101):
    best_f = -1.0
    best_t = 0.5
    for t in np.linspace(0.0, 1.0, grid):
        f = f1_score(y_true, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t, float(best_f)


def metrics_from_proba(y_true, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="thresholds")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--max_rules", type=int, default=20)
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "rules"), exist_ok=True)

    all_rows = []

    for seed in seeds:
        X, y = make_case(args.case, seed=seed)

        # split train/val/test
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed)

        # Teacher HGB
        teacher = HistGradientBoostingClassifier(random_state=seed)
        teacher.fit(X_tr, y_tr)
        p_val_t = teacher.predict_proba(X_val)[:, 1]
        t_thr, t_f1 = best_thr_f1(y_val, p_val_t)
        p_te_t = teacher.predict_proba(X_te)[:, 1]
        m_teacher = metrics_from_proba(y_te, p_te_t, thr=t_thr)

        all_rows.append({"case": args.case, "seed": seed, "model": "HGB_teacher", **m_teacher, "rules": None})

        # A) LogisticRegression (weights-only) + surrogate threshold rules
        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]
        )
        lr.fit(X_tr, y_tr)
        p_val_lr = lr.predict_proba(X_val)[:, 1]
        lr_thr, _ = best_thr_f1(y_val, p_val_lr)
        p_te_lr = lr.predict_proba(X_te)[:, 1]
        m_lr = metrics_from_proba(y_te, p_te_lr, thr=lr_thr)
        all_rows.append({"case": args.case, "seed": seed, "model": "A_LogReg", **m_lr, "rules": None})

        # dump A rules
        Xtr_scaled = lr.named_steps["scaler"].transform(X_tr)
        lr_rules = extract_linear_threshold_rules(lr.named_steps["clf"], X_ref=Xtr_scaled, top_k=20, use_simplify=True)
        with open(
            os.path.join(args.out, "rules", f"{args.case}_seed{seed}_A_logreg_rules.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(lr_rules, f, ensure_ascii=False, indent=2)

        # B) DecisionTree (rules-only)
        dt = DecisionTreeClassifier(max_depth=4, random_state=seed)
        dt.fit(X_tr, y_tr)
        p_te_dt = dt.predict_proba(X_te)[:, 1]
        # tree threshold тоже тюним на val
        p_val_dt = dt.predict_proba(X_val)[:, 1]
        dt_thr, _ = best_thr_f1(y_val, p_val_dt)
        m_dt = metrics_from_proba(y_te, p_te_dt, thr=dt_thr)
        all_rows.append({"case": args.case, "seed": seed, "model": "B_Tree", **m_dt, "rules": "tree"})

        # dump tree text
        from sklearn.tree import export_text

        tree_txt = export_text(dt, feature_names=[f"x{i}" for i in range(X.shape[1])])
        with open(
            os.path.join(args.out, "rules", f"{args.case}_seed{seed}_B_tree_rules.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(tree_txt)

        # C) RuleFit-lite distilled to HGB + prune under teacher F1
        rf = SimpleRuleFitClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            max_rules=2500,
            max_rule_depth=3,
            min_rule_support=0.01,
            C=0.6,
            random_state=seed,
            simplify_top_k=args.max_rules,
        )

        info = rf.fit_distilled_hgb(
            X_tr,
            y_tr,
            X_val,
            y_val,
            eps=args.eps,
            max_rules_after=args.max_rules,
            distill_alpha=2.0,
        )

        p_te_rf = rf.predict_proba(X_te)[:, 1]
        m_rf = metrics_from_proba(y_te, p_te_rf, thr=rf.threshold_)

        all_rows.append(
            {
                "case": args.case,
                "seed": seed,
                "model": "C_RuleFit_distilled_pruned",
                **m_rf,
                "rules": len(rf.rules_),
                **{f"info_{k}": v for k, v in info.items()},
            }
        )

        # dump C rules
        rf_rules = rf.interpret(use_simplify=True)
        with open(
            os.path.join(args.out, "rules", f"{args.case}_seed{seed}_C_rulefit_rules.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(rf_rules, f, ensure_ascii=False, indent=2)

    # save json
    with open(os.path.join(args.out, f"{args.case}_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("Saved:", os.path.join(args.out, f"{args.case}_results.json"))


if __name__ == "__main__":
    main()
