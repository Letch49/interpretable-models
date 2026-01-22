from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass(frozen=True)
class _Condition:
    feat_idx: int
    op: str  # "<=" or ">"
    thr: float


@dataclass(frozen=True)
class _Rule:
    conditions: tuple[_Condition, ...]

    def to_string(self, feature_names: Sequence[str]) -> str:
        parts = []
        for c in self.conditions:
            name = feature_names[c.feat_idx] if feature_names is not None else f"x{c.feat_idx}"
            parts.append(f"{name} {c.op} {c.thr:.6g}")
        return " AND ".join(parts) if parts else "<TRUE>"

    def evaluate_mask(self, X: np.ndarray) -> np.ndarray:
        if len(self.conditions) == 0:
            return np.ones(X.shape[0], dtype=bool)
        mask = np.ones(X.shape[0], dtype=bool)
        for c in self.conditions:
            col = X[:, c.feat_idx]
            if c.op == "<=":
                mask &= col <= c.thr
            else:
                mask &= col > c.thr
            if not mask.any():
                break
        return mask


class SimpleRuleFitClassifier(BaseEstimator, ClassifierMixin):
    """
    RuleFit-lite (binary focus) + дополнительные фичи для paper-экспериментов:
      - sample_weight (для distillation)
      - подбор threshold_ по F1 на валидации
      - distill from HGB teacher
      - prune rules under target F1 constraint (greedy)
    """

    def __init__(
        self,
        # base trees for rule generation
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int | None = 42,
        # rule generation limits
        max_rules: int = 2000,
        max_rule_depth: int = 3,
        min_rule_support: float = 0.01,  # fraction of samples
        # linear model
        C: float = 1.0,
        max_iter: int = 5000,
        # interpretation / simplify (output only)
        simplify_top_k: int = 20,
        simplify_min_support: float = 0.02,
        simplify_round_thresholds: int = 6,
        # threshold tuning
        optimize_threshold: bool = True,
        threshold_grid_size: int = 101,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        self.max_rules = max_rules
        self.max_rule_depth = max_rule_depth
        self.min_rule_support = min_rule_support

        self.C = C
        self.max_iter = max_iter

        self.simplify_top_k = simplify_top_k
        self.simplify_min_support = simplify_min_support
        self.simplify_round_thresholds = simplify_round_thresholds

        self.optimize_threshold = optimize_threshold
        self.threshold_grid_size = threshold_grid_size

    # ---------- sklearn API ----------

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # feature names
        self.feature_names_in_ = getattr(X, "columns", None)
        if self.feature_names_in_ is None:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            self.feature_names_in_ = np.array(list(self.feature_names_in_), dtype=object)

        # 1) fit tree ensemble (rule generator)
        self._gb_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._gb_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._gb_.classes_

        # binary only for the paper pipeline (можно расширить позже)
        if len(self.classes_) != 2:
            raise ValueError("This implementation currently targets binary classification for paper experiments.")

        # 2) extract rules
        rules = self._extract_rules_from_gb(self._gb_, max_rules=self.max_rules, max_depth=self.max_rule_depth)
        rules = self._dedupe_and_filter_by_support(rules, X, min_support=self.min_rule_support)
        self.rules_ = rules
        self.rule_strings_ = [r.to_string(self.feature_names_in_) for r in self.rules_]

        # 3) build rule matrix (no side-effects)
        R = self._build_rule_matrix(X, self.rules_)

        # 4) fit linear model on [scaled X + rules]
        self._scaler_ = StandardScaler(with_mean=True, with_std=True)
        Xs = self._scaler_.fit_transform(X)
        Z = self._hstack(Xs, R)

        self._lr_ = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=None,
            random_state=self.random_state,
        )
        self._lr_.fit(Z, y, sample_weight=sample_weight)

        # split coefficients for interpretability
        coef = self._lr_.coef_
        self.coef_original_ = coef[:, : self.n_features_in_]
        self.coef_rules_ = coef[:, self.n_features_in_ :]
        self.intercept_ = self._lr_.intercept_.copy()

        # store train supports (fixed)
        self.rule_supports_ = self._compute_rule_supports(X, self.rules_)

        # default threshold
        self.threshold_ = 0.5

        # optional threshold tuning on TRAIN (для paper лучше тюнить на val через tune_threshold)
        if self.optimize_threshold:
            p = self.predict_proba(X)[:, 1]
            self.threshold_, _ = self._best_threshold_f1(y, p)

        return self

    def tune_threshold(self, X_val, y_val):
        """Тюнит порог по F1 на валидации."""
        check_is_fitted(self, ["_lr_", "_scaler_", "rules_"])
        X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)
        p = self.predict_proba(X_val)[:, 1]
        thr, best = self._best_threshold_f1(y_val, p)
        self.threshold_ = thr
        return thr, best

    def predict_proba(self, X):
        check_is_fitted(self, ["_gb_", "_lr_", "_scaler_", "rules_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        Xs = self._scaler_.transform(X)
        R = self._build_rule_matrix(X, self.rules_)
        Z = self._hstack(Xs, R)
        return self._lr_.predict_proba(Z)

    def predict(self, X):
        proba = self.predict_proba(X)
        # binary: use tuned threshold_
        p1 = proba[:, 1]
        thr = getattr(self, "threshold_", 0.5)
        return np.where(p1 >= thr, self.classes_[1], self.classes_[0])

    def decision_function(self, X):
        check_is_fitted(self, ["_gb_", "_lr_", "_scaler_", "rules_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        Xs = self._scaler_.transform(X)
        R = self._build_rule_matrix(X, self.rules_)
        Z = self._hstack(Xs, R)
        return self._lr_.decision_function(Z)

    # ---------- paper: distillation + pruning ----------

    def fit_distilled_hgb(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        eps: float = 0.0,
        max_rules_after: int = 20,
        distill_alpha: float = 2.0,
        teacher_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Teacher: HistGradientBoostingClassifier.
        Student: this RuleFit-lite trained on pseudo labels + confidence weights,
        then pruned to keep F1 >= teacher_F1 - eps (measured on val, with threshold tuned).
        """
        X_train = check_array(X_train, accept_sparse=False, dtype=np.float64)
        X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)

        # 1) fit teacher
        teacher_params = teacher_params or {}
        teacher = HistGradientBoostingClassifier(random_state=self.random_state, **teacher_params)
        teacher.fit(X_train, y_train)

        # teacher F1 with its own tuned threshold on val
        pt = teacher.predict_proba(X_val)[:, 1]
        t_thr, t_f1 = self._best_threshold_f1(y_val, pt)

        # 2) distill labels + weights on train
        p_tr = teacher.predict_proba(X_train)[:, 1]
        y_soft = (p_tr >= 0.5).astype(int)  # hard pseudo labels
        conf = np.abs(p_tr - 0.5) * 2.0  # [0..1]
        sample_weight = 1.0 + distill_alpha * conf

        # 3) fit student on distilled labels
        self.fit(X_train, y_soft, sample_weight=sample_weight)

        # tune student threshold on val (w.r.t true y)
        self.tune_threshold(X_val, y_val)
        p0 = self.predict_proba(X_val)[:, 1]
        f1_0 = f1_score(y_val, (p0 >= self.threshold_).astype(int))

        # 4) prune under teacher target
        target = float(t_f1) - float(eps)
        prune_info = self.prune_to_target_f1(
            X_train,
            y_soft,
            sample_weight,
            X_val,
            y_val,
            target_f1=target,
            max_rules_after=max_rules_after,
        )

        # final student F1
        p1 = self.predict_proba(X_val)[:, 1]
        f1_1 = f1_score(y_val, (p1 >= self.threshold_).astype(int))

        return {
            "teacher_f1_val": float(t_f1),
            "teacher_thr_val": float(t_thr),
            "student_f1_val_before_prune": float(f1_0),
            "student_f1_val_after_prune": float(f1_1),
            "student_rules_before": int(prune_info["rules_before"]),
            "student_rules_after": int(prune_info["rules_after"]),
        }

    def prune_to_target_f1(
        self,
        X_train,
        y_train,
        sample_weight,
        X_val,
        y_val,
        *,
        target_f1: float,
        max_rules_after: int = 20,
    ) -> dict[str, Any]:
        """
        Greedy rule deletion with refit (LR only) while keeping F1 >= target_f1 on val.
        Uses |coef| importance.
        """
        check_is_fitted(self, ["_lr_", "_scaler_", "rules_"])

        X_train = check_array(X_train, accept_sparse=False, dtype=np.float64)
        X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)

        if len(self.rules_) == 0:
            return {"rules_before": 0, "rules_after": 0}

        # active rules = nonzero coef
        w = self.coef_rules_[0]  # binary
        active = np.flatnonzero(np.abs(w) > 0.0)
        rules_before = len(active)

        # if already small
        if rules_before <= max_rules_after:
            return {"rules_before": rules_before, "rules_after": rules_before}

        # rank by ascending importance (remove weakest first)
        order = active[np.argsort(np.abs(w[active]))]

        keep = set(active.tolist())

        # cached transforms for speed
        Xs_tr = self._scaler_.transform(X_train)
        Xs_val = self._scaler_.transform(X_val)
        R_tr_full = self._build_rule_matrix(X_train, self.rules_)
        R_val_full = self._build_rule_matrix(X_val, self.rules_)

        def refit_and_score(keep_idx: np.ndarray) -> tuple[float, float, LogisticRegression]:
            # refit LR on subset of rules
            R_tr = self._slice_cols(R_tr_full, keep_idx)
            R_val = self._slice_cols(R_val_full, keep_idx)
            Z_tr = self._hstack(Xs_tr, R_tr)
            Z_val = self._hstack(Xs_val, R_val)

            lr = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=self.C,
                max_iter=self.max_iter,
                n_jobs=None,
                random_state=self.random_state,
            )
            lr.fit(Z_tr, y_train, sample_weight=sample_weight)
            p = lr.predict_proba(Z_val)[:, 1]
            thr, best = self._best_threshold_f1(y_val, p)
            return float(best), float(thr), lr

        # baseline (current)
        base_keep = np.array(sorted(list(keep)), dtype=int)
        best_f1, best_thr, best_lr = refit_and_score(base_keep)

        # if even baseline < target, we still prune "best effort" but won't claim constraint
        for idx in order:
            if len(keep) <= max_rules_after:
                break

            # try remove
            keep_candidate = keep.copy()
            keep_candidate.remove(int(idx))
            keep_idx = np.array(sorted(list(keep_candidate)), dtype=int)

            cand_f1, cand_thr, cand_lr = refit_and_score(keep_idx)

            if cand_f1 >= target_f1:
                keep = keep_candidate
                best_f1, best_thr, best_lr = cand_f1, cand_thr, cand_lr

        # commit pruned model: shrink rules_ and refit lr on final keep
        final_keep = np.array(sorted(list(keep)), dtype=int)

        # update rule list & strings & supports
        self.rules_ = [self.rules_[i] for i in final_keep.tolist()]
        self.rule_strings_ = [r.to_string(self.feature_names_in_) for r in self.rules_]
        self.rule_supports_ = self._compute_rule_supports(X_train, self.rules_)

        # refit LR on final set (recompute matrices aligned to new rules_)
        R_tr = self._build_rule_matrix(X_train, self.rules_)
        R_val = self._build_rule_matrix(X_val, self.rules_)
        Z_tr = self._hstack(Xs_tr, R_tr)
        Z_val = self._hstack(Xs_val, R_val)

        self._lr_ = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=None,
            random_state=self.random_state,
        )
        self._lr_.fit(Z_tr, y_train, sample_weight=sample_weight)

        coef = self._lr_.coef_
        self.coef_original_ = coef[:, : self.n_features_in_]
        self.coef_rules_ = coef[:, self.n_features_in_ :]
        self.intercept_ = self._lr_.intercept_.copy()

        # tune threshold on val wrt true y
        p = self._lr_.predict_proba(Z_val)[:, 1]
        self.threshold_, _ = self._best_threshold_f1(y_val, p)

        return {"rules_before": rules_before, "rules_after": len(self.rules_), "best_f1_val": best_f1}

    # ---------- interpretation ----------

    def interpret(self, use_simplify: bool = True) -> list[dict[str, Any]]:
        check_is_fitted(self, ["_lr_", "rules_", "rule_strings_"])

        W = self.coef_rules_
        if W.ndim == 1:
            W = W.reshape(1, -1)

        supports = getattr(self, "rule_supports_", None)
        if supports is None:
            supports = np.full(len(self.rules_), np.nan, dtype=float)

        items = []
        for j, rule_str in enumerate(self.rule_strings_):
            wj = W[:, j]
            weight = float(wj[0])
            items.append(
                {
                    "rule": rule_str,
                    "support": float(supports[j]) if np.isfinite(supports[j]) else None,
                    "weight": weight,
                }
            )

        items = [it for it in items if abs(float(it["weight"])) > 0.0]

        def _absw(it):
            return abs(float(it["weight"]))

        if not use_simplify:
            items.sort(key=_absw, reverse=True)
            return items

        if any(it["support"] is not None for it in items):
            items = [it for it in items if (it["support"] is None or it["support"] >= self.simplify_min_support)]

        items.sort(key=_absw, reverse=True)
        items = items[: int(self.simplify_top_k)]

        if self.simplify_round_thresholds is not None:
            nd = int(self.simplify_round_thresholds)
            items2 = []
            for it in items:
                tokens = it["rule"].split(" ")
                out = []
                for t in tokens:
                    try:
                        v = float(t)
                        t = str(np.round(v, nd))
                    except Exception:
                        pass
                    out.append(t)
                it2 = dict(it)
                it2["rule"] = " ".join(out)
                items2.append(it2)
            items = items2

        seen = set()
        deduped = []
        for it in items:
            if it["rule"] in seen:
                continue
            seen.add(it["rule"])
            deduped.append(it)

        return deduped

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, ["rules_"])
        original = np.array(self.feature_names_in_, dtype=object)
        rules = np.array([f"RULE: {s}" for s in self.rule_strings_], dtype=object)
        return np.concatenate([original, rules])

    # ---------- utilities ----------

    def _best_threshold_f1(self, y_true, p1) -> tuple[float, float]:
        y_true = np.asarray(y_true).astype(int)
        p1 = np.asarray(p1)

        grid = np.linspace(0.0, 1.0, int(self.threshold_grid_size))
        best_f1 = -1.0
        best_thr = 0.5
        for thr in grid:
            yhat = (p1 >= thr).astype(int)
            f = f1_score(y_true, yhat, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = float(thr)
        return best_thr, float(best_f1)

    def _slice_cols(self, M, cols: np.ndarray):
        if sp is not None and sp.issparse(M):
            return M[:, cols]
        return M[:, cols]

    def _hstack(self, Xs: np.ndarray, R):
        if sp is not None and sp.issparse(R):
            return sp.hstack([sp.csr_matrix(Xs), R], format="csr")
        if isinstance(R, np.ndarray):
            return np.hstack([Xs, R])
        return np.hstack([Xs, np.asarray(R)])

    def _build_rule_matrix(self, X: np.ndarray, rules: list[_Rule]):
        n = X.shape[0]
        m = len(rules)
        if m == 0:
            if sp is not None:
                return sp.csr_matrix((n, 0), dtype=np.int8)
            return np.zeros((n, 0), dtype=np.int8)

        if sp is None:
            R = np.zeros((n, m), dtype=np.int8)
            for j, r in enumerate(rules):
                R[:, j] = r.evaluate_mask(X).astype(np.int8)
            return R

        # sparse CSR
        indptr = [0]
        indices = []
        data = []

        for r in rules:
            mask = r.evaluate_mask(X)
            idx = np.flatnonzero(mask)
            indices.extend(idx.tolist())
            data.extend([1] * len(idx))
            indptr.append(len(indices))

        R_csc = sp.csc_matrix(
            (np.array(data, dtype=np.int8), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
            shape=(n, m),
        )
        return R_csc.tocsr()

    def _compute_rule_supports(self, X: np.ndarray, rules: list[_Rule]) -> np.ndarray:
        n = X.shape[0]
        if len(rules) == 0:
            return np.array([], dtype=float)
        sup = np.zeros(len(rules), dtype=float)
        for j, r in enumerate(rules):
            sup[j] = float(r.evaluate_mask(X).sum()) / max(1, n)
        return sup

    def _dedupe_and_filter_by_support(self, rules: list[_Rule], X: np.ndarray, min_support: float):
        if len(rules) == 0:
            return rules

        def key_for_rule(r: _Rule) -> str:
            conds = sorted(r.conditions, key=lambda c: (c.feat_idx, c.op, c.thr))
            parts = [f"{c.feat_idx}{c.op}{np.round(c.thr, 10)}" for c in conds]
            return "&".join(parts)

        uniq: dict[str, _Rule] = {}
        for r in rules:
            k = key_for_rule(r)
            if k not in uniq:
                uniq[k] = r

        rules_uniq = list(uniq.values())

        kept = []
        n = X.shape[0]
        for r in rules_uniq:
            sup = float(r.evaluate_mask(X).sum()) / max(1, n)
            if sup >= min_support:
                kept.append(r)
        return kept

    def _extract_rules_from_gb(self, gb: GradientBoostingClassifier, max_rules: int, max_depth: int) -> list[_Rule]:
        est = gb.estimators_
        rules: list[_Rule] = []
        for i in range(est.shape[0]):
            for k in range(est.shape[1]):
                tree = est[i, k].tree_
                rules.extend(self._tree_leaf_rules(tree, max_depth=max_depth))
                if len(rules) >= max_rules:
                    return rules[:max_rules]
        return rules[:max_rules]

    def _tree_leaf_rules(self, tree, max_depth: int) -> list[_Rule]:
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        rules: list[_Rule] = []
        stack: list[tuple[int, int, list[_Condition]]] = [(0, 0, [])]

        while stack:
            node_id, depth, conds = stack.pop()
            is_leaf = children_left[node_id] == children_right[node_id]

            if is_leaf:
                rules.append(_Rule(tuple(conds)))
                continue

            if depth >= max_depth:
                rules.append(_Rule(tuple(conds)))
                continue

            f = int(feature[node_id])
            thr = float(threshold[node_id])

            stack.append((children_left[node_id], depth + 1, conds + [_Condition(f, "<=", thr)]))
            stack.append((children_right[node_id], depth + 1, conds + [_Condition(f, ">", thr)]))

        return rules
