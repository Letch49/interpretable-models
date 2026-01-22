from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        """Return boolean mask of samples that satisfy the rule."""
        if len(self.conditions) == 0:
            return np.ones(X.shape[0], dtype=bool)
        mask = np.ones(X.shape[0], dtype=bool)
        for c in self.conditions:
            col = X[:, c.feat_idx]
            if c.op == "<=":
                mask &= col <= c.thr
            else:
                mask &= col > c.thr
            # early exit
            if not mask.any():
                break
        return mask


class SimpleRuleFitClassifier(BaseEstimator, ClassifierMixin):
    """
    Минимальный RuleFit-подобный классификатор в стиле sklearn.

    Идея:
      1) учим ансамбль деревьев (GradientBoostingClassifier)
      2) извлекаем правила (пути до листьев)
      3) превращаем правила в бинарные фичи
      4) учим L1-логистическую регрессию на [исходные фичи + правила]

    Ограничения (минимальный вариант):
      - ожидаются числовые признаки (категориальные -> one-hot/ordinal заранее)
      - правило: конъюнкция порогов вида (x_j <= t) и (x_k > t)
    """

    def __init__(
        self,
        # base trees
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
        l1_ratio: float = 1.0,  # kept for API symmetry; LogisticRegression uses pure L1 here
        max_iter: int = 5000,
        # interpretation / simplify
        simplify_top_k: int = 20,
        simplify_min_support: float = 0.02,
        simplify_round_thresholds: int = 6,
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
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

        self.simplify_top_k = simplify_top_k
        self.simplify_min_support = simplify_min_support
        self.simplify_round_thresholds = simplify_round_thresholds

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # feature names
        self.feature_names_in_ = getattr(X, "columns", None)
        if self.feature_names_in_ is None:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            self.feature_names_in_ = np.array(list(self.feature_names_in_), dtype=object)

        # 1) fit tree ensemble
        self._gb_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._gb_.fit(X, y)
        self.classes_ = self._gb_.classes_

        # 2) extract rules
        rules = self._extract_rules_from_gb(self._gb_, max_rules=self.max_rules, max_depth=self.max_rule_depth)
        rules = self._dedupe_and_filter_by_support(rules, X, min_support=self.min_rule_support)
        self.rules_ = rules  # list[_Rule]
        self.rule_strings_ = [r.to_string(self.feature_names_in_) for r in self.rules_]

        R = self._build_rule_matrix(X, self.rules_)  # (n_samples, n_rules) sparse or dense
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
        self._lr_.fit(Z, y)

        # keep coefficients split for interpretability
        coef = self._lr_.coef_
        self.coef_original_ = coef[:, : self.n_features_in_]
        self.coef_rules_ = coef[:, self.n_features_in_ :]
        self.intercept_ = self._lr_.intercept_.copy()

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["_gb_", "_lr_", "_scaler_", "rules_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        Xs = self._scaler_.transform(X)
        R = self._build_rule_matrix(X, self.rules_)
        Z = self._hstack(Xs, R)
        return self._lr_.predict_proba(Z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        check_is_fitted(self, ["_gb_", "_lr_", "_scaler_", "rules_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        Xs = self._scaler_.transform(X)
        R = self._build_rule_matrix(X, self.rules_)
        Z = self._hstack(Xs, R)
        return self._lr_.decision_function(Z)

    def interpret(self, use_simplify: bool = True) -> list[dict[str, Any]]:
        """
        Возвращает список правил с весами.
        Для мультикласса: вес для каждого класса (one-vs-rest в sklearn).

        Формат элемента:
          {
            "rule": "x1 <= 3.2 AND x5 > 0.1",
            "support": 0.123,
            "weight": float | List[float]
          }
        """
        check_is_fitted(self, ["_lr_", "rules_", "rule_strings_"])

        # weights for rules
        W = self.coef_rules_  # shape: (n_classes or 1, n_rules)
        if W.ndim == 1:
            W = W.reshape(1, -1)

        # compute support from training cache if exists, else approximate not available
        supports = getattr(self, "rule_supports_", None)
        if supports is None:
            supports = np.full(len(self.rules_), np.nan, dtype=float)

        items = []
        for j, rule_str in enumerate(self.rule_strings_):
            wj = W[:, j]
            weight: float | list[float]
            weight = float(wj[0]) if W.shape[0] == 1 else [float(v) for v in wj]

            items.append(
                {
                    "rule": rule_str,
                    "support": float(supports[j]) if np.isfinite(supports[j]) else None,
                    "weight": weight,
                }
            )

        # remove zero-ish weights
        def _absmax_weight(it: dict[str, Any]) -> float:
            w = it["weight"]
            if isinstance(w, list):
                return float(np.max(np.abs(w)))
            return abs(float(w))

        items = [it for it in items if _absmax_weight(it) > 0.0]

        if not use_simplify:
            # just sort by importance
            items.sort(key=_absmax_weight, reverse=True)
            return items

        # --- simplify heuristics (default minimal working) ---
        # 1) support filter
        if any(it["support"] is not None for it in items):
            items = [it for it in items if (it["support"] is None or it["support"] >= self.simplify_min_support)]

        # 2) sort by abs(weight) and take top_k
        items.sort(key=_absmax_weight, reverse=True)
        items = items[: int(self.simplify_top_k)]

        # 3) round thresholds in rule text (cosmetic simplification)
        if self.simplify_round_thresholds is not None:
            items = [self._round_rule_thresholds(it, ndigits=int(self.simplify_round_thresholds)) for it in items]

        # 4) dedupe identical text after rounding
        seen = set()
        deduped = []
        for it in items:
            key = it["rule"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        return deduped

    # ---------- internals ----------

    def _round_rule_thresholds(self, it: dict[str, Any], ndigits: int) -> dict[str, Any]:
        # very lightweight: round any float-like tokens
        # Safe-ish: parse tokens and round those that look like numbers.
        rule = it["rule"]
        tokens = rule.split(" ")
        out = []
        for t in tokens:
            try:
                # if token is numeric
                v = float(t)
                t = str(np.round(v, ndigits))
            except Exception:
                pass
            out.append(t)
        it2 = dict(it)
        it2["rule"] = " ".join(out)
        return it2

    def _hstack(self, Xs: np.ndarray, R):
        if sp is not None and sp.issparse(R):
            return sp.hstack([sp.csr_matrix(Xs), R], format="csr")
        # dense
        if isinstance(R, np.ndarray):
            return np.hstack([Xs, R])
        # fallback
        return np.hstack([Xs, np.asarray(R)])

    def _build_rule_matrix(self, X, rules, store_supports: bool = False):
        n = X.shape[0]
        m = len(rules)
        if m == 0:
            if sp is not None:
                return sp.csr_matrix((n, 0), dtype=np.int8)
            return np.zeros((n, 0), dtype=np.int8)

        # build sparse matrix column-by-column
        if sp is None:
            R = np.zeros((n, m), dtype=np.int8)
            for j, r in enumerate(rules):
                R[:, j] = r.evaluate_mask(X).astype(np.int8)
            return R

        indptr = [0]
        indices = []
        data = []
        supports = np.zeros(m, dtype=float)

        for j, r in enumerate(rules):
            mask = r.evaluate_mask(X)
            idx = np.flatnonzero(mask)
            indices.extend(idx.tolist())
            data.extend([1] * len(idx))
            indptr.append(len(indices))
            supports[j] = len(idx) / max(1, n)

        # store supports from latest call (fit uses training X)
        supports = np.zeros(m, dtype=float)
        if store_supports:
            self.rule_supports_ = supports

        # We built CSC-like arrays but will create CSC and convert to CSR for speed in LR
        R_csc = sp.csc_matrix(
            (np.array(data, dtype=np.int8), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
            shape=(n, m),
        )
        return R_csc.tocsr()

    def _dedupe_and_filter_by_support(self, rules: list[_Rule], X: np.ndarray, min_support: float):
        if len(rules) == 0:
            self.rule_supports_ = np.array([], dtype=float)
            return rules

        # canonicalize to string key with rounded thresholds (so duplicates collapse)
        def key_for_rule(r: _Rule) -> str:
            # sort conditions by feat/op/thr for stable key
            conds = sorted(r.conditions, key=lambda c: (c.feat_idx, c.op, c.thr))
            # keep original order? For AND it doesn't matter; canonical helps dedupe
            parts = [f"{c.feat_idx}{c.op}{np.round(c.thr, 10)}" for c in conds]
            return "&".join(parts)

        uniq: dict[str, _Rule] = {}
        for r in rules:
            k = key_for_rule(r)
            if k not in uniq:
                uniq[k] = r

        rules_uniq = list(uniq.values())

        # support filter
        supports = []
        kept = []
        n = X.shape[0]
        for r in rules_uniq:
            sup = float(r.evaluate_mask(X).sum()) / max(1, n)
            if sup >= min_support:
                kept.append(r)
                supports.append(sup)

        self.rule_supports_ = np.array(supports, dtype=float)
        return kept

    def _extract_rules_from_gb(
        self,
        gb: GradientBoostingClassifier,
        max_rules: int,
        max_depth: int,
    ) -> list[_Rule]:
        # GradientBoostingClassifier.estimators_ shape:
        # binary: (n_estimators, 1), multiclass: (n_estimators, n_classes)
        est = gb.estimators_
        rules: list[_Rule] = []

        # iterate trees, breadth-first extraction of leaf paths
        for i in range(est.shape[0]):
            for k in range(est.shape[1]):
                tree = est[i, k].tree_
                rules.extend(self._tree_leaf_rules(tree, max_depth=max_depth))
                if len(rules) >= max_rules:
                    return rules[:max_rules]

        return rules[:max_rules]

    def _tree_leaf_rules(self, tree, max_depth: int) -> list[_Rule]:
        # sklearn tree structure arrays
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        rules: list[_Rule] = []

        # stack: (node_id, depth, conditions_so_far)
        stack: list[tuple[int, int, list[_Condition]]] = [(0, 0, [])]

        while stack:
            node_id, depth, conds = stack.pop()
            is_leaf = children_left[node_id] == children_right[node_id]

            if is_leaf:
                rules.append(_Rule(tuple(conds)))
                continue

            if depth >= max_depth:
                # stop expanding further; convert this node to a "truncated" rule
                rules.append(_Rule(tuple(conds)))
                continue

            f = feature[node_id]
            thr = float(threshold[node_id])

            # left: x[f] <= thr
            conds_left = [*conds, _Condition(int(f), "<=", thr)]
            # right: x[f] > thr
            conds_right = [*conds, _Condition(int(f), ">", thr)]

            stack.append((children_left[node_id], depth + 1, conds_left))
            stack.append((children_right[node_id], depth + 1, conds_right))

        return rules

    # Optional: sklearn-friendly names for transformed feature space
    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, ["rules_"])
        original = np.array(self.feature_names_in_, dtype=object)
        rules = np.array([f"RULE: {s}" for s in self.rule_strings_], dtype=object)
        return np.concatenate([original, rules])


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=6, n_redundant=2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    clf = SimpleRuleFitClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        max_rules=1500,
        max_rule_depth=3,
        min_rule_support=0.01,
        C=0.5,
        simplify_top_k=15,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print("AUC:", roc_auc_score(yte, p))

    rules = clf.interpret(use_simplify=True)
    for r in rules[:10]:
        print(r)
