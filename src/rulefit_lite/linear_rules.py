from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.utils.validation import check_array, check_is_fitted


@dataclass(frozen=True)
class LinearRule:
    """Простое правило по одному признаку: x_j > thr или x_j <= thr."""

    feat_idx: int
    op: str  # ">" or "<="
    thr: float

    def to_string(self, feature_names: Sequence[str] | None = None) -> str:
        name = feature_names[self.feat_idx] if feature_names is not None else f"x{self.feat_idx}"
        return f"{name} {self.op} {self.thr:.6g}"

    def mask(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feat_idx]
        if self.op == ">":
            return col > self.thr
        return col <= self.thr


def _get_feature_names(estimator, n_features: int) -> np.ndarray:
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        return np.array([f"x{i}" for i in range(n_features)], dtype=object)
    return np.array(list(names), dtype=object)


def _get_linear_coef_and_intercept(estimator) -> tuple[np.ndarray, np.ndarray]:
    """
    Достаёт coef_/intercept_ из:
      - sklearn линейных моделей
      - твоего SimpleRuleFitClassifier (возьмём coef_original_)
    """
    if hasattr(estimator, "coef_original_"):
        coef = estimator.coef_original_
        intercept = getattr(estimator, "intercept_", None)
        if intercept is None:
            intercept = np.zeros((coef.shape[0],), dtype=float)
        return np.asarray(coef), np.asarray(intercept)

    check_is_fitted(estimator, ["coef_", "intercept_"])
    return np.asarray(estimator.coef_), np.asarray(estimator.intercept_)


def extract_linear_threshold_rules(
    estimator,
    X_ref,
    *,
    y_ref=None,
    n_thresholds: int = 8,
    top_k: int = 20,
    use_simplify: bool = True,
    min_support: float = 0.02,
    round_thresholds: int = 6,
) -> list[dict[str, Any]]:
    """
    Строит “список правил” из весов линейной модели:
      - для каждого признака берём несколько порогов (квантили по X_ref)
      - оцениваем “силу” правила как |w_j| * (разница средних предиктов / просто |w_j| * std)
      - возвращаем top_k правил

    Это surrogate-объяснение (не влияет на предикт модели).
    """

    X_ref = check_array(X_ref, accept_sparse=False, dtype=np.float64)
    n, p = X_ref.shape
    feature_names = _get_feature_names(estimator, p)

    coef, _ = _get_linear_coef_and_intercept(estimator)
    # coef shape: (n_classes or 1, p)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    n_out = coef.shape[0]

    # Кандидатные пороги: квантили
    # берём внутренние квантили (без 0 и 1)
    qs = np.linspace(0.1, 0.9, num=max(2, n_thresholds))
    thresholds = [np.quantile(X_ref[:, j], qs) for j in range(p)]

    items: list[dict[str, Any]] = []

    # для каждого признака и порога создаём два направления: >thr и <=thr
    for j in range(p):
        wj = coef[:, j]  # (n_out,)
        # если вес почти нулевой — правила по этому признаку редко нужны
        if np.max(np.abs(wj)) == 0:
            continue

        for thr in thresholds[j]:
            for op in (">", "<="):
                rule = LinearRule(j, op, float(thr))
                m = rule.mask(X_ref)
                sup = float(m.mean())

                if use_simplify and sup < min_support:
                    continue

                # простой скоринг правила:
                # базовая “важность” = abs(weight) * sqrt(support*(1-support))
                # (чтобы не выбирать совсем редкие/совсем частые)
                balance = np.sqrt(max(1e-12, sup * (1.0 - sup)))
                score_vec = np.abs(wj) * balance  # (n_out,)

                weight_out: float | list[float]
                score_out: float
                if n_out == 1:
                    weight_out = float(wj[0])
                    score_out = float(score_vec[0])
                else:
                    weight_out = [float(v) for v in wj]
                    score_out = float(np.max(score_vec))

                items.append(
                    {
                        "rule": rule.to_string(feature_names),
                        "support": sup,
                        "weight": weight_out,
                        "score": score_out,
                        "feature": feature_names[j],
                    }
                )

    # сортируем по score
    items.sort(key=lambda d: float(d["score"]), reverse=True)

    # simplify эвристики
    if use_simplify:
        items = items[: int(top_k)]

        # округление порогов в тексте
        if round_thresholds is not None:
            items2 = []
            for it in items:
                tokens = it["rule"].split(" ")
                out = []
                for t in tokens:
                    try:
                        v = float(t)
                        t = str(np.round(v, int(round_thresholds)))
                    except Exception:
                        pass
                    out.append(t)
                it2 = dict(it)
                it2["rule"] = " ".join(out)
                items2.append(it2)
            items = items2

        # дедуп по rule
        seen = set()
        deduped = []
        for it in items:
            if it["rule"] in seen:
                continue
            seen.add(it["rule"])
            deduped.append(it)
        items = deduped

    # наружу score обычно не нужен
    for it in items:
        it.pop("score", None)

    return items
