"""Tier-0 sklearn 经典 SSL 包装（sklearn 缺失时自动跳过）。作者: 晨星

踩坑备忘：
- SelfTrainingClassifier 接口是 fit(X_all, y_all)（无标注=-1），在此封装。
- sklearn 1.9 概率为 float32 概率行和差 1e-7 → 归一化兜底。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import MethodError

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

    SKLEARN_OK = True
    _LabelSpreading = LabelSpreading
    _LabelPropagation = LabelPropagation
except ImportError:  # pragma: no cover
    SKLEARN_OK = False
    _LabelSpreading = None
    _LabelPropagation = None


def _norm(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


class SelfTrainingRF:
    """强经典基线：sklearn SelfTrainingClassifier(RF, threshold=0.75)。

    仅用绝对置信阈值，无类平衡、无图平滑 —— 旗舰 GACS 的直接对照。
    """

    name = "selftrain_rf"

    def __init__(
        self, rf_trees: int = 120, seed: int = 0, threshold: float = 0.75, max_iter: int = 20
    ) -> None:
        if not SKLEARN_OK:
            raise MethodError("sklearn 不可用")
        self.rf_trees = rf_trees
        self.seed = seed
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "SelfTrainingRF":
        self.classes_ = np.unique(y_l)
        X_all = np.concatenate([X_l, X_u], axis=0)
        y_all = np.concatenate([y_l, np.full(len(X_u), -1)])
        base = RandomForestClassifier(
            n_estimators=self.rf_trees, random_state=self.seed, n_jobs=1
        )
        self.model_ = SelfTrainingClassifier(
            base, threshold=self.threshold, max_iter=self.max_iter
        )
        self.model_.fit(X_all, y_all)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _norm(self.model_.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


class _GraphBase:
    """LabelPropagation / LabelSpreading 公共包装（全图 transductive + 归纳近邻近似）。"""

    _cls = None

    def __init__(self, k: int = 10, seed: int = 0) -> None:
        if not SKLEARN_OK:
            raise MethodError("sklearn 不可用")
        self.k = k
        self.seed = seed

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "_GraphBase":
        self.classes_ = np.unique(y_l)
        X_all = np.concatenate([X_l, X_u], axis=0)
        y_all = np.concatenate([y_l, np.full(len(X_u), -1)])
        self.model_ = self._cls(kernel="knn", n_neighbors=self.k)
        self.model_.fit(X_all, y_all)
        self.X_l_ = X_l.copy()
        self.P_l_ = _norm(self.model_.predict_proba(X_l))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d2 = (
            np.sum(X**2, axis=1)[:, None]
            - 2 * X @ self.X_l_.T
            + np.sum(self.X_l_**2, axis=1)[None, :]
        )
        k = min(self.k, len(self.X_l_))
        idx = np.argsort(d2, axis=1)[:, :k]
        dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
        w = 1.0 / (dist + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkc->nc", w, self.P_l_[idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LabelSpreadSK(_GraphBase):
    name = "label_spread"
    _cls = _LabelSpreading


class LabelPropSK(_GraphBase):
    name = "label_prop"
    _cls = _LabelPropagation
