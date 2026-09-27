"""基线方法。作者: 晨星"""

from __future__ import annotations

import numpy as np


class MajorityClassifier:
    """Naive 基线：永远预测标注集中多数类。"""

    name = "majority"

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "MajorityClassifier":
        classes, counts = np.unique(y_l, return_counts=True)
        self.classes_ = classes
        self.majority_ = classes[int(np.argmax(counts))]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        P = np.zeros((n, len(self.classes_)))
        P[:, int(np.where(self.classes_ == self.majority_)[0][0])] = 1.0
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_, dtype=np.int64)


class SupervisedRF:
    """强基线：仅用标注集训练 RandomForest（不用无标注池）。"""

    name = "supervised_rf"

    def __init__(self, rf_trees: int = 120, seed: int = 0) -> None:
        self.rf_trees = rf_trees
        self.seed = seed

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "SupervisedRF":
        from sklearn.ensemble import RandomForestClassifier

        self.model_ = RandomForestClassifier(
            n_estimators=self.rf_trees, random_state=self.seed, n_jobs=1
        )
        self.model_.fit(X_l, y_l)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
