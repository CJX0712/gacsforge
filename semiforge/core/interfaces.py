"""统一接口 Protocol：分数语义 = macro-F1 越大越好。作者: 晨星"""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class SSLMethod(Protocol):
    """半监督方法统一契约。

    fit(X_l, y_l, X_u): 少量标注 + 无标注池（transductive 信息）。
    predict_proba(X): 返回 (n, n_classes) 概率，列序 = sorted(np.unique(y_l))。
    """

    name: str

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "SSLMethod": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def classes_of(y_l: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(sorted(np.unique(y_l).tolist()))
