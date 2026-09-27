"""Tier-1 纯 numpy 离线兜底（零 sklearn 可跑）。作者: 晨星

不变量：kNN 图传播的收敛序列 F_{t+1} = (1-α)Y0 + α·S·F_t，
S 为行归一化对称加权图，谱半径 ≤ α < 1，保证收敛（Zhu 2002）。
"""

from __future__ import annotations

import numpy as np


class NumpyKNNClassifier:
    """距离加权 kNN（predict_proba = 核加权投票归一）。"""

    name = "numpy_knn"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "NumpyKNNClassifier":
        self.X_ = X_l.copy()
        self.y_ = y_l.copy()
        self.classes_ = np.unique(y_l)
        return self

    def _topk(self, X: np.ndarray) -> tuple:
        d2 = (
            np.sum(X**2, axis=1)[:, None]
            - 2 * X @ self.X_.T
            + np.sum(self.X_**2, axis=1)[None, :]
        )
        k = min(self.k, len(self.X_))
        idx = np.argsort(d2, axis=1)[:, :k]
        dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
        return idx, dist

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        idx, dist = self._topk(X)
        w = 1.0 / (dist + 1e-8)
        P = np.zeros((len(X), len(self.classes_)))
        col = {c: i for i, c in enumerate(self.classes_)}
        for i in range(len(X)):
            for j, d in zip(idx[i], w[i]):
                P[i, col[self.y_[j]]] += d
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class NumpyLabelPropagate:
    """纯 numpy kNN 图标签传播（Zhu 2002 简化版，hard-label one-hot 初始化）。

    fit 时在 X_all = [X_l; X_u] 上建图，迭代 F ← (1-α)Y0 + α S F，
    收敛后对任意查询点用其图上近邻的 F 均值作为概率（归纳近似）。
    """

    name = "numpy_labelprop"

    def __init__(self, k: int = 10, alpha: float = 0.75, iters: int = 60) -> None:
        self.k = k
        self.alpha = alpha
        self.iters = iters

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "NumpyLabelPropagate":
        X_all = np.concatenate([X_l, X_u], axis=0)
        n_l, n_all = len(X_l), len(X_all)
        self.classes_ = np.unique(y_l)

        # 加权 kNN 图（对称化 + 行归一化）
        d2 = (
            np.sum(X_all**2, axis=1)[:, None]
            - 2 * X_all @ X_all.T
            + np.sum(X_all**2, axis=1)[None, :]
        )
        np.fill_diagonal(d2, np.inf)
        k = min(self.k, n_all - 1)
        idx = np.argsort(d2, axis=1)[:, :k]
        W = np.zeros((n_all, n_all))
        rows = np.repeat(np.arange(n_all), k)
        dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
        w = 1.0 / (dist + 1e-8)
        W[rows, idx.ravel()] = w.ravel()
        W = np.maximum(W, W.T)  # 对称化
        deg = W.sum(axis=1, keepdims=True)
        S = W / (deg + 1e-12)

        Y0 = np.zeros((n_all, len(self.classes_)))
        col = {c: i for i, c in enumerate(self.classes_)}
        for i, c in enumerate(y_l):
            Y0[i, col[c]] = 1.0

        F = Y0.copy()
        for _ in range(self.iters):
            F = (1.0 - self.alpha) * Y0 + self.alpha * (S @ F)
        self.F_ = F
        self.X_l_ = X_l.copy()
        self.F_l_ = F[:n_l].copy()
        return self

    def _propagate_query(self, X: np.ndarray) -> np.ndarray:
        # 查询点 -> 标注点 kNN，加权平均其收敛 F（归纳近似）
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
        return np.einsum("nk,nkc->nc", w, self.F_l_[idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = self._propagate_query(X)
        return P / (P.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class NumpySelfTrain:
    """纯 numpy 自训练（kNN 基学习器 + 固定阈值），SOTA 缺失时的兜底自训练。"""

    name = "numpy_selftrain"

    def __init__(self, k: int = 10, threshold: float = 0.8, max_iter: int = 15) -> None:
        self.k = k
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "NumpySelfTrain":
        X_train, y_train = X_l.copy(), y_l.copy()
        pool = X_u.copy()
        for _ in range(self.max_iter):
            if len(pool) == 0:
                break
            base = NumpyKNNClassifier(k=self.k).fit(
                X_train, y_train, np.empty((0, X_train.shape[1]))
            )
            P = base.predict_proba(pool)
            conf = P.max(axis=1)
            pick = conf >= self.threshold
            if not pick.any():
                break
            X_train = np.concatenate([X_train, pool[pick]], axis=0)
            y_train = np.concatenate([y_train, base.classes_[np.argmax(P[pick], axis=1)]])
            pool = pool[~pick]
        self.model_ = NumpyKNNClassifier(k=self.k).fit(
            X_train, y_train, np.empty((0, X_train.shape[1]))
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
