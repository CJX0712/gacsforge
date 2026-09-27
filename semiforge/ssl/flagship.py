"""旗舰自研：GACS v2 —— Graph-Arborescence Class-balanced Self-training（图-树混合自训练）。作者: 晨星

设计（对照经典 SelfTrainingClassifier / LabelSpreading 的差异）：
1. 双分支评分：图分支 F = (1-α)Y0 + αSF（标注 one-hot 全图传播，流形结构）
   与树分支 RF 自训练概率混合 P_mix = β·P_rf + (1-β)·P_graph；
2. β 由验证集网格选择（val macro-F1 最大化，无测试泄漏）；
3. 类平衡选择：逐类预算 ∝ 类先验 + 逐类置信分位阈值（池内相对阈值）；
4. 课程退火：分位数 q_t = q0·anneal^t 随轮次下降；
5. 置信度样本加权（weight = 混合置信度）。

离线兜底：sklearn 缺失时树分支自动降级纯 numpy kNN，图分支本就纯 numpy。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import MethodError
from ..eval.metrics import macro_f1
from .numpy_fallback import NumpyKNNClassifier

try:
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_OK = True
except ImportError:  # pragma: no cover
    SKLEARN_OK = False

BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _knn_S(X: np.ndarray, k: int) -> tuple:
    """对称加权 kNN 图：返回 (S 行归一化, 距离矩阵 topk 索引)。"""
    n = len(X)
    d2 = np.sum(X**2, axis=1)[:, None] - 2 * X @ X.T + np.sum(X**2, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    kk = min(k, n - 1)
    idx = np.argsort(d2, axis=1)[:, :kk]
    dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), kk)
    W[rows, idx.ravel()] = (1.0 / (dist + 1e-8)).ravel()
    W = np.maximum(W, W.T)
    S = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    return S, d2, idx


class GACS:
    """Graph-Arborescence Class-balanced Self-Training（旗舰）。"""

    name = "gacs"

    def __init__(
        self,
        rf_trees: int = 80,
        seed: int = 0,
        rounds: int = 8,
        alpha: float = 0.75,
        quantile: float = 0.85,
        anneal: float = 0.9,
        graph_k: int = 10,
        balance: bool = True,
        use_graph: bool = True,
        use_anneal: bool = True,
        lp_iters: int = 15,
    ) -> None:
        self.rf_trees = rf_trees
        self.seed = seed
        self.rounds = rounds
        self.alpha = alpha
        self.quantile = quantile
        self.anneal = anneal
        self.graph_k = graph_k
        self.balance = balance
        self.use_graph = use_graph
        self.use_anneal = use_anneal
        self.lp_iters = lp_iters
        if not (0.5 <= quantile < 1.0):
            raise MethodError("quantile 必须在 [0.5, 1)")

    # ---------- 验证集注入（用于 β 选择，可选） ----------
    def set_validation(self, X_val: np.ndarray, y_val: np.ndarray) -> "GACS":
        self._X_val = X_val
        self._y_val = y_val
        return self

    def _new_base(self):
        if SKLEARN_OK:
            return RandomForestClassifier(
                n_estimators=self.rf_trees, random_state=self.seed, n_jobs=1
            )
        return NumpyKNNClassifier(k=self.graph_k)

    # ---------- 图分支：标注 one-hot 全图传播 ----------
    def _graph_branch(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> None:
        X_all = np.concatenate([X_l, X_u], axis=0)
        self.X_all_ = X_all
        n_l = len(X_l)
        S, d2_all, _ = _knn_S(X_all, self.graph_k)
        K = len(self.classes_)
        Y0 = np.zeros((len(X_all), K))
        col = {c: i for i, c in enumerate(self.classes_)}
        for i, c in enumerate(y_l):
            Y0[i, col[c]] = 1.0
        F = Y0.copy()
        for _ in range(self.lp_iters):
            F = (1.0 - self.alpha) * Y0 + self.alpha * (S @ F)
        self.F_all_ = F
        self.F_pool_ = F[n_l:]

    def _graph_proba(self, X: np.ndarray) -> np.ndarray:
        d2 = (
            np.sum(X**2, axis=1)[:, None]
            - 2 * X @ self.X_all_.T
            + np.sum(self.X_all_**2, axis=1)[None, :]
        )
        k = min(self.graph_k, len(self.X_all_))
        idx = np.argsort(d2, axis=1)[:, :k]
        dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
        w = 1.0 / (dist + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkc->nc", w, self.F_all_[idx])

    # ---------- β 选择（val macro-F1 网格） ----------
    def _select_beta(self, P_rf_val: np.ndarray, P_graph_val: np.ndarray) -> float:
        if not self.use_graph or getattr(self, "_X_val", None) is None:
            return 0.5 if self.use_graph else 1.0
        best_b, best_f1 = 0.5, -1.0
        for b in BETA_GRID:
            P = b * P_rf_val + (1 - b) * P_graph_val
            pred = self.classes_[np.argmax(P, axis=1)]
            f1 = macro_f1(self._y_val, pred)
            if f1 > best_f1 + 1e-12:
                best_b, best_f1 = b, f1
        return float(best_b)

    # ---------- 主训练 ----------
    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> "GACS":
        self.classes_ = np.unique(y_l)
        self._X_val = getattr(self, "_X_val", None)
        self._y_val = getattr(self, "_y_val", None)
        priors = np.array([np.mean(y_l == c) for c in self.classes_], dtype=np.float64)

        # 图分支（一次建图传播）
        if self.use_graph and len(X_u) > 3:
            self._graph_branch(X_l, y_l, X_u)
        else:
            self.use_graph = False
            self.beta_ = 1.0

        # 初始树分支 + β 选择
        base0 = self._new_base()
        if SKLEARN_OK:
            base0.fit(X_l, y_l)
        else:
            base0.fit(X_l, y_l, X_l[:0])
        if self.use_graph:
            has_val = self._X_val is not None and len(self._X_val) > 0
            if has_val:
                P_graph_val = self._graph_proba(self._X_val)
                P_rf_val = self._align(
                    np.asarray(base0.predict_proba(self._X_val), dtype=np.float64),
                    base0.classes_ if hasattr(base0, "classes_") else None,
                )
                self.beta_ = self._select_beta(P_rf_val, P_graph_val)
            else:
                self.beta_ = 0.5
        else:
            self.beta_ = 1.0

        # 自训练循环
        X_train, y_train = X_l.copy(), y_l.copy()
        w_train = np.ones(len(y_train), dtype=np.float64)
        pool = X_u.copy()
        F_pool = self.F_pool_ if self.use_graph else None
        n_pseudo = 0
        q = self.quantile
        for t in range(self.rounds):
            if len(pool) == 0:
                break
            base = self._new_base()
            if SKLEARN_OK:
                base.fit(X_train, y_train, sample_weight=w_train)
            else:
                base.fit(X_train, y_train, X_train[:0])
            P_rf = self._align(
                np.asarray(base.predict_proba(pool), dtype=np.float64),
                base.classes_ if hasattr(base, "classes_") else None,
            )
            if self.use_graph and F_pool is not None and self.beta_ < 1.0:
                P = self.beta_ * P_rf + (1.0 - self.beta_) * F_pool
            else:
                P = P_rf
            P = P / (P.sum(axis=1, keepdims=True) + 1e-12)

            conf = P.max(axis=1)
            pred = np.argmax(P, axis=1)
            q_t = q if not self.use_anneal else max(0.5, q * (self.anneal**t))
            taus = self._pool_threshold(conf, pred, q_t)

            B = max(6, int(0.10 * len(pool)))
            picks: list = []
            if self.balance:
                budgets = np.floor(B * priors).astype(int)
                budgets[budgets == 0] = 1
                for ci in range(len(self.classes_)):
                    cand = np.where((pred == ci) & (conf >= taus[ci]))[0]
                    if len(cand) == 0:
                        continue
                    cand = cand[np.argsort(-conf[cand])][: budgets[ci]]
                    picks.extend(cand.tolist())
            else:
                thr = float(np.quantile(conf, q_t))
                cand = np.where(conf >= thr)[0]
                cand = cand[np.argsort(-conf[cand])][:B]
                picks.extend(cand.tolist())

            if not picks:
                break
            picks = np.asarray(sorted(set(picks)))
            X_train = np.concatenate([X_train, pool[picks]], axis=0)
            y_train = np.concatenate([y_train, self.classes_[pred[picks]]])
            w_train = np.concatenate([w_train, conf[picks]])
            pool = np.delete(pool, picks, axis=0)
            F_pool = np.delete(F_pool, picks, axis=0) if F_pool is not None else None
            n_pseudo += len(picks)
            q = q_t

        self.final_ = self._new_base()
        if SKLEARN_OK:
            self.final_.fit(X_train, y_train, sample_weight=w_train)
        else:
            self.final_.fit(X_train, y_train, X_train[:0])
        self.n_pseudo_ = n_pseudo
        self.n_labeled_ = len(y_l)
        return self

    def _pool_threshold(self, conf: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
        n_classes = len(self.classes_)
        taus = np.full(n_classes, float(np.quantile(conf, q)))
        for ci in range(n_classes):
            cand = conf[pred == ci]
            if len(cand) >= 3:
                taus[ci] = float(np.quantile(cand, q))
        floor = 1.0 / n_classes + 0.05
        return np.maximum(taus, floor)

    def _align(self, P: np.ndarray, fitted_classes=None) -> np.ndarray:
        """概率列对齐到全局 classes 顺序。"""
        if fitted_classes is None:
            return P
        fc = np.asarray(fitted_classes)
        if np.array_equal(fc, self.classes_):
            return P
        col = {c: i for i, c in enumerate(fc)}
        Pfull = np.zeros((len(P), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            if c in col:
                Pfull[:, i] = P[:, col[c]]
        return Pfull

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P_rf = self._align(
            np.asarray(self.final_.predict_proba(X), dtype=np.float64),
            getattr(self.final_, "classes_", self.classes_),
        )
        P_rf = P_rf / (P_rf.sum(axis=1, keepdims=True) + 1e-12)
        if self.use_graph and self.beta_ < 1.0:
            P_g = self._graph_proba(X)
            P = self.beta_ * P_rf + (1.0 - self.beta_) * P_g
        else:
            P = P_rf
        return P / (P.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
