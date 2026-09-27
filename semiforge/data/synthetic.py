"""合成 SSL 数据集 + 切分（固定 seed 可复现）。作者: 晨星

难度设计原则（踩坑：数据过易则所有方法满分，无区分度）：
- 少量标注（每类 8 个）、标注与测试分布一致；
- 注入结构性难度（moons/circles/spiral 非线性、blobs 重叠、xor 高斯、digits 真实数据子采样）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.datasets import load_digits, make_blobs, make_circles, make_moons

from ..core.errors import DataError
from ..core.types import SSLSplit

DATASETS = ("moons", "circles", "blobs_overlap", "xor_gauss", "spiral", "digits_sub")


def _spiral(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    per = n // 2
    xs, ys = [], []
    for label, sign in ((0, 1.0), (1, -1.0)):
        t = np.linspace(0.35, 2.6 * np.pi, per)
        r = 0.06 * t
        x = r * np.sin(t) * sign + rng.normal(0, 0.055, per)
        y = r * np.cos(t) * sign + rng.normal(0, 0.055, per)
        xs.append(np.stack([x, y], axis=1))
        ys.append(np.full(per, label))
    return np.concatenate(xs), np.concatenate(ys)


def _xor_gauss(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    per = n // 4
    centers = np.array([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    xs, ys = [], []
    for c, lab in zip(centers, labels):
        xs.append(rng.normal(c, 0.85, size=(per, 2)))
        ys.append(np.full(per, lab))
    return np.concatenate(xs), np.concatenate(ys)


def make_dataset(name: str, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成全量 (X, y)。"""
    rng = np.random.default_rng(seed)
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=0.32, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, factor=0.55, noise=0.14, random_state=seed)
    elif name == "blobs_overlap":
        X, y = make_blobs(
            n_samples=n, centers=3, cluster_std=2.1, center_box=(-3.0, 3.0), random_state=seed
        )
    elif name == "xor_gauss":
        X, y = _xor_gauss(n, rng)
    elif name == "spiral":
        X, y = _spiral(n, rng)
    elif name == "digits_sub":
        digits = load_digits()
        rng2 = np.random.default_rng(seed)
        idx = rng2.choice(len(digits.data), size=n, replace=False)
        X = digits.data[idx].astype(np.float64) / 16.0
        y = digits.target[idx]
        keep = np.isin(y, [0, 1, 2, 3, 4, 5])
        X, y = X[keep], y[keep]
        remap = {c: i for i, c in enumerate(sorted(np.unique(y)))}
        y = np.array([remap[v] for v in y])
    else:
        raise DataError(f"未知数据集: {name}（可选: {DATASETS}）")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def make_ssl_split(
    name: str,
    n: int,
    labeled_per_class: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> SSLSplit:
    """切分协议：stratified；labeled ⊥ val ⊥ test ⊥ unlabeled，无泄漏。"""
    X, y = make_dataset(name, n, seed)
    classes = np.unique(y)
    rng = np.random.default_rng(seed + 777)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        if len(idx_by_class[c]) < labeled_per_class + 3:
            raise DataError(f"类 {c} 样本不足")

    lab_idx, val_idx, test_idx = [], [], []
    for c in classes:
        perm = rng.permutation(idx_by_class[c])
        lab_idx.append(perm[:labeled_per_class])
        val_idx.append(perm[labeled_per_class : labeled_per_class + n_val // len(classes)])
        test_idx.append(
            perm[labeled_per_class + n_val // len(classes) :][: n_test // len(classes)]
        )
    lab_idx = np.concatenate(lab_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    used = np.concatenate([lab_idx, val_idx, test_idx])
    unl_idx = np.setdiff1d(np.arange(len(y)), used, assume_unique=False)

    return SSLSplit(
        X_l=X[lab_idx],
        y_l=y[lab_idx],
        X_u=X[unl_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        n_classes=len(classes),
        dataset=name,
        seed=seed,
    )


def dataset_dict(name: str) -> Dict[str, str]:
    desc = {
        "moons": "双月牙非线性 (sklearn make_moons, noise=0.32)",
        "circles": "同心圆 (factor=0.55, noise=0.14)",
        "blobs_overlap": "三簇重叠高斯 (cluster_std=2.1)",
        "xor_gauss": "四高斯 XOR 结构（类内不连通）",
        "spiral": "双螺旋 (噪声 0.055)",
        "digits_sub": "sklearn digits 真实数据 6 类子采样",
    }
    return {name: desc.get(name, "")}
