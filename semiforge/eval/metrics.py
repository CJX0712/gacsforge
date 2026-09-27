"""评测指标（手写实现，避免 sklearn 同名递归坑）。作者: 晨星

语义统一：accuracy / macro-F1 均为越大越好。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("长度不一致")
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1s))


def aggregate_mean_std(values: List[float]) -> tuple:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def significant(m1: float, s1: float, m2: float, s2: float) -> bool:
    """V4 简易显著性门槛：均值差 > 0.5*(std1+std2)。"""
    return (m1 - m2) > 0.5 * (s1 + s2)


def group_by(rows: List, key_fn) -> Dict[str, List]:
    out: Dict[str, List] = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out
