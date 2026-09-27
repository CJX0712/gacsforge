"""类型定义（dataclass）。作者: 晨星"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SSLSplit:
    """一次半监督切分：少量标注 + 大量无标注 + 独立 val/test。"""

    X_l: np.ndarray
    y_l: np.ndarray
    X_u: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    dataset: str
    seed: int


@dataclass
class MethodResult:
    """单方法在单(数据集, seed)上的结果。"""

    method: str
    dataset: str
    seed: int
    accuracy: float
    macro_f1: float
    backend: str  # sklearn / numpy / lightgbm ...
    skipped: bool = False
    note: str = ""


@dataclass
class Aggregated:
    """跨 seed 聚合（mean±std）。"""

    method: str
    dataset: str
    mean_f1: float
    std_f1: float
    mean_acc: float
    n_seeds: int
    backend: str


@dataclass
class GateResult:
    """显著性判定（V4 统计严谨：均值差 > 0.5*(std1+std2)）。"""

    name: str
    passed: bool
    detail: str


@dataclass
class FailureCase:
    """失败案例（全部从 results 派生，禁止预设结论）。"""

    dataset: str
    seed: int
    method: str
    metric: float
    best_metric: float
    gap: float
    cause: str


@dataclass
class BenchmarkReport:
    rows: List[MethodResult] = field(default_factory=list)
    aggregated: Dict[str, List[Aggregated]] = field(default_factory=dict)  # per-dataset
    gates: List[GateResult] = field(default_factory=list)
    ablation: Dict[str, "Aggregated"] = field(default_factory=dict)
    failures: List[FailureCase] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "meta": self.meta,
            "rows": [
                {
                    "method": r.method,
                    "dataset": r.dataset,
                    "seed": r.seed,
                    "accuracy": round(r.accuracy, 6),
                    "macro_f1": round(r.macro_f1, 6),
                    "backend": r.backend,
                    "skipped": r.skipped,
                    "note": r.note,
                }
                for r in self.rows
            ],
            "aggregated": {
                ds: [
                    {
                        "method": a.method,
                        "mean_macro_f1": round(a.mean_f1, 6),
                        "std_macro_f1": round(a.std_f1, 6),
                        "mean_accuracy": round(a.mean_acc, 6),
                        "n_seeds": a.n_seeds,
                        "backend": a.backend,
                    }
                    for a in agg
                ]
                for ds, agg in self.aggregated.items()
            },
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail} for g in self.gates
            ],
            "ablation": {
                k: {
                    "mean_macro_f1": round(v.mean_f1, 6),
                    "std_macro_f1": round(v.std_f1, 6),
                }
                for k, v in self.ablation.items()
            },
            "failures": [
                {
                    "dataset": f.dataset,
                    "seed": f.seed,
                    "method": f.method,
                    "metric": round(f.metric, 6),
                    "best_metric": round(f.best_metric, 6),
                    "gap": round(f.gap, 6),
                    "cause": f.cause,
                }
                for f in self.failures
            ],
        }


@dataclass
class TuneResult:
    best_params: Dict[str, object]
    best_value: float
    n_trials: int
    used: bool
