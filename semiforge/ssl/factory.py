"""方法工厂：后端探测 + 自动降级。作者: 晨星"""

from __future__ import annotations

from typing import Dict, List

from ..core.config import Config
from .baselines import MajorityClassifier, SupervisedRF
from .classical import SKLEARN_OK, LabelPropSK, LabelSpreadSK, SelfTrainingRF
from .flagship import GACS
from .numpy_fallback import NumpyLabelPropagate, NumpySelfTrain


def available_sklearn() -> bool:
    return SKLEARN_OK


def build_methods(
    cfg: Config,
    seed: int,
    only: List[str] | None = None,
    flagship_params: Dict[str, object] | None = None,
) -> List[object]:
    """按可用性构建方法列表（顺序 = 基线 → 经典 → 旗舰）。

    sklearn 缺失时：跳过 Tier-0，旗舰自动 numpy 兜底，Benchmark 标注降级。
    """
    flag_params = dict(
        rf_trees=cfg.rf_trees,
        seed=seed,
        rounds=cfg.gacs_rounds,
        alpha=cfg.gacs_alpha,
        quantile=cfg.gacs_quantile,
        anneal=cfg.gacs_anneal,
        graph_k=cfg.gacs_graph_k,
        balance=cfg.gacs_balance,
        use_graph=cfg.gacs_use_graph,
        use_anneal=cfg.gacs_use_anneal,
    )
    if flagship_params:
        flag_params.update(flagship_params)

    methods: List[object] = [MajorityClassifier()]
    if available_sklearn():
        methods.append(SupervisedRF(rf_trees=cfg.rf_trees, seed=seed))
        methods.append(SelfTrainingRF(rf_trees=cfg.rf_trees, seed=seed, max_iter=10))
        methods.append(LabelSpreadSK(k=cfg.knn_k, seed=seed))
        methods.append(LabelPropSK(k=cfg.knn_k, seed=seed))
    else:
        methods.append(NumpySelfTrain(k=cfg.knn_k))
        methods.append(NumpyLabelPropagate(k=cfg.knn_k))
    methods.append(GACS(**flag_params))

    if only:
        want = set(only)
        methods = [m for m in methods if m.name in want]
    return methods
