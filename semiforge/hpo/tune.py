"""Optuna TPE 超参搜索（val macro-F1 全数据集均值目标，防简单数据退化）。作者: 晨星"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.types import TuneResult
from ..data.synthetic import make_ssl_split
from ..eval.metrics import macro_f1
from ..ssl.flagship import GACS

SEARCH = {
    "rounds": (6, 18),
    "alpha": (0.2, 0.8),
    "quantile": (0.70, 0.95),
    "anneal": (0.82, 0.98),
    "graph_k": (5, 20),
}


def tune_flagship(cfg: Config, seed: int = 9042, hpo_seed: int = 9043) -> TuneResult:
    """目标：所有数据集 val macro-F1 均值（HPO 数据 seed 与 benchmark 不相交）。"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = [
        make_ssl_split(name, cfg.n_samples, cfg.labeled_per_class, cfg.n_val, cfg.n_test, seed)
        for name in cfg.datasets
    ]

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            rounds=trial.suggest_int("rounds", *SEARCH["rounds"]),
            alpha=trial.suggest_float("alpha", *SEARCH["alpha"]),
            quantile=trial.suggest_float("quantile", *SEARCH["quantile"]),
            anneal=trial.suggest_float("anneal", *SEARCH["anneal"]),
            graph_k=trial.suggest_int("graph_k", *SEARCH["graph_k"]),
        )
        f1s = []
        for sp in splits:
            m = GACS(
                rf_trees=cfg.rf_trees,
                seed=seed,
                balance=cfg.gacs_balance,
                use_graph=cfg.gacs_use_graph,
                use_anneal=cfg.gacs_use_anneal,
                **params,
            )
            m.fit(sp.X_l, sp.y_l, sp.X_u)
            pred = m.predict(sp.X_val)
            f1s.append(macro_f1(sp.y_val, pred))
        return float(np.mean(f1s))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=hpo_seed)
    )
    study.optimize(objective, n_trials=cfg.hpo_trials, show_progress_bar=False)
    return TuneResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        used=True,
    )
