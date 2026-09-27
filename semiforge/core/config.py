"""配置：支持 ENV_SEMIFORGE_* 覆盖 + schema 校验。作者: 晨星"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Dict, List

from .errors import ConfigError


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ENV_SEMIFORGE_{name}", default)


@dataclass
class Config:
    # 数据
    n_samples: int = 600
    labeled_per_class: int = 8
    n_val: int = 100
    n_test: int = 200
    datasets: List[str] = field(
        default_factory=lambda: [
            "moons",
            "circles",
            "blobs_overlap",
            "xor_gauss",
            "spiral",
            "digits_sub",
        ]
    )
    # 方法
    rf_trees: int = 80
    knn_k: int = 10
    # 旗舰 GACS 超参（HPO 可覆盖）
    gacs_rounds: int = 8
    gacs_alpha: float = 0.75  # 图传播强度（LP 迭代）
    gacs_quantile: float = 0.85  # 类平衡阈值分位数
    gacs_anneal: float = 0.9  # 分位数退火系数（每轮）
    gacs_graph_k: int = 10
    gacs_balance: bool = True
    gacs_use_graph: bool = True
    gacs_use_anneal: bool = True
    # 评测
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    gate_margin: float = 0.02  # 门槛：aggregate ΔF1 ≥ margin 且显著
    hpo_trials: int = 40
    # 输出
    out_dir: str = "outputs"

    def validate(self) -> None:
        if self.n_samples < 100:
            raise ConfigError("n_samples 必须 >= 100")
        if self.labeled_per_class < 2:
            raise ConfigError("labeled_per_class 必须 >= 2")
        if not (0.0 <= self.gacs_alpha <= 1.0):
            raise ConfigError("gacs_alpha 必须在 [0,1]")
        if not (0.5 <= self.gacs_quantile < 1.0):
            raise ConfigError("gacs_quantile 必须在 [0.5,1)")
        if not self.seeds:
            raise ConfigError("seeds 不能为空")
        if len(self.seeds) < 3:
            raise ConfigError("V4 统计严谨：seeds 必须 >= 3")


_INT_FIELDS = {
    "n_samples",
    "labeled_per_class",
    "n_val",
    "n_test",
    "rf_trees",
    "knn_k",
    "gacs_rounds",
    "gacs_graph_k",
    "hpo_trials",
}
_FLOAT_FIELDS = {"gacs_alpha", "gacs_quantile", "gacs_anneal", "gate_margin"}


def load_config() -> Config:
    """从环境变量读覆盖项（ENV_SEMIFORGE_<FIELD>），schema 校验。"""
    cfg = Config()
    for f in fields(Config):
        env = os.environ.get(f"ENV_SEMIFORGE_{f.name.upper()}")
        if env is None:
            continue
        if f.name in _INT_FIELDS:
            setattr(cfg, f.name, int(env))
        elif f.name in _FLOAT_FIELDS:
            setattr(cfg, f.name, float(env))
        elif f.name == "datasets":
            setattr(cfg, f.name, [s.strip() for s in env.split(",") if s.strip()])
        elif f.name == "seeds":
            setattr(cfg, f.name, [int(s) for s in env.split(",") if s.strip()])
        elif f.name in ("gacs_balance", "gacs_use_graph", "gacs_use_anneal"):
            setattr(cfg, f.name, env.lower() in ("1", "true", "yes"))
        elif f.name == "out_dir":
            setattr(cfg, f.name, env)
    cfg.validate()
    return cfg


def config_overrides(cfg: Config) -> Dict[str, object]:
    """导出非默认项（写入报告 meta，保证可溯源）。"""
    d: Dict[str, object] = {}
    default = Config()
    for f in fields(Config):
        cur, base = getattr(cfg, f.name), getattr(default, f.name)
        if cur != base:
            d[f.name] = cur
    return d
