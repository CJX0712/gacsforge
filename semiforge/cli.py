"""CLI 入口: python -m semiforge.cli demo --out outputs/benchmark.json。作者: 晨星"""

from __future__ import annotations

import argparse
import sys

from .core.config import load_config
from .pipeline.benchmark import print_table, save_benchmark
from .pipeline.pipeline import SemiForgePipeline


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="semiforge", description="SemiForge 半监督学习系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="端到端演示并落盘 benchmark.json")
    p_demo.add_argument("--out", default="outputs/benchmark.json")
    p_demo.add_argument("--hpo", action="store_true", help="先跑 Optuna HPO 再 benchmark")
    p_demo.add_argument("--datasets", default=None, help="逗号分隔，覆盖默认数据集")
    p_demo.add_argument("--seeds", default=None, help="逗号分隔，覆盖默认 seeds")

    sub.add_parser("tune", help="仅运行 Optuna HPO 并打印最优参数")

    args = parser.parse_args(argv)
    cfg = load_config()
    if args.cmd == "tune":
        from .hpo.tune import tune_flagship

        res = tune_flagship(cfg)
        print(f"best_value(val F1)={res.best_value:.4f} trials={res.n_trials}")
        print(res.best_params)
        return 0

    datasets = args.datasets.split(",") if args.datasets else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    flagship_params = None
    if args.hpo:
        from .hpo.tune import tune_flagship

        res = tune_flagship(cfg)
        print(f"[HPO] best={res.best_value:.4f} params={res.best_params}")
        flagship_params = res.best_params
    pipe = SemiForgePipeline(cfg, flagship_params=flagship_params)
    report = pipe.run(datasets=datasets, seeds=seeds)
    save_benchmark(report, args.out)
    print_table(report)
    print(f"\nbenchmark.json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
