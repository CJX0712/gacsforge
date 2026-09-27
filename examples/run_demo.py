"""端到端演示：跑完整 benchmark 并落盘 benchmark.json。作者: 晨星

用法:
    python examples/run_demo.py
    python examples/run_demo.py --hpo
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semiforge.core.config import load_config
from semiforge.core.seed import set_all
from semiforge.hpo.tune import tune_flagship
from semiforge.pipeline.benchmark import print_table, save_benchmark
from semiforge.pipeline.pipeline import SemiForgePipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--out", default="outputs/benchmark.json")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    set_all(2026)
    cfg = load_config()
    flagship_params = None
    if args.hpo:
        res = tune_flagship(cfg)
        print(f"[HPO] best val F1={res.best_value:.4f} params={res.best_params}")
        flagship_params = res.best_params
    pipe = SemiForgePipeline(cfg, flagship_params=flagship_params)
    report = pipe.run()
    save_benchmark(report, args.out)
    print_table(report)
    gates_ok = all(g.passed for g in report.gates)
    print(
        f"\nAll gates: {'PASS' if gates_ok else 'FAIL'} | elapsed={report.meta['elapsed_sec']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
