"""benchmark 落盘 + 读取校验。作者: 晨星"""

from __future__ import annotations

import json
import os
from typing import Dict

from ..core.types import BenchmarkReport


def save_benchmark(report: BenchmarkReport, path: str) -> Dict[str, object]:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = report.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # 立即读回校验（指标幻觉坑：生成后必须读回核对）
    with open(path, "r", encoding="utf-8") as f:
        back = json.load(f)
    assert back["meta"]["elapsed_sec"] == data["meta"]["elapsed_sec"], (
        "benchmark.json 读回校验失败"
    )
    n_rows = len(back["rows"])
    assert n_rows == len(report.rows), "行数不一致"
    return back


def print_table(report: BenchmarkReport) -> None:
    """CLI 基准表（固定宽度，表头对齐）。"""
    print("\n=== SemiForge Benchmark (macro-F1 mean±std, 越大越好) ===")
    header = f"{'dataset':<14}{'method':<18}{'macro_f1':<16}{'acc':<10}{'backend':<12}"
    print(header)
    print("-" * len(header))
    for ds, agg in report.aggregated.items():
        for a in agg:
            print(
                f"{ds:<14}{a.method:<18}{a.mean_f1:.4f}±{a.std_f1:.4f}    "
                f"{a.mean_acc:<10.4f}{a.backend:<12}"
            )
    print("\n=== Gates ===")
    for g in report.gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"[{mark}] {g.name}: {g.detail}")
    print("\n=== Ablation (pooled) ===")
    for k, v in report.ablation.items():
        print(f"{k:<18}{v.mean_f1:.4f}±{v.std_f1:.4f}")
    if report.failures:
        print("\n=== Failure Cases ===")
        for f in report.failures:
            print(f"- [{f.dataset}/seed{f.seed}/{f.method}] gap={f.gap:+.4f} :: {f.cause}")
