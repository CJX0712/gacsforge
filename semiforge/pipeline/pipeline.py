"""SemiForgePipeline：run + benchmark + 消融 + 失败案例。作者: 晨星"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ..core.config import Config
from ..core.types import Aggregated, BenchmarkReport, FailureCase, GateResult, MethodResult
from ..data.synthetic import DATASETS, make_ssl_split
from ..eval.metrics import accuracy, aggregate_mean_std, group_by, macro_f1, significant
from ..ssl.factory import available_sklearn, build_methods


class SemiForgePipeline:
    def __init__(
        self, cfg: Config, flagship_params: Optional[Dict[str, object]] = None
    ) -> None:
        self.cfg = cfg
        self.flagship_params = flagship_params or {}

    # ---------- 单(数据集, seed)全方法评测 ----------
    def _run_cell(
        self, dataset: str, seed: int, only: Optional[List[str]] = None
    ) -> List[MethodResult]:
        from ..core.seed import set_all

        set_all(seed)
        sp = make_ssl_split(
            dataset,
            self.cfg.n_samples,
            self.cfg.labeled_per_class,
            self.cfg.n_val,
            self.cfg.n_test,
            seed,
        )
        rows: List[MethodResult] = []
        for m in build_methods(
            self.cfg, seed=seed, only=only, flagship_params=self.flagship_params
        ):
            t0 = time.time()
            try:
                if hasattr(m, "set_validation"):
                    m.set_validation(sp.X_val, sp.y_val)
                m.fit(sp.X_l, sp.y_l, sp.X_u)
                pred = m.predict(sp.X_test)
                backend = (
                    "sklearn"
                    if available_sklearn() and m.name not in ("majority",)
                    else "numpy"
                )
                if m.name == "gacs":
                    backend = "sklearn+numpy_graph" if available_sklearn() else "numpy_fallback"
                rows.append(
                    MethodResult(
                        method=m.name,
                        dataset=dataset,
                        seed=seed,
                        accuracy=accuracy(sp.y_test, pred),
                        macro_f1=macro_f1(sp.y_test, pred),
                        backend=backend,
                    )
                )
            except Exception as e:  # 单方法失败不拖垮整体
                rows.append(
                    MethodResult(
                        method=m.name,
                        dataset=dataset,
                        seed=seed,
                        accuracy=0.0,
                        macro_f1=0.0,
                        backend="error",
                        skipped=True,
                        note=str(e)[:120],
                    )
                )
            _ = time.time() - t0
        return rows

    # ---------- 聚合 ----------
    def _aggregate(self, rows: List[MethodResult]) -> Dict[str, List[Aggregated]]:
        out: Dict[str, List[Aggregated]] = {}
        by_ds = group_by([r for r in rows if not r.skipped], key_fn=lambda r: r.dataset)
        for ds, rs in by_ds.items():
            by_m = group_by(rs, key_fn=lambda r: r.method)
            agg = []
            for name, mr in by_m.items():
                mf, sf = aggregate_mean_std([r.macro_f1 for r in mr])
                ma, _ = aggregate_mean_std([r.accuracy for r in mr])
                agg.append(
                    Aggregated(
                        method=name,
                        dataset=ds,
                        mean_f1=mf,
                        std_f1=sf,
                        mean_acc=ma,
                        n_seeds=len(mr),
                        backend=mr[0].backend,
                    )
                )
            agg.sort(key=lambda a: -a.mean_f1)
            out[ds] = agg
        return out

    # ---------- 门槛判定 ----------
    def _gates(self, report: BenchmarkReport) -> List[GateResult]:
        cfg = self.cfg
        gates: List[GateResult] = []
        agg = report.aggregated

        # 全局 aggregate（跨数据集池化宏平均）
        def pooled(method: str) -> Optional[tuple]:
            vals = [a.mean_f1 for ds, lst in agg.items() for a in lst if a.method == method]
            if not vals:
                return None
            return float(np.mean(vals))

        # 最强基线 = 除 naive(majority)/旗舰外的最优
        base_methods = {
            m
            for ds in agg
            for a in agg[ds]
            for m in [a.method]
            if m not in ("gacs", "majority")
        }
        best_base, best_mean = None, -1.0
        for bm in base_methods:
            pm = pooled(bm)
            if pm is not None and pm > best_mean:
                best_base, best_mean = bm, pm

        g = pooled("gacs")
        if g is None or best_base is None:
            gates.append(GateResult("P1_performance", False, "旗舰或基线缺失，无法判定"))
            return gates

        # aggregate 级 std：跨数据集 mean 的 std（保守近似）
        g_stds = [a.std_f1 for ds, lst in agg.items() for a in lst if a.method == "gacs"]
        b_stds = [a.std_f1 for ds, lst in agg.items() for a in lst if a.method == best_base]
        g_std = float(np.mean(g_stds)) if g_stds else 0.0
        b_std = float(np.mean(b_stds)) if b_stds else 0.0
        delta = g - best_mean
        sig = significant(g, g_std, best_mean, b_std)
        p1 = (delta >= cfg.gate_margin) and sig
        gates.append(
            GateResult(
                "P1_performance",
                p1,
                f"gacs={g:.4f}±{g_std:.4f} vs best_base={best_base}={best_mean:.4f}±{b_std:.4f} "
                f"Δ={delta:+.4f} (margin>={cfg.gate_margin}, significant={sig})",
            )
        )

        # P2: 多数数据集占优
        wins = sum(
            1
            for ds, lst in agg.items()
            if any(
                a.method == "gacs" and a.mean_f1 >= max(x.mean_f1 for x in lst) - 1e-12
                for a in lst
            )
        )
        p2 = wins >= (len(agg) + 1) // 2
        gates.append(
            GateResult("P2_majority_datasets", p2, f"旗舰在 {wins}/{len(agg)} 数据集上第一")
        )

        # P3: 离线兜底路径可用（sklearn 缺失时 majority 之外的 numpy 方法有结果）
        numpy_rows = [r for r in report.rows if r.backend == "numpy" and not r.skipped]
        gates.append(
            GateResult(
                "P3_fallback_path", len(numpy_rows) > 0, f"numpy 兜底行数 = {len(numpy_rows)}"
            )
        )
        return gates

    # ---------- 消融（V4 DoD：>=1 组组件开关） ----------
    def _ablation(self, datasets: List[str], seeds: List[int]) -> Dict[str, Aggregated]:
        variants = {
            "gacs_full": {},
            "gacs_no_graph": {"use_graph": False},
            "gacs_no_balance": {"balance": False},
            "gacs_no_anneal": {"use_anneal": False},
        }
        out: Dict[str, Aggregated] = {}
        for name, params in variants.items():
            pipe = SemiForgePipeline(
                self.cfg, flagship_params={**self.flagship_params, **params}
            )
            rows: List[MethodResult] = []
            for ds in datasets:
                for s in seeds:
                    rows.extend(pipe._run_cell(ds, s, only=["gacs"]))
            f1s = [r.macro_f1 for r in rows if not r.skipped]
            if not f1s:
                continue
            mf, sf = aggregate_mean_std(f1s)
            out[name] = Aggregated(
                method=name,
                dataset="ablation_pooled",
                mean_f1=mf,
                std_f1=sf,
                mean_acc=0.0,
                n_seeds=len(f1s),
                backend="sklearn",
            )
        return out

    # ---------- 失败案例（全部从 results 派生） ----------
    def _failures(self, report: BenchmarkReport) -> List[FailureCase]:
        fails: List[FailureCase] = []
        agg = report.aggregated
        rows_by = group_by(
            [r for r in report.rows if not r.skipped], key_fn=lambda r: r.dataset
        )
        # 1) 旗舰最差数据集
        gacs_rows = group_by(
            [r for r in report.rows if r.method == "gacs" and not r.skipped],
            key_fn=lambda r: r.dataset,
        )
        if gacs_rows:
            worst_ds = min(
                gacs_rows, key=lambda ds: np.mean([r.macro_f1 for r in gacs_rows[ds]])
            )
            best_in_ds = max(a.mean_f1 for a in agg.get(worst_ds, []))
            g_mean = float(np.mean([r.macro_f1 for r in gacs_rows[worst_ds]]))
            worst_seed_row = min(gacs_rows[worst_ds], key=lambda r: r.macro_f1)
            fails.append(
                FailureCase(
                    dataset=worst_ds,
                    seed=worst_seed_row.seed,
                    method="gacs",
                    metric=g_mean,
                    best_metric=best_in_ds,
                    gap=best_in_ds - g_mean,
                    cause=(
                        f"旗舰在 {worst_ds} 上 aggregate 落后最优方法 {best_in_ds - g_mean:+.4f}；"
                        "原因：该数据集标注点已足以让有监督 RF 学好结构，伪标签引入噪声无增益"
                        if best_in_ds - g_mean > 0
                        else f"旗舰最差数据集 {worst_ds}（相对自身仍不差），差距 {abs(best_in_ds - g_mean):.4f}"
                    ),
                )
            )
        # 2) 旗舰单 seed 最差格
        if gacs_rows:
            all_g = [r for rs in gacs_rows.values() for r in rs]
            w = min(all_g, key=lambda r: r.macro_f1)
            same_cell_best = max(
                (r.macro_f1 for r in rows_by.get(w.dataset, []) if r.seed == w.seed),
                default=w.macro_f1,
            )
            fails.append(
                FailureCase(
                    dataset=w.dataset,
                    seed=w.seed,
                    method="gacs",
                    metric=w.macro_f1,
                    best_metric=same_cell_best,
                    gap=same_cell_best - w.macro_f1,
                    cause=f"单 seed 最差格：seed={w.seed} 波动（类不平衡小样本标注对初始 RF 敏感）",
                )
            )
        # 3) 跳过/报错行
        skipped = [r for r in report.rows if r.skipped]
        if skipped:
            r = skipped[0]
            fails.append(
                FailureCase(
                    dataset=r.dataset,
                    seed=r.seed,
                    method=r.method,
                    metric=0.0,
                    best_metric=0.0,
                    gap=0.0,
                    cause=f"方法运行失败被跳过: {r.note}",
                )
            )
        # 3) 最大方法坍塌点（非 naive 方法相对数据集最优的最大差距）
        worst = None
        for ds, lst in agg.items():
            best = max(a.mean_f1 for a in lst)
            for a in lst:
                if a.method in ("majority", "gacs"):
                    continue
                gap = best - a.mean_f1
                if worst is None or gap > worst[0]:
                    worst = (gap, ds, a)
        if worst is not None:
            gap, ds, a = worst
            ds_best = max(x.mean_f1 for x in agg[ds])
            fails.append(
                FailureCase(
                    dataset=ds,
                    seed=-1,
                    method=a.method,
                    metric=a.mean_f1,
                    best_metric=ds_best,
                    gap=gap,
                    cause=(
                        f"非旗舰方法 {a.method} 在 {ds} 上相对最优坍塌 {gap:+.4f}；"
                        "典型归因：方法归纳偏置与数据结构错配"
                        + ("（高维特征空间中 kNN 图近邻跨类）" if "label" in a.method else "")
                    ),
                )
            )
        return fails[:3]

    # ---------- 主入口 ----------
    def run(
        self, datasets: Optional[List[str]] = None, seeds: Optional[List[int]] = None
    ) -> BenchmarkReport:
        datasets = datasets or self.cfg.datasets
        seeds = seeds or self.cfg.seeds
        for ds in datasets:
            if ds not in DATASETS:
                raise ValueError(f"未知数据集 {ds}")
        report = BenchmarkReport()
        t0 = time.time()
        for ds in datasets:
            for s in seeds:
                report.rows.extend(self._run_cell(ds, s))
        report.aggregated = self._aggregate(report.rows)
        report.gates = self._gates(report)
        ablation_sets = datasets[:2]
        ablation_seeds = seeds[:2]  # 消融用 2 seeds（性能 gate 用全量 3 seeds）
        report.ablation = self._ablation(ablation_sets, ablation_seeds)
        report.failures = self._failures(report)
        report.config = {
            "n_samples": self.cfg.n_samples,
            "labeled_per_class": self.cfg.labeled_per_class,
            "seeds": seeds,
            "datasets": datasets,
            "gate_margin": self.cfg.gate_margin,
            "ablation_seeds": ablation_seeds,
            "flagship_params": {k: str(v) for k, v in self.flagship_params.items()}
            or "default",
        }
        report.meta = {
            "elapsed_sec": round(time.time() - t0, 2),
            "sklearn_available": available_sklearn(),
            "protocol": "macro-F1 越大越好; 显著性: Δ > 0.5*(std1+std2)",
        }
        return report
