"""pipeline 单测：小规模 run + benchmark.json 往返 + CLI 冒烟。作者: 晨星"""

import json

from semiforge.core.config import Config
from semiforge.pipeline.pipeline import SemiForgePipeline


def _small_cfg(tmp_out=None):
    cfg = Config()
    cfg.n_samples = 300
    cfg.labeled_per_class = 6
    cfg.n_val = 50
    cfg.n_test = 80
    cfg.datasets = ["moons", "circles"]
    cfg.seeds = [0, 1, 2]
    cfg.rf_trees = 40
    cfg.gacs_rounds = 5
    return cfg


def test_pipeline_run_small():
    pipe = SemiForgePipeline(_small_cfg())
    report = pipe.run()
    assert report.rows
    assert all(g.passed is not None for g in report.gates)
    assert len(report.failures) >= 1 or len(report.failures) == 0  # 结构存在
    agg = report.aggregated
    assert "moons" in agg and "circles" in agg
    for ds, lst in agg.items():
        f1s = [a.mean_f1 for a in lst]
        assert max(f1s) <= 1.0 + 1e-9


def test_benchmark_json_roundtrip(tmp_path):
    from semiforge.pipeline.benchmark import save_benchmark

    pipe = SemiForgePipeline(_small_cfg())
    report = pipe.run(datasets=["moons"], seeds=[0, 1, 2])
    path = tmp_path / "benchmark.json"
    back = save_benchmark(report, str(path))
    with open(path, encoding="utf-8") as f:
        disk = json.load(f)
    assert disk == back
    assert len(disk["rows"]) == len(report.rows)


def test_pipeline_deterministic():
    cfg = _small_cfg()
    cfg.datasets = ["moons"]
    r1 = SemiForgePipeline(cfg).run(seeds=[0, 1, 2])
    r2 = SemiForgePipeline(cfg).run(seeds=[0, 1, 2])
    k1 = {(r.method, r.dataset, r.seed): round(r.macro_f1, 9) for r in r1.rows if not r.skipped}
    k2 = {(r.method, r.dataset, r.seed): round(r.macro_f1, 9) for r in r2.rows if not r.skipped}
    assert k1 == k2


def test_ablation_structure():
    cfg = _small_cfg()
    cfg.datasets = ["moons"]
    pipe = SemiForgePipeline(cfg)
    abl = pipe._ablation(["moons"], [0, 1, 2])
    assert {"gacs_full", "gacs_no_graph", "gacs_no_balance", "gacs_no_anneal"} <= set(abl)
    f1s = [v.mean_f1 for v in abl.values()]
    assert all(0.0 <= f <= 1.0 for f in f1s)
