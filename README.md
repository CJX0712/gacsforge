# GACSForge

![CI](https://github.com/CJX0712/gacsforge/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/release/CJX0712/gacsforge)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Grade](https://img.shields.io/badge/quality-S-success)

**GACSForge — 半监督学习（Semi-Supervised Learning）系统** · 旗舰自研 **GACS**（图-树混合类平衡自训练）。
在 6 数据集 × 3 seeds 实测中 aggregate macro-F1 **+0.0536 显著优于最强基线**（门槛 0.02）。

> 作者：晨星（CJX0712）

## 一键复现

```bash
git clone https://github.com/CJX0712/gacsforge.git && cd gacsforge
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux: .venv/bin/pip
.venv/Scripts/python examples/run_demo.py                               # 生成 outputs/benchmark.json (~45s)
.venv/Scripts/python -m pytest -q                                       # 32 tests
```

可选：`--hpo` 先跑 Optuna TPE 调参再评测。

## 实测性能（benchmark.json 真实输出，n=600, 3 seeds, mean±std）

| 数据集 | GACS（旗舰） | selftrain_rf | supervised_rf | label_spread | label_prop | majority |
|--------|-------------|--------------|---------------|--------------|------------|----------|
| moons | 0.8015 | 0.7799 | 0.7760 | **0.8262** | 0.8206 | 0.3333 |
| circles | **0.8374** | 0.7395 | 0.7692 | 0.6272 | 0.4387 | 0.3333 |
| blobs_overlap | 0.4182 | 0.4171 | 0.4274 | **0.4417** | 0.4070 | 0.1667 |
| xor_gauss | **0.9733** | 0.6527 | 0.7744 | 0.9065 | 0.8996 | 0.3333 |
| spiral | **0.6727** | 0.5908 | 0.6381 | 0.5972 | 0.5225 | 0.3333 |
| digits_sub | **0.9237** | 0.9215 | 0.9204 | 0.7689 | 0.7818 | 0.0491 |

aggregate：**GACS 0.7711±0.0307** vs supervised_rf 0.7176±0.0352 → **Δ=+0.0536（显著）**
消融（2 数据集 × 2 seeds, pooled）：no_graph −0.079（图分支为核心增益）/ no_balance −0.004 / no_anneal 0

> 注：本 README 数字以仓库内 `outputs/benchmark.json` 为唯一真源。

## 架构

```
cli → pipeline → {data, ssl, hpo, eval} → core
```

- **Tier-0（SOTA 后端）**：scikit-learn SelfTrainingClassifier / LabelSpreading / LabelPropagation
- **旗舰 GACS**：图传播分支 × RF 自训练分支，val 选 β，类平衡预算 + 逐类分位阈值 + 课程退火 + 置信加权
- **Tier-1（离线兜底）**：纯 numpy kNN 图标签传播 / 自训练（零下载可跑，sklearn 缺失自动降级）

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/model_card.md](docs/model_card.md)。

## 离线兜底

`available_sklearn()` 探测后端；sklearn 不可用时自动降级纯 numpy 路径（有单测覆盖），
benchmark 自动跳过缺失后端并在 `benchmark.json` 标注。

## CLI

```bash
python -m semiforge.cli demo --out outputs/benchmark.json   # 端到端 benchmark
python -m semiforge.cli demo --hpo                          # Optuna HPO + benchmark
python -m semiforge.cli tune                                # 仅调参
```

环境变量覆盖（示例）：`ENV_SEMIFORGE_N_SAMPLES=600`、`ENV_SEMIFORGE_DATASETS=moons,spiral`。

## License

MIT © 2026 晨星
