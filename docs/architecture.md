# SemiForge 架构

> 半监督学习（Semi-Supervised Learning, SSL）系统 · 作者：晨星

## 1. 系统定位

在「少量标注 + 大量无标注」场景下，统一评测与自研旗舰 GACS（图-树混合自训练），
对标 SSL 经典 SOTA：Graph-based 方法（Label Propagation, Zhu et al. 2002；Label Spreading, Zhou et al. 2004）
与 Self-Training（Yarowsky 1995; Amini et al. 2022 综述）。

## 2. 模块划分（单向无环调用）

```
cli → pipeline → {data, hpo, ssl, eval} → core
```

| 模块 | 职责 | 关键接口 |
|------|------|----------|
| `core/` | 类型(dataclass)、错误(E100~E500)、配置(ENV_SEMIFORGE_* 覆盖+校验)、seed | `set_all(seed)`, `load_config()` |
| `data/` | 6 个合成/半真实 SSL 数据集 + 无泄漏切分 | `make_ssl_split(name, n, lpc, n_val, n_test, seed)` |
| `ssl/` | 方法库：基线/经典/兜底/旗舰 | `build_methods(cfg, seed)` |
| `eval/` | 手写 accuracy / macro-F1 + 显著性判定 | `macro_f1`, `significant` |
| `hpo/` | Optuna TPE 调参（val 全数据集均值目标） | `tune_flagship(cfg)` |
| `pipeline/` | 编排、聚合、gate、消融、失败案例 | `SemiForgePipeline.run()` |

## 3. 方法清单与语义

统一协议：`fit(X_l, y_l, X_u)` + `predict_proba(X)`；分数语义 = macro-F1 越大越好。

| 方法 | 层级 | 说明 |
|------|------|------|
| `majority` | naive | 多数类基线 |
| `supervised_rf` | 强基线 | 仅标注集 RandomForest |
| `selftrain_rf` | 强基线 | sklearn SelfTrainingClassifier(RF, thr=0.75) |
| `label_spread` / `label_prop` | Tier-0 | sklearn 图方法（kNN 核 + 归纳近邻近似） |
| `numpy_selftrain` / `numpy_labelprop` | Tier-1 兜底 | 纯 numpy 零下载实现 |
| **`gacs`** | 旗舰 | 见下节 |

## 4. 旗舰 GACS（Graph-Arborescence Class-balanced Self-training）

四个创新组件（与经典 SelfTrainingClassifier 的差异）：

1. **图-树双分支**：图分支 `F = (1-α)Y0 + αSF`（标注 one-hot 全图传播，kNN 对称加权图，
   谱半径 ≤ α < 1 保证收敛）× 树分支 RandomForest 自训练概率，`P = β·P_rf + (1-β)·F`；
2. **验证集网格选 β**：`β ∈ {0, .25, .5, .75, 1}` 在独立 val 上最大化 macro-F1（无测试泄漏）；
3. **类平衡选择**：逐类预算 ∝ 类先验 + 逐类池内置信分位阈值（相对阈值，随难度自适应，防多数类霸占伪标签）；
4. **课程退火 + 置信加权**：分位数 `q_t = q0·anneal^t` 逐轮下降；伪标签样本权重 = 混合置信度。

离线兜底：sklearn 缺失时树分支降级纯 numpy kNN，图分支本就纯 numpy（Tier-1 全链路可跑）。

## 5. 评测协议

- 6 数据集 × 3 seeds（V4 统计严谨），mean±std；显著性：`Δ > 0.5·(σ₁+σ₂)`
- 切分：stratified，labeled(8/类) ⊥ val(100) ⊥ test(200) ⊥ unlabeled，索引级互斥
- 数据难度梯度：moons(noise=0.32)、circles(0.14)、blobs 重叠(std=2.1)、四高斯 XOR、双螺旋、digits 真实 6 类
- Gate（定死于方案阶段）：aggregate ΔF1 ≥ +0.02 且显著；旗舰 ≥ 半数数据集第一；兜底路径可跑

## 6. 实测结果（benchmark.json 真实输出，n=600, 3 seeds）

| 数据集 | GACS（旗舰） | 最强基线 | Δ |
|--------|-------------|----------|---|
| xor_gauss | **0.9733±0.0024** | label_spread 0.9065 | +0.0668 |
| spiral | **0.6727±0.0335** | supervised_rf 0.6381 | +0.0346 |
| digits_sub | **0.9237±0.0238** | selftrain_rf 0.9215 | +0.0022 |
| blobs_overlap | 0.4182 | supervised_rf 0.4274 | −0.0092 |

- aggregate：GACS **0.7711±0.0307** vs supervised_rf 0.7176±0.0352 → **Δ=+0.0536, 显著=True**（门槛 0.02）
- 消融（pooled, 2 数据集 × 2 seeds）：full 0.8180 / no_graph 0.7387 / no_balance 0.8140 / no_anneal 0.8180
  → 结论：**图分支是核心增益来源（−0.079）**；类平衡微正（−0.004）；退火在本数据集上中性
- 失败案例 3 条（全部从 results 派生）：blobs_overlap 无增益（标注已足）、单 seed 波动、label_prop 高维坍塌

## 7. 确定性

`core.seed.set_all(seed)` 唯一入口；RF/numpy 全部 seed 化；无并行非确定性源。
同 seed 两次运行 benchmark.json 除 elapsed_sec 外逐位一致（已实测）。

## 8. 性能预算

demo 端到端（6 数据集 × 3 seeds × 6 方法 + 消融）：**~45s**（CPU）≤ 60s ✓；内存峰值 < 500MB。
