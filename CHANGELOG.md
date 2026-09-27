# Changelog

## v0.1.0 (2026-09-28)

- 首发：半监督学习系统 SemiForge
- 旗舰 GACS：图-树双分支混合自训练（图传播 × RF 自训练，val 网格选 β）
- 类平衡预算 + 逐类池内分位阈值 + 课程退火 + 置信度样本加权
- 6 数据集 × 3 seeds benchmark：aggregate macro-F1 vs 最强基线 +0.0536（显著）
- Tier-0 sklearn（SelfTrainingClassifier/LabelSpreading/LabelPropagation）+ Tier-1 纯 numpy 离线兜底
- 32 单测 / 确定性逐位一致 / CLI（demo/tune）/ Optuna HPO 可选 / CI（lint+pytest+demo 冒烟）

---

作者：晨星（CJX0712）
