# PhysGAT 源码模块概览（src/）

本文面向开发者，概述 src 目录各模块的职责、关键类/函数、在整体工作流中的位置、与其他模块的依赖关系，并提供最小可用的使用示例，帮助快速理解与扩展系统。

目录：
- analysis.py
- config_validator.py
- data_utils.py
- ensemble.py
- model.py
- proj_fix.py
- trainer.py
- utils.py
- visualization.py
- __init__.py

---

## analysis.py — 结果后处理与报告汇总
- 主要功能：
  - 计算内梅罗指数与污染等级；
  - 汇总源类型贡献（全量与 Top5 版本），并生成归一化后的透视表；
  - 计算主要/次要污染物；
  - 产出用于可视化与报告的分析结构；
  - 可选：XAI/Explainer 辅助分析接口。
- 关键函数：
  - calculate_nemerow_index(df, metals, config) → DataFrame
  - get_source_contributions_summary(predictions_df, sources_df) → (type_summary, pivot_summary)
  - get_normalized_source_contributions_summary(top5_report_df) → (type_summary, pivot_summary)
  - generate_contribution_report(predictions_df, receptors_df, sources_df, config=None)
  - calculate_primary_secondary_pollutants(receptors_df, metals, config=None)
  - filter_and_renormalize_contributions(...)
  - run_full_analysis(predictions_df, dataset, config) → dict
- 在工作流中的位置：训练/预测之后，承接预测结果与数据集，生成分析产物，供 visualization.py 使用。
- 依赖关系：依赖 pandas/numpy；与 visualization.py 强耦合，消费 trainer/ensemble 的预测输出与 data_utils 的数据字段。
- 使用示例：
```python
from physgat_project.src.analysis import run_full_analysis
results = run_full_analysis(pred_df, dataset, config)
pivot = results["contribution_pivot_summary"]
```

---

## config_validator.py — 配置校验与自动修复
- 主要功能：对 DictConfig/Hydra 配置进行完整性与取值合法性校验，必要时自动修复并返回告警/修复信息。
- 关键实体：
  - class ConfigValidator
  - validate_and_fix_config(config) → (fixed_config, changed: bool, warnings: list, errors: list)
- 在工作流中的位置：主流程启动与数据/训练前，应先校验配置，避免运行中断或隐性错误。
- 依赖关系：与 data_utils、trainer、ensemble 等模块共享配置命名空间。
- 使用示例：
```python
from physgat_project.src.config_validator import validate_and_fix_config
config, changed, warns, errs = validate_and_fix_config(config)
```

---

## data_utils.py — 数据加载、清洗、PMF与图构建
- 主要功能：
  - 读取原始 CSV/表格，执行缺失值/异常值处理；
  - 生成地形/风场/水文等物理先验特征（DEM 可选）；
  - 运行 PMF 预处理（含诊断与因子数选择），将 PMF 因子作为受体特征；
  - 构建源-受体图（PyG Data），并缓存处理结果；
  - 提供 InMemoryDataset：PhysGATDataset。
- 关键类/函数（节选）：
  - class PhysGATDataset(InMemoryDataset)
  - _run_preliminary_pmf, _run_pmf_diagnostics, _select_optimal_pmf_factors
  - 多个物理特征计算函数：_calculate_enhanced_distance_weights、_calculate_wind_influence_range、_calculate_atmospheric_dispersion_weights 等
- 在工作流中的位置：最先运行，产出图数据与诊断信息，供模型与可视化使用。
- 依赖关系：scikit-learn、torch_geometric、numpy、pandas；与 model.py/trainer.py/visualization.py/analysis.py 共享字段命名。
- 使用示例：
```python
from physgat_project.src.data_utils import PhysGATDataset
dataset = PhysGATDataset(root=config.paths.data_root, config=config)
pyg_data = dataset[0]
```

---

## ensemble.py — 集成训练与不确定性估计
- 主要功能：
  - 训练多个独立模型并聚合预测（如截尾平均），抑制单模型偶然性；
  - 基于成员间差异度量不确定性；
  - 记录/合并训练历史以供可视化。
- 关键实体：
  - class EnsembleTrainer
- 在工作流中的位置：封装多模型训练与推理，向 analysis/visualization 提供稳健预测。
- 依赖关系：依赖 model.py 与 trainer.py；与 analysis.py/visualization.py 对接输出。
- 使用示例：
```python
from physgat_project.src.ensemble import EnsembleTrainer
ens = EnsembleTrainer(config)
model_data = ens.train_and_predict(dataset)
```

---

## model.py — GATv2 编码器与解码器
- 主要功能：
  - GATv2 基础的多跳（multi-hop）图注意力编码器；
  - 源→受体的跨域注意力（S2R）/融合；
  - 链路解码与最终输出头；
  - 封装综合模型 PhysGAT。
- 关键类：
  - class PhysGATModel(nn.Module)
  - class LinkDecoder(nn.Module)
  - class PhysGAT(nn.Module)
- 在工作流中的位置：对 data_utils 产出的图进行表征学习与归因预测。
- 依赖关系：torch、torch_geometric。
- 使用示例：
```python
from physgat_project.src.model import PhysGAT
model = PhysGAT(receptor_in_channels=..., source_in_channels=..., hidden_channels=..., out_channels=..., num_heads=..., dropout_rate=..., hops=[1,2,3])
```

---

## trainer.py — 单模型训练器
- 主要功能：
  - 复合损失（化学/源强/距离）与加权；
  - 学习率调度（ReduceLROnPlateau）与早停；
  - 记录训练历史（total/chemistry/strength/distance/learning_rate）。
- 关键实体：
  - class Trainer
- 在工作流中的位置：训练单个 PhysGAT 模型，供 ensemble.py 调用或独立使用。
- 依赖关系：torch、model.py；与 analysis.py/visualization.py 共享训练历史结构。
- 使用示例：
```python
from physgat_project.src.trainer import Trainer
trainer = Trainer(model, config)
model_data = trainer.train(dataset)
```

---

## visualization.py — 可视化套件（Figures 1–18）
- 主要功能：
  - VisualizationSuite 生成统一风格的科研图表：
    - Figure 1（源端化学指纹）、Figure 2（受体重金属相关性）；
    - Figures 3–6（各源贡献热力图矩阵）；
    - Figure 7（训练过程曲线）；
    - Figures 8–10（受体/源空间分布与综合连边示意图）；
    - Figure 11（PMF 诊断碎石图）；
    - Figure 12（最终溯源结论堆叠柱状图）；
    - Figure 13–18（不确定性/一致性检查/XAI/参数敏感性/DEM 3D 等）。
- 关键实体：
  - class VisualizationSuite
  - 代表性方法：figure1_source_chemical_fingerprints、figure2_soil_correlation_heatmap、figure3_6_source_contribution_heatmaps、figure7_training_process、figure8_receptor_pollution_spatial、figure9_source_spatial_distribution、figure10_ego_graph_spatial、figure11_pmf_scree_plot_enhanced、figure12_contribution_barcharts_extended 等。
- 在工作流中的位置：消费 analysis/trainer/ensemble 输出，生成图件与报告素材。
- 依赖关系：matplotlib、seaborn、geopandas/shapely（如有）、pandas；与 analysis.py 严格对齐数据列命名；与 data_utils/model 的字段含义一致。
- 使用示例：
```python
from physgat_project.src.visualization import VisualizationSuite
viz = VisualizationSuite(analysis_results, dataset, model_data, output_dir)
viz.figure12_contribution_barcharts_extended()
```

---

## proj_fix.py — PROJ/GDAL 环境兼容性修复
- 主要功能：配置坐标投影库（PROJ）运行环境，抑制噪声告警，提供一键修复与装饰器包装，确保地理计算可重复且健壮。
- 关键实体：
  - setup_proj_environment(), suppress_proj_warnings(), verify_proj_functionality(), apply_proj_fix()
  - class ProjEnvironment
  - with_proj_fix(func) 装饰器
- 在工作流中的位置：在数据加载/空间计算前调用（如需要）。
- 依赖关系：pyproj/PROJ（如使用），logging；与 data_utils/visualization 的空间功能相配合。
- 使用示例：
```python
from physgat_project.src.proj_fix import apply_proj_fix
apply_proj_fix()
```

---

## utils.py — 通用工具
- 主要功能：
  - set_seed(seed) 设定全局随机种子，保证实验可复现。
- 使用示例：
```python
from physgat_project.src.utils import set_seed
set_seed(42)
```

---

## __init__.py — 包初始化
- 主要功能：定义包级导出与版本元数据（如有）。当前实现轻量，主要用于标识包。

---

## 模块关系与典型工作流
1) 数据阶段：data_utils.py → 构建图与特征（含 PMF 诊断输出）
2) 模型阶段：model.py → 定义 PhysGAT；trainer.py → 训练单模；ensemble.py → 集成训练/推理与不确定性
3) 分析阶段：analysis.py → 汇总贡献、内梅罗指数、Top5 清单等
4) 可视化阶段：visualization.py → 生成 Figures 1–18 科研图表
5) 环境与工具：config_validator.py（配置校验）、proj_fix.py（坐标环境修复，可选）、utils.py（随机种子）

以上模块协同实现“物理先验 + 图注意力 + 集成不确定性”的端到端污染源精准溯源流程。
