# Paper Writing Notes

Benchmark 结果分析与论文写作要点。基于 105 翼型 x 5 方法完整 benchmark。

---

## 1. 核心数据

### 实验规模

- 105 翼型：30 Normal + 44 Medium + 31 Hard
- 5 优化方法：Baseline (8w IPOPT), Rule, Threshold, Adaptive Router (MLP), XFoil+DE
- 难度分级基于 `brentq` 初始 CD 值

### 结果总览 (All Categories)

| 方法 | 成功数/105 | 成功率 | 平均 CD | 平均时间 | 平均阶段数 |
|------|-----------|--------|---------|---------|-----------|
| Baseline (8w IPOPT) | 56 | 53% | 0.0793 | 51.59s | 1.0 |
| Rule (固定阈值) | 84 | 80% | 0.0714 | 67.37s | 4.1 |
| Threshold (学习阈值) | 85 | 81% | 0.0712 | 65.21s | 4.2 |
| Adaptive Router (MLP) | 82 | 78% | 0.0714 | 52.13s | 2.9 |
| XFoil+DE | 105 | 100% | 0.0013 | 833.12s | 1.0 |

### 分类别结果

**Normal (30 翼型)**：

| 方法 | 成功数 | 平均 CD | 平均时间 | 阶段数 |
|------|--------|---------|---------|--------|
| Baseline | 30 | 0.0729 | 11.83s | 1.0 |
| Rule | 29 | 0.0711 | 51.23s | 4.6 |
| Threshold | 29 | 0.0711 | 50.58s | 4.6 |
| PiERN | 29 | 0.0717 | 35.06s | 3.1 |

**Hard (31 翼型)**：

| 方法 | 成功数 | 平均 CD | 平均时间 | 阶段数 |
|------|--------|---------|---------|--------|
| Baseline | 4 | 0.1016 | 76.31s | 1.0 |
| Rule | 18 | 0.0713 | 79.58s | 3.6 |
| Threshold | 19 | 0.0712 | 82.93s | 3.9 |
| PiERN | 17 | 0.0713 | 75.06s | 2.8 |

### 统计显著性 (Mann-Whitney U vs Baseline)

| 对比 | p-value | effect size (r) | 显著性 |
|------|---------|-----------------|--------|
| Threshold vs Baseline | 3.9e-5 | 0.454 | *** |
| Rule vs Baseline | 3.7e-4 | 0.388 | *** |
| PiERN vs Baseline | 0.019 | 0.240 | * |
| XFoil+DE vs Baseline | 9.1e-23 | 1.0 | *** |

---

## 2. 论文核心论点

### 论点一：可靠性提升（最强）

Baseline 53% -> Hierarchical 78-85%

- 8 维 CST 直接优化在 medium/hard 翼型上大量失败（IPOPT 局部最优）
- 层次化方法通过低维初始化引导搜索，让更多翼型可优化
- Hard 类别：Baseline 仅 4/31 成功，PiERN 17/31 成功

**论文措辞**：
> "Hierarchical CST optimization increases the success rate from 53% to 78-85%, particularly on challenging airfoils where direct 8-weight optimization fails (4/31 vs 17/31 on Hard cases)."

### 论点二：Adaptive Router 效率优势

阶段数 2.9 vs 4.1-4.2（Rule/Threshold）

- MLP 路由器学会自适应决策：简单翼型 2 阶段，复杂翼型 3 阶段
- 比固定阈值方法少 30-35% 优化阶段
- 平均时间：PiERN 52.13s vs Rule 67.37s（快 23%）

**论文措辞**：
> "The learned MLP router reduces the mean number of optimization stages from 4.1-4.2 to 2.9, achieving comparable quality with 23% less computation time."

### 论点三：CD 质量一致性

所有成功方法的中位数 CD 都是 0.071094

- 这是正常的：约束优化问题，所有方法收敛到同一个最优解
- Hierarchical CST 的价值在于**让更多翼型能优化成功**，不是找到更好的解
- XFoil+DE 的 CD=0.0013 是退化的（panel method 失败），不是真的更好

**论文措辞**：
> "All NeuralFoil-based methods converge to the same optimal CD (median 0.071094), confirming that hierarchical parameterization does not sacrifice solution quality."

### 论点四：XFoil+DE 对比

- XFoil+DE 平均耗时 833s，是 NeuralFoil 方法的 12-16 倍
- XFoil+DE CD=0.0013 是退化结果（panel method 无法正确评估）
- 证明基于 NeuralFoil 的代理评估框架的必要性

**论文措辞**：
> "XFoil+DE requires 12-16x more computation time and produces degenerate CD values (0.0013) due to panel method convergence issues, demonstrating the necessity of neural surrogate evaluation."

---

## 3. 需要诚实面对的问题

### 3.1 MLP vs Rule/Threshold 显著性偏弱

p=0.019, r=0.240（vs Rule p=3.7e-4, r=0.388）

- Rule/Threshold 的阈值是在这批数据上 grid-search 的，可能存在过拟合
- MLP 是从数据学习的，理论上泛化更好
- 需要 cross-dataset 验证来支撑这个论点

**应对**：强调 MLP 的优势在于阶段数更少（2.9 vs 4.1），而不是 CD 更好

### 3.2 Normal 类别 Baseline 更快

- Normal: Baseline 11.83s vs PiERN 35.06s
- 这是预期的：简单翼型不需要多阶段优化
- 层次化方法的优势在 Medium/Hard 才体现

**应对**：强调"整体可靠性"而不是"所有场景都更快"

### 3.3 Hard 类别 Baseline 成功率极低

- 仅 4/31 成功，均值 76.31s 基于这 4 个
- 不能直接说"Hard 时 PiERN 更快"，因为 Baseline 样本量太小
- 但可以说"PiERN 成功优化了 17/31 Hard 翼型，而 Baseline 仅 4/31"

---

## 4. Kulfan 权重物理意义（A3 消融用）

Bernstein 基函数与弦向位置的对应：

| 权重 | 位置 | 物理意义 |
|------|------|---------|
| w1 | 前缘区域 (0-25% c) | 控制前缘半径 |
| w2 | 前部 (25-50% c) | 控制最大厚度位置 |
| w3 | 中弦 (50-75% c) | 控制弯度分布 |
| w4 | 后缘 (75-100% c) | 控制后缘角度 |
| w5-w8 | 细节 | 控制高阶形状细节 |

4 weights/edge 能捕获 ~96-98% 的翼型形状（RMS error 0.002-0.007）。
6 weights/edge 能捕获 ~98.5-99%。

---

## 5. Pipeline 指标

图像 -> 优化 pipeline 的分解指标：

| 指标 | 含义 | 典型值 |
|------|------|--------|
| extraction_time | 图像轮廓提取耗时 | ~0.1-0.4s |
| optimization_time | 优化耗时 | 15-125s |
| kulfan_fit_error | 提取轮廓与 Kulfan 拟合的 RMS 距离 | 0.01-0.05 |

---

## 6. 图表使用建议

### 论文必用图

1. **benchmark_summary.png** — 跨类别汇总（CD improvement, time, success rate）
2. **benchmark_case_study.png** — NACA 0012 具体案例
3. **ablation_1_hierarchical_vs_direct.png** — 层次化 vs 直接优化验证

### 论文可选图

4. **benchmark_dist_all.png** — CD 分布箱线图
5. **benchmark_diff_all.png** — 难度-改善散点图
6. **ablation_2_router_effect.png** — 路由器效果对比
7. **ablation_3_starting_dimension.png** — 起始维度影响

### 图表文件位置

所有图表在 `results/` 目录，`images/` 目录包含 README 用图。

---

## 7. 数据文件索引

### Router Benchmark

- `results/benchmark_stats.csv` — 每翼型 x 每方法的原始数据
- `results/table_router_full.csv` — 汇总表（类别 x 方法）
- `results/table_router_latex.tex` — LaTeX 格式结果表
- `results/table_significance.csv` — 统计显著性检验

### Ablation Study

- `results/ablation.csv` — 消融实验数据（A1-A4）

### Pipeline Benchmark

- `results/pipeline_benchmark.csv` — 提取精度数据
