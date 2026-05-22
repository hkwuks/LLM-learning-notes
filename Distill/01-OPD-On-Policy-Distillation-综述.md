# On-Policy Distillation (OPD) 综述

## 概述

On-Policy Distillation (OPD) 是一种革新性的知识蒸馏范式，通过重组训练循环来解决传统监督微调(SFT)中的**曝光偏差(Exposure Bias)**问题。

## 核心思想

传统知识蒸馏中，学生在完美的教师前缀上训练，但在推理时必须生成自己的前缀。小错误会累积成学生很少训练过如何恢复的轨迹。曝光偏差的严重程度与序列长度的**平方**成正比。

OPD 的关键创新：**教师对学生实际产生的输出提供反馈**，而非让学生模仿教师的完美输出。

## 数学形式化

OPD 被形式化为在学生采样轨迹上的 **$f$-散度最小化**问题：

$$\min_{\theta} \mathbb{E}_{x \sim p_{\theta}}[D_f(p_{\theta}(y|x) \| p_{teacher}(y|x))]$$

其中：
- $p_{\theta}$: 学生策略
- $p_{teacher}$: 教师策略
- $D_f$: $f$-散度（Forward KL、Reverse KL、JSD 等）

## 三大设计维度

### 1. 优化目标 (What to optimize)
- **Forward KL**: 模式覆盖，鼓励学生探索多样化输出
- **Reverse KL**: 模式寻找，聚焦高概率区域
- **JSD (Jensen-Shannon Divergence)**: 平衡两者

### 2. 信号来源 (Where the signal comes from)
- 外部教师模型
- 多教师集成
- 自蒸馏（模型自身作为教师）
- 奖励模型

### 3. 训练稳定性 (How to stabilize)
- KL 约束
- 熵感知正则化
- 重要性采样
- 课程学习

## 与 RL 的联系

OPD 与 KL 约束的强化学习密切相关：

$$\max_{\theta} \mathbb{E}_{x \sim p_{\theta}}[r(x)] - \beta D_{KL}(p_{\theta} \| p_{ref})$$

当奖励来自教师模型时，OPD 可视为一种特殊的 RL 形式。

## 核心优势

| 特性 | 传统 KD | OPD |
|------|---------|-----|
| 训练数据 | 教师生成的静态数据 | 学生生成的动态数据 |
| 错误累积 | $O(L^2)$ | $O(L)$ |
| 泛化能力 | 受限于训练分布 | 在策略分布上优化 |
| 计算成本 | 较低 | 中等（需在线采样） |

## 代表性工作

1. **GKD (Generalized Knowledge Distillation)** - 建立标准框架
2. **MiniLLM** - Reverse KL 优化
3. **DistiLLM** - Skewed KL 稳定性改进
4. **G-OPD** - 理论统一

## 实现资源

- **HuggingFace TRL**: `trl.experimental` 命名空间包含 OPD 实现
- **GitHub**: nick7nlp/Awesome-LLM-On-Policy-Distillation

## 关键论文

- A Survey of On-Policy Distillation for Large Language Models (arXiv:2604.00626)

## 应用场景

- 长序列推理任务
- 多步工具使用
- 代码生成
- 数学推理

## 与 SFT 的对比

```python
# 传统 SFT (离线)
loss = -log p_student(y_teacher | x)

# OPD (在线)
y_student = sample(p_student, x)
loss = D_f(p_student(·|x) || p_teacher(·|x, y_student))
```