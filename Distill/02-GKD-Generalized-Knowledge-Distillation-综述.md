# GKD (Generalized Knowledge Distillation) 综述

## 概述

GKD (Generalized Knowledge Distillation) 是 On-Policy Distillation (OPD) 领域的奠基性工作之一，提出了一个灵活的框架来缓解传统知识蒸馏中的曝光偏差问题。

## 核心贡献

GKD 建立了 OPD 的标准框架，通过显式地混合学生和真实数据来桥接训练与推理之间的差距。

## 方法详解

### 数据混合策略

GKD 引入混合参数 $\lambda \in [0, 1]$ 来平衡学生生成数据和教师/真实数据：

$$x_{mix} = \begin{cases} x_{student} & \text{with probability } \lambda \\ x_{ground\_truth} & \text{with probability } 1-\lambda \end{cases}$$

### 训练目标

$$\mathcal{L}_{GKD} = \mathbb{E}_{(x,y) \sim D_{mix}}[D_{KL}(p_{teacher}(y|x) \| p_{student}(y|x))]$$

### 散度选择

GKD 对散度选择保持中立，实验证明以下三种散度都能取得竞争性能：

1. **Forward KL**: 
   $$D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$
   - 模式覆盖，鼓励学生探索

2. **Reverse KL**: 
   $$D_{KL}(Q \| P) = \sum_i Q(i) \log \frac{Q(i)}{P(i)}$$
   - 模式寻找，聚焦高概率区域

3. **JSD (Jensen-Shannon Divergence)**:
   $$JSD(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$$
   其中 $M = \frac{1}{2}(P + Q)$

## 算法流程

```
Algorithm GKD Training:
1. Initialize student model θ from pre-trained checkpoint
2. For each training step:
   a. Sample batch of prompts {x}
   b. For each λ in [0, 1]:
      - Generate student outputs: y_student ~ p_θ(·|x)
      - Sample mixed input: x_mix ~ Bernoulli(λ)
        * If λ: use y_student as prefix
        * Else: use ground truth prefix
      - Compute KL divergence loss
   c. Update student parameters θ
3. Return trained student model
```

## 关键设计决策

### 1. 混合参数 λ 调度

- **固定 λ**: 通常设置为 0.5
- **退火策略**: 从 0 逐渐增加到 1
- **自适应 λ**: 基于训练进度动态调整

### 2. 采样策略

- **Temperature 退火**: 从高温度采样开始，逐渐降低
- **Top-p/Nucleus 采样**: 控制采样多样性
- **束搜索**: 用于确定性推理场景

### 3. 长度处理

- **截断**: 限制最大生成长度
- **动态长度**: 根据任务自适应调整

## 优势与局限

### 优势

- 简单直观的混合机制
- 兼容多种散度度量
- 实现复杂度适中
- 泛化性能提升显著

### 局限

- 需要平衡学生和真实数据比例
- 在长序列上计算成本增加
- 对超参数 λ 敏感

## 变体与扩展

### GKD with Reward Guidance

结合奖励模型进行过滤：

$$\mathcal{L}_{GKD-R} = \mathbb{E}[r(x,y) \cdot D_{KL}(p_{teacher} \| p_{student})]$$

### Multi-round GKD

迭代式蒸馏，每轮使用上一轮的学生作为新学生：

$$\theta^{(t+1)} = \arg\min_{\theta} \mathcal{L}_{GKD}(\theta; \theta^{(t)})$$

## 实现要点

```python
# 伪代码示例
import torch
import torch.nn.functional as F

def gkd_loss(student, teacher, x, y_true, lambda_mix=0.5):
    # 学生生成
    if torch.rand(1) < lambda_mix:
        y_prefix = student.generate(x, max_length=partial_len)
    else:
        y_prefix = y_true[:partial_len]
    
    # 计算 KL 散度
    student_logits = student(x, y_prefix)
    teacher_logits = teacher(x, y_prefix)
    
    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    
    return loss
```

## 实验设置建议

1. **λ 搜索**: 在 {0.3, 0.5, 0.7} 中验证
2. **散度选择**: 根据任务性质选择 Forward/Reverse/JSD
3. **温度**: 采样温度从 1.0 开始，逐步降至 0.7
4. **学习率**: 通常比 SFT 低 2-5 倍

## 相关论文

- Generalized Knowledge Distillation for Efficient Large Language Model Distillation
- A Survey of On-Policy Distillation for Large Language Models (arXiv:2604.00626)

## 应用案例

- 长文本生成任务
- 对话系统蒸馏
- 代码补全模型压缩
- 多语言模型迁移