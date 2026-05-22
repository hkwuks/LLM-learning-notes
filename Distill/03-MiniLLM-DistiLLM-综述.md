# MiniLLM & DistiLLM 综述

## 概述

MiniLLM 和 DistiLLM 是 On-Policy Distillation 领域中两种重要的方法，分别针对不同的优化目标（Reverse KL 和 Skewed KL）进行设计，以解决知识蒸馏中的模式坍塌和训练稳定性问题。

---

## MiniLLM: Reverse KL 优化

### 核心思想

MiniLLM 采用 **Reverse KL 散度**作为优化目标：

$$D_{KL}(p_{student} \| p_{teacher}) = \sum_y p_{student}(y|x) \log \frac{p_{student}(y|x)}{p_{teacher}(y|x)}$$

### 数学优势

与 Forward KL 相比，Reverse KL 具有以下特性：

1. **模式寻找 (Mode-seeking)**:
   - 倾向于匹配教师分布的峰值区域
   - 避免在低概率区域分配质量
   - 生成质量更高但多样性可能降低

2. **零避免 (Zero-avoiding)**:
   - 当 $p_{teacher} \approx 0$ 时，$p_{student}$ 也被推近 0
   - 避免学生产生教师认为不可能的输出

3. **优化稳定性**:
   - 相比 Forward KL，训练过程更稳定
   - 不易受到学生分布长尾的影响

### 训练目标

$$\min_{\theta} \mathbb{E}_{x \sim p_{student}}[D_{KL}(p_{student}(y|x) \| p_{teacher}(y|x))]$$

### 实际实现

由于直接计算期望涉及从学生采样，MiniLLM 使用重要性采样：

$$\mathcal{L}_{MiniLLM} = \mathbb{E}_{y \sim p_{student}}\left[\log \frac{p_{student}(y|x)}{p_{teacher}(y|x)}\right]$$

**关键技巧**: 使用学生自己的采样分布来计算梯度，避免了分布不匹配问题。

---

## DistiLLM: Skewed KL 稳定性改进

### 核心思想

DistiLLM 引入 **Skewed KL 散度**来平衡 Forward KL 和 Reverse KL 的优势：

$$D_{SKL}^{\alpha}(P \| Q) = D_{KL}(P \| \alpha P + (1-\alpha)Q)$$

其中 $\alpha \in [0, 1]$ 是倾斜参数。

### 特殊情况

- **$\alpha = 0$**: 退化为 Forward KL
- **$\alpha = 1$**: 退化为 Reverse KL
- **$0 < \alpha < 1$**: 中间行为，结合两者优势

### 优势分析

| 特性 | Forward KL | Reverse KL | Skewed KL |
|------|------------|------------|-----------|
| 模式覆盖 | ✅ 强 | ❌ 弱 | 🟡 可调 |
| 模式寻找 | ❌ 弱 | ✅ 强 | 🟡 可调 |
| 训练稳定性 | ❌ 易发散 | 🟡 中等 | ✅ 稳定 |
| 零避免行为 | ❌ 否 | ✅ 是 | ✅ 是 |
| 计算效率 | ✅ 高 | 🟡 中等 | 🟡 中等 |

### Skewed KL 的直观理解

Skewed KL 在目标分布中混合了学生自身的分布，这提供了一种"正则化"效果：

$$P_{target} = \alpha P_{student} + (1-\alpha)P_{teacher}$$

当学生偏离教师太远时，混合分布会提供一种"锚定"效应，防止过度优化。

### 自适应 Skewing

DistiLLM 还可以使用自适应的 $\alpha$：

$$\alpha_t = \alpha_{min} + (\alpha_{max} - \alpha_{min}) \cdot f(t)$$

其中 $f(t)$ 可以是：
- **线性**: $f(t) = t/T$
- **指数**: $f(t) = 1 - e^{-t/\tau}$
- **基于性能**: 根据验证损失调整

---

## 两种方法的对比

### 应用场景

**选择 MiniLLM (Reverse KL) 当**:
- 任务对输出质量要求高（如代码生成、数学推理）
- 需要避免生成荒谬/不可能的内容
- 教师分布相对集中

**选择 DistiLLM (Skewed KL) 当**:
- 需要平衡质量和多样性
- 训练过程中观察到不稳定性
- 希望精细控制模式覆盖 vs 模式寻找

### 计算复杂度

```
MiniLLM:
  前向: O(|V| · L)  # 词汇表大小 × 序列长度
  反向: O(|V| · L)  # 需要计算学生分布

DistiLLM:
  前向: O(|V| · L) + O(|V|)  # 额外计算混合分布
  反向: O(|V| · L)
```

---

## 实现伪代码

### MiniLLM

```python
def mini_llm_loss(student_logits, teacher_logits, student_sample):
    """
    student_logits: [batch, seq_len, vocab]
    teacher_logits: [batch, seq_len, vocab]
    student_sample: [batch, seq_len] - 学生采样的token
    """
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    
    # Reverse KL: KL(student || teacher)
    # 使用学生样本估计期望
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # 收集采样位置的log概率
    gathered_student = torch.gather(
        student_log_probs, 
        -1, 
        student_sample.unsqueeze(-1)
    ).squeeze(-1)
    
    gathered_teacher = torch.gather(
        teacher_probs, 
        -1, 
        student_sample.unsqueeze(-1)
    ).squeeze(-1)
    
    # Reverse KL ≈ log p_student(y) - log p_teacher(y)
    loss = (gathered_student - torch.log(gathered_teacher + 1e-10)).mean()
    
    return loss
```

### DistiLLM

```python
def distillm_loss(student_logits, teacher_logits, alpha=0.5):
    """
    Skewed KL: KL(P || αP + (1-α)Q)
    """
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    
    # 混合分布
    mixed_probs = alpha * student_probs + (1 - alpha) * teacher_probs
    
    # Skewed KL
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    loss = F.kl_div(
        student_log_probs,
        mixed_probs,
        reduction='batchmean'
    )
    
    return loss
```

---

## 训练技巧

### 1. 温度退火

使用退火温度来平滑概率分布：

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- 初始 $T = 2.0$，逐步降至 $T = 1.0$

### 2. Top-k/Top-p 截断

在计算 KL 时只考虑头部概率：

```python
def truncate_topk(probs, k=50):
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return topk_probs, topk_indices
```

### 3. 梯度累积

由于需要学生采样，有效 batch size 较小，使用梯度累积：

```python
# 每4步累积一次梯度
accumulation_steps = 4
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

---

## 关键超参数

| 参数 | MiniLLM | DistiLLM | 建议范围 |
|------|---------|----------|----------|
| 学习率 | 1e-5 | 1e-5 | 1e-6 ~ 5e-5 |
| Temperature | 1.0-2.0 | 1.0-1.5 | 0.7 ~ 2.0 |
| Alpha (Skew) | N/A | 0.3-0.7 | 0.0 ~ 1.0 |
| Batch size | 16-32 | 16-32 | 8 ~ 64 |
| 序列长度 | 512-2048 | 512-2048 | 256 ~ 4096 |

---

## 相关论文

1. MiniLLM: Knowledge Distillation of Large Language Models
2. DistiLLM: Towards Streamlined Distillation for Large Language Models
3. A Survey of On-Policy Distillation for Large Language Models

---

## 工具和资源

- **HuggingFace TRL**: `trl.experimental.minillm_trainer`, `trl.experimental.distillm_trainer`
- **GitHub**: 查看 Awesome-LLM-On-Policy-Distillation 仓库获取实现链接