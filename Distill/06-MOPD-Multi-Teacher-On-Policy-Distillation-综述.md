# MOPD (Multi-Teacher On-Policy Distillation) 综述

## 概述

MOPD (Multi-Teacher On-Policy Distillation) 是一种**多教师在线策略蒸馏**方法，允许单个学生模型从多个专业教师模型中学习，将不同领域的能力融合到一个统一模型中。

## 核心问题

### 后训练的挑战

现代 LLM 后训练通常分为多个阶段：

```
基础模型 → SFT → 领域RL → RLHF → 部署
            ↓
      多个领域专家
         (数学、代码、推理)
```

**问题**: 如何将这些不同领域的专家能力整合到一个统一模型中？

### 简单合并的局限

1. **直接平均参数**: 破坏预训练知识
2. **顺序微调**: 灾难性遗忘
3. **任务向量相加**: 需要相同的模型架构
4. **MoE**: 推理成本增加

## MOPD 核心思想

### 关键洞察

**不合并参数，而是合并分布。**

学生从每个教师在其自己的 rollouts 上学习，教师提供token级别的指导。

### 数学形式化

给定 $K$ 个教师模型 $\{T_1, T_2, ..., T_K\}$，学生 $S$ 的目标是：

$$\min_{\theta_S} \sum_{k=1}^{K} w_k \cdot \mathbb{E}_{y \sim p_S}[D_{KL}(p_S(y|x) \| p_{T_k}(y|x, y_{prefix}))]$$

其中 $w_k$ 是第 $k$ 个教师的权重。

### 多教师融合策略

#### 策略 1: 加权平均

$$p_{target}(y|x) = \sum_{k=1}^{K} w_k \cdot p_{T_k}(y|x)$$

权重可以是：
- **均匀**: $w_k = 1/K$
- **基于性能**: $w_k \propto \text{perf}(T_k)$
- **自适应**: $w_k(x)$ 根据输入动态选择

#### 策略 2: 最大值融合

$$p_{target}(y|x) = \max_k p_{T_k}(y|x)$$

保留每个教师最强的知识。

#### 策略 3: 门控融合

$$p_{target}(y|x) = \sum_{k=1}^{K} g_k(x) \cdot p_{T_k}(y|x)$$

其中 $g_k(x) = \text{softmax}(W_g \cdot \text{encode}(x))_k$ 是门控网络。

## 方法详解

### 训练流程

```
Algorithm MOPD:
1. 初始化学生模型 S
2. 准备 K 个教师模型 {T_1, ..., T_K}
3. For each training batch:
   a. 学生生成: y_student ~ S(x)
   b. For each 教师 k:
      - 获取教师反馈: p_Tk(y|x, y_student)
      - 计算 KL 散度: L_k = KL(S || T_k)
   c. 融合损失: L = Σ w_k · L_k
   d. 更新学生: θ_S ← θ_S - α∇L
4. Return 融合后的学生模型
```

### 教师选择策略

#### 静态选择

训练前固定教师集合：

```python
teachers = {
    'math': math_expert,
    'code': code_expert,
    'reasoning': reasoning_expert
}

weights = {'math': 0.4, 'code': 0.4, 'reasoning': 0.2}
```

#### 动态选择

根据输入内容动态选择相关教师：

```python
def select_teachers(x, all_teachers):
    """基于输入选择教师"""
    # 使用简单分类器或启发式规则
    if is_math_question(x):
        return [math_expert, reasoning_expert]
    elif is_code_question(x):
        return [code_expert, reasoning_expert]
    else:
        return all_teachers
```

#### 课程式选择

训练过程中逐渐增加教师数量：

```python
# 早期: 少而精
if epoch < 10:
    active_teachers = [base_teacher]
# 中期: 加入专家
elif epoch < 20:
    active_teachers = [base_teacher, math_teacher]
# 后期: 全部
else:
    active_teachers = all_teachers
```

## 伪代码实现

```python
class MOPDTrainer:
    def __init__(self, student, teachers, teacher_weights=None):
        self.student = student
        self.teachers = teachers  # List of teacher models
        
        if teacher_weights is None:
            self.teacher_weights = [1.0 / len(teachers)] * len(teachers)
        else:
            self.teacher_weights = teacher_weights

    def compute_mopd_loss(self, batch):
        x = batch['input_ids']
        
        # 1. 学生生成 rollout
        with torch.no_grad():
            student_outputs = self.student.generate(
                x, 
                max_length=self.max_length,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            student_sequences = student_outputs.sequences

        # 2. 从每个教师获取反馈
        total_loss = 0
        
        for teacher, weight in zip(self.teachers, self.teacher_weights):
            # 教师基于学生输出提供分布
            with torch.no_grad():
                teacher_logits = teacher(student_sequences)
                teacher_probs = F.softmax(teacher_logits, dim=-1)

            # 学生在该分布上的损失
            student_logits = self.student(student_sequences)
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            # KL 散度
            kl_div = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(dim=-1)
            loss = kl_div.mean()
            
            total_loss += weight * loss

        return total_loss

    def training_step(self, batch):
        loss = self.compute_mopd_loss(batch)
        loss.backward()
        
        # 可选: 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 工业级应用案例

### 1. Nemotron-Cascade 2 (NVIDIA)

**架构**: 三阶段后训练 + MOPD

```
Stage 1: 基础 SFT
  → Checkpoint A

Stage 2: 领域 RL (Math, Code, General)
  → Checkpoint Math, Checkpoint Code, Checkpoint RLHF

Stage 3: MOPD
  教师: A, Math, Code, RLHF
  学生: Nemotron-Cascade-2-30B-A3B
```

MOPD 作为**稳定化步骤**放置在领域 RL 和最终部署之间。

### 2. MiMo-V2-Flash (小米)

**架构**: 通用 SFT + 领域专家 + MOPD

```
通用 SFT 模型
  ↓
领域专家:
  - Math Specialist
  - Code Specialist  
  - Chat Specialist
  ↓
MOPD 统一融合
  ↓
MiMo-V2-Flash
```

**特点**: 使用从后训练管道各阶段选择的检查点作为教师。

### 3. DeepSeek-V4 (DeepSeek)

**架构**: 两阶段蒸馏

```
Stage 1: 多领域 RL (GRPO)
  - Math RL
  - Code RL
  ↓
Stage 2: On-Policy Distillation
  统一模型整合各域能力
```

### 4. GLM-5 (智谱)

同样采用 OPD 进行多域能力整合。

## 关键优势

| 特性 | 参数平均 | 任务向量 | MoE | MOPD |
|------|---------|---------|-----|------|
| 架构要求 | 相同 | 相同 | 特殊 | 任意 |
| 推理成本 | 1x | 1x | 2x+ | 1x |
| 能力融合 | 差 | 中 | 好 | 好 |
| 训练稳定性 | 低 | 中 | 高 | 高 |
| 灵活性 | 低 | 中 | 中 | 高 |

## 训练技巧

### 1. 温度退火

不同教师使用不同温度：

```python
temperatures = {
    'math_teacher': 0.8,    # 精确任务: 低温度
    'creative_teacher': 1.2, # 创造性任务: 高温度
    'base_teacher': 1.0
}
```

### 2. 重要性采样

根据教师表现动态调整权重：

```python
# 每轮评估教师性能
for k, teacher in enumerate(teachers):
    perf = evaluate(teacher, val_set)
    teacher_weights[k] = softmax(perf / temperature)
```

### 3. 冲突解决

当教师意见冲突时：

```python
def resolve_conflict(teacher_distributions, student_dist):
    """解决教师间的冲突"""
    # 策略: 选择与当前学生最接近的
    similarities = [
        kl_div(student_dist, t_dist) 
        for t_dist in teacher_distributions
    ]
    best_teacher = argmin(similarities)
    return teacher_distributions[best_teacher]
```

## 超参数调优

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `num_teachers` | 教师数量 | 2 ~ 8 |
| `teacher_weights` | 教师权重 | 均匀或基于性能 |
| `temperature` | 采样温度 | 0.7 ~ 1.5 |
| `kl_coef` | KL 约束 | 0.01 ~ 0.1 |
| `max_seq_len` | 最大序列长度 | 512 ~ 2048 |

## 局限与挑战

### 1. 教师质量依赖

学生性能上界受教师质量限制：

```
如果所有教师在某任务上都弱，学生也会弱
```

### 2. 教师间冲突

不同教师可能给出矛盾指导：

```
教师A: "应该详细解释"
教师B: "应该简洁回答"
```

### 3. 计算成本

需要前向传播所有教师：

$$\text{Cost} = O(|S| + \sum_k |T_k|)$$

### 4. 权重调优复杂

寻找最优教师权重组合可能需要大量实验。

## 解决方案

### 自适应权重学习

让模型学习权重：

```python
class AdaptiveMOPD(nn.Module):
    def __init__(self, num_teachers, hidden_dim):
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, num_teachers),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x, teachers):
        # 基于输入动态计算权重
        weights = self.weight_net(encode(x))
        # ... MOPD 计算
```

### 蒸馏教师

使用较小的"教师蒸馏器"替代大教师：

```python
teacher_distillers = {
    k: distill(teacher, small_model)
    for k, teacher in teachers.items()
}
```

## 关键论文

- Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation
- MiMo-V2 Technical Report
- DeepSeek-V4 Technical Report
- A Survey of On-Policy Distillation for Large Language Models (arXiv:2604.00626)

## 资源

- **HuggingFace**: deepseek-ai/DeepSeek-V4-Flash
- **GitHub**: nick7nlp/Awesome-LLM-On-Policy-Distillation
