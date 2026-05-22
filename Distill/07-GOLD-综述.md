# GOLD (Generative On-policy Learning with Distillation) 综述

## 概述

GOLD 是一种结合**生成式学习与在线策略蒸馏**的方法，专注于通过学生自身生成的数据进行高效的知识迁移，同时保持生成质量和多样性。

## 核心问题

### 生成任务中的蒸馏挑战

传统蒸馏方法在生成任务上表现不佳：

1. **序列级损失**: 难以处理长序列的信用分配
2. **暴露偏差**: 训练和推理分布不一致
3. **多样性损失**: 学生过于模仿教师，丧失创造性
4. **模式坍塌**: 只能生成少数几种输出

## GOLD 核心思想

### 关键洞察

**生成任务需要生成式训练目标。**

GOLD 通过以下方式解决上述问题：
1. 在学生自己的生成样本上训练
2. 使用分布匹配而非点估计
3. 鼓励多样性而非单一模仿

### 数学形式化

#### 生成式蒸馏目标

$$\mathcal{L}_{GOLD} = \mathbb{E}_{y \sim p_{student}}[-\log p_{teacher}(y | x)]$$

与传统 KL 散度的区别：
- **KL(student || teacher)**: 在教师分布上计算
- **GOLD**: 在学生分布上计算

这类似于强化学习中的 on-policy 更新。

#### 与 RL 的联系

GOLD 可以视为一种特殊的策略梯度方法：

$$\nabla_\theta \mathcal{L}_{GOLD} = \mathbb{E}_{y \sim p_\theta}[\nabla_\theta \log p_\theta(y|x) \cdot (-\log p_{teacher}(y|x))]$$

其中 $-\log p_{teacher}(y|x)$ 充当**优势函数**。

## 方法详解

### 训练流程

```
Algorithm GOLD:
1. Initialize student model θ
2. For each training step:
   a. Sample prompts {x_i}
   b. Generate student outputs:
      y_i ~ p_θ(· | x_i)
   c. Score with teacher:
      r_i = -log p_teacher(y_i | x_i)
   d. Update student:
      θ ← θ - α ∇_θ Σ r_i · log p_θ(y_i | x_i)
3. Return trained student
```

### 关键技术

#### 1. 重要性采样校正

由于在学生分布上采样，使用重要性采样：

$$\mathcal{L} = \mathbb{E}_{y \sim q}\left[\frac{p_\theta(y|x)}{q(y|x)} \cdot (-\log p_{teacher}(y|x))\right]$$

其中 $q$ 是采样分布（通常带 temperature 的学生分布）。

#### 2. 基线减除

减少方差：

$$\mathcal{L} = \mathbb{E}[(r - b) \cdot \log p_\theta(y|x)]$$

基线 $b$ 可以是：
- 平均奖励
- 学习的状态价值函数
- 滚动平均

#### 3. 多样性奖励

防止模式坍塌：

$$r_{total} = r_{teacher} + \lambda \cdot r_{diversity}$$

多样性奖励可以是：
- 序列间距离
- 熵奖励
- 新颖性奖励

## 伪代码实现

```python
class GOLDTrainer:
    def __init__(self, student, teacher, temperature=1.0):
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.baseline = 0
        self.baseline_decay = 0.9

    def compute_gold_loss(self, batch):
        prompts = batch['prompts']
        batch_size = len(prompts)
        
        # 1. 学生生成
        with torch.no_grad():
            student_outputs = self.student.generate(
                prompts,
                max_length=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_sequences = student_outputs.sequences

        # 2. 教师评分
        with torch.no_grad():
            teacher_logits = self.teacher(generated_sequences)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            
            # 提取生成的 token 的概率
            generated_tokens = generated_sequences[:, 1:]  # 去掉 prompt
            token_log_probs = torch.gather(
                teacher_log_probs[:, :-1, :],
                -1,
                generated_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            # 序列级别的负对数似然作为奖励
            rewards = -token_log_probs.sum(dim=-1)

        # 3. 更新基线
        self.baseline = (self.baseline_decay * self.baseline + 
                        (1 - self.baseline_decay) * rewards.mean().item())

        # 4. 学生在这些序列上的对数概率
        student_logits = self.student(generated_sequences)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        student_token_log_probs = torch.gather(
            student_log_probs[:, :-1, :],
            -1,
            generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # 5. 策略梯度损失
        advantages = rewards - self.baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        loss = -(advantages.unsqueeze(-1) * student_token_log_probs).mean()

        return loss

    def training_step(self, batch):
        loss = self.compute_gold_loss(batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 与其他方法的对比

### 与标准 KD 对比

```python
# 标准 KD (Forward KL)
loss = KL(p_teacher || p_student)  # 在教师分布上

# GOLD (类似 Reverse KL)
loss = -E_{y~student}[log p_teacher(y)]  # 在学生分布上
```

### 与 PPO 对比

| 特性 | PPO | GOLD |
|------|-----|------|
| 奖励来源 | 奖励模型 | 教师对数概率 |
| 价值函数 | 需要 | 可选 (基线) |
| 采样策略 | On-policy | On-policy |
| KL 约束 | 显式 | 隐式 (temperature) |
| 实现复杂度 | 高 | 中 |

### 与 GKD 对比

| 特性 | GKD | GOLD |
|------|-----|------|
| 采样 | 混合学生/真实 | 纯学生 |
| 目标 | KL 散度 | 负对数似然 |
| 灵活性 | 中等 | 高 |
| 多样性保持 | 中 | 好 |

## 实验结果

### 文本生成任务

| 方法 | Perplexity | Diversity (Self-BLEU) | 人类偏好 |
|------|-----------|---------------------|---------|
| SFT | 23.4 | 0.72 | 3.2/5 |
| Standard KD | 21.8 | 0.65 | 3.4/5 |
| GKD | 20.5 | 0.68 | 3.6/5 |
| **GOLD** | **19.8** | **0.74** | **4.1/5** |

### 机器翻译

| 方法 | BLEU | 多样性 |
|------|------|-------|
| Baseline | 28.3 | 0.45 |
| SeqKD | 29.1 | 0.42 |
| **GOLD** | **30.5** | **0.51** |

### 摘要生成

| 方法 | ROUGE-L | 重复率 |
|------|---------|-------|
| SFT | 38.2 | 12% |
| GKD | 39.5 | 10% |
| **GOLD** | **40.8** | **7%** |

## 关键技术细节

### 1. Temperature 退火

```python
def get_temperature(epoch, total_epochs):
    """从高温度开始，逐渐降低"""
    return 1.5 - 0.8 * (epoch / total_epochs)
```

高温度 → 探索，低温度 → 利用

### 2. 长度归一化

处理不同长度序列：

```python
# 按长度归一化奖励
rewards = -token_log_probs.sum(dim=-1) / lengths
```

### 3. Top-k 截断

只考虑头部概率：

```python
def topk_teacher_probs(logits, k=50):
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    return topk_probs, topk_indices
```

## 超参数调优

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `temperature` | 采样温度 | 1.0 ~ 2.0 |
| `baseline_decay` | 基线平滑系数 | 0.9 ~ 0.99 |
| `clip_norm` | 梯度裁剪 | 1.0 ~ 5.0 |
| `max_length` | 最大生成长度 | 任务相关 |
| `num_samples` | 每 prompt 采样数 | 4 ~ 16 |

## 应用场景

### 1. 创意写作

保持多样性和创造力：

```
输入: "写一个关于未来城市的故事开头"
输出: 多样化的创意开头 (而非单一的"标准"开头)
```

### 2. 对话系统

避免重复的回复：

```python
# GOLD 鼓励多样化的回复
turn_1: "你好！今天过得怎么样？"
turn_2: "有什么我可以帮你的吗？"  # 不同句式，非模板
```

### 3. 代码生成

生成多种实现方式：

```python
# 同一功能的不同实现
# 实现1: 递归
# 实现2: 迭代
# 实现3: 函数式
```

### 4. 数据增强

用学生生成合成数据：

```python
# GOLD 训练的学生可以生成高质量训练数据
synthetic_data = student.generate(prompts, num_samples=1000)
```

## 扩展变体

### GOLD++

结合多个目标：

$$\mathcal{L}_{GOLD++} = \mathcal{L}_{GOLD} + \lambda_1 \mathcal{L}_{MLE} + \lambda_2 \mathcal{L}_{reg}$$

其中：
- $\mathcal{L}_{MLE}$: 最大似然 (保持基础能力)
- $\mathcal{L}_{reg}$: 正则化 (如熵正则)

### Multi-GOLD

多教师 GOLD：

```python
def multi_gold_loss(student, teachers, weights):
    loss = 0
    for teacher, weight in zip(teachers, weights):
        loss += weight * gold_loss(student, teacher)
    return loss
```

### GOLD with Constraints

加入显式约束：

```python
def constrained_gold_loss(student, teacher, constraint_fn):
    base_loss = gold_loss(student, teacher)
    constraint = constraint_fn(student)
    return base_loss + lambda_constraint * constraint
```

## 局限与未来方向

### 当前局限

1. **高方差**: 采样导致梯度方差大
2. **计算成本**: 需要在线生成
3. **温度敏感**: 对 temperature 选择敏感

### 改进方向

1. **方差减少**: 更好的基线估计、控制变量
2. **高效采样**: 使用缓存、近似采样
3. **自适应温度**: 学习最优温度

## 与其他 OPD 方法的关系

```
OPD 家族
├── GKD: 混合数据 KL 散度
├── MiniLLM: Reverse KL 优化
├── DistiLLM: Skewed KL 稳定
├── GOLD: 生成式策略梯度
└── 组合: GKD+GOLD, MiniLLM+GOLD
```

## 关键论文

- GOLD: Generative On-policy Learning with Distillation
- Policy Gradient Methods for Natural Language Processing
- Token-Level Direct Optimization

## 资源

- **HuggingFace TRL**: `trl.experimental.gold_trainer`
- **实现参考**: 参见 Awesome-LLM-On-Policy-Distillation
