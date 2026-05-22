# SDFT (Self-Distillation Fine-Tuning) 综述

## 概述

SDFT (Self-Distillation Fine-Tuning) 是一种**无需外部教师模型**的在线学习方法，使模型能够从演示中进行**策略学习(On-Policy Learning)**。它解决了传统 SFT 中的灾难性遗忘问题，同时保持了简单性。

## 核心问题

### 持续学习的挑战

持续学习(Continual Learning)需要模型在不降低现有能力的前提下获取新技能。现有方法面临两难：

1. **在线强化学习**: 可以减少遗忘，但需要明确的奖励函数
2. **监督微调(SFT)**: 从专家演示学习，但本质是离线(off-policy)的，容易导致遗忘

### 传统 SFT 的局限

```
传统 SFT:
- 输入: 完美演示 (x, y_demo)
- 目标: p_θ(y_demo | x)
- 问题: 模型在完美前缀上训练，推理时却要自己生成
```

## SDFT 核心思想

### 关键洞察

利用**上下文学习(In-Context Learning)**能力，让模型自己作为自己的教师：

1. **条件化**: 先将演示作为上下文输入模型
2. **自蒸馏**: 让模型基于演示条件生成预测
3. **策略学习**: 生成的预测成为训练信号

### 数学形式化

给定输入 $x$ 和演示 $d$，SDFT 训练模型满足：

$$p_\theta(y | x, d) \approx p_\theta(y | x \oplus d)$$

其中 $\oplus$ 表示拼接操作。

**训练目标**：

$$\mathcal{L}_{SDFT} = -\mathbb{E}_{(x,d) \sim D}\left[\log p_\theta(y | x, d)\right]$$

但这里的 $y$ 是通过**演示条件模型**生成的，而非人工标注。

## 方法详解

### 三阶段流程

```
Phase 1: 演示条件化
  输入: x, d
  输出: 条件化表示 h = Encode(x ⊕ d)

Phase 2: 自生成
  输入: h
  输出: ŷ ~ p_θ(· | x, d)

Phase 3: 蒸馏
  目标: 使 p_θ(· | x) 接近 p_θ(· | x, d)
```

### 实现变体

#### 变体 1: 硬蒸馏 (Hard Distillation)

直接从条件模型采样作为目标：

```python
with torch.no_grad():
    # 演示条件模型生成
    conditioned_logits = model(input_ids, demo_ids)
    target_tokens = sample(conditioned_logits)

# 标准模型学习这些 token
logits = model(input_ids)
loss = CrossEntropy(logits, target_tokens)
```

#### 变体 2: 软蒸馏 (Soft Distillation)

匹配概率分布：

$$\mathcal{L} = D_{KL}(p_\theta(\cdot | x) \| p_\theta(\cdot | x, d))$$

#### 变体 3: 混合蒸馏 (Hybrid)

结合两者：

$$\mathcal{L} = \lambda_{hard} \mathcal{L}_{hard} + \lambda_{soft} \mathcal{L}_{soft}$$

## 核心优势

### 1. 避免灾难性遗忘

由于训练信号来自模型自身（只是条件不同），不会引入与预训练分布不一致的新信号。

### 2. 无需外部教师

完全自包含的方法，不依赖：
- 更大的教师模型
- 昂贵的标注数据
- 人工奖励函数

### 3. 简单高效

实现复杂度与标准 SFT 相当，仅需：
- 修改数据格式（添加演示）
- 两次前向传播（条件化 + 标准）

## 伪代码实现

```python
class SDFTTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_loss(self, batch):
        x = batch['input_ids']      # 原始输入
        d = batch['demonstration']   # 演示/示例

        # Step 1: 演示条件模型 (无梯度)
        with torch.no_grad():
            demo_input = torch.cat([d, x], dim=1)
            conditioned_outputs = self.model(demo_input)
            
            # 提取演示后的预测
            # 假设 d 的长度为 L_d
            conditioned_logits = conditioned_outputs.logits[:, L_d:, :]
            
            # 采样或取 argmax
            if self.use_soft_targets:
                targets = F.softmax(conditioned_logits / self.temperature, dim=-1)
            else:
                targets = torch.argmax(conditioned_logits, dim=-1)

        # Step 2: 标准模型
        outputs = self.model(x)
        logits = outputs.logits

        # Step 3: 计算损失
        if self.use_soft_targets:
            # KL 散度
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(
                log_probs.view(-1, vocab_size),
                targets.view(-1, vocab_size),
                reduction='batchmean'
            )
        else:
            # 交叉熵
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

        return loss

    def training_step(self, batch):
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

## 实验结果

### 技能学习任务

| 方法 | 新任务准确率 | 旧任务保持率 | 综合得分 |
|------|------------|------------|----------|
| SFT | 78% | 45% | 61.5 |
| SFT + Replay | 75% | 72% | 73.5 |
| **SDFT** | **82%** | **85%** | **83.5** |

### 知识获取任务

- **事实性知识**: SDFT 在知识获取上比 SFT 高 15-20%
- **推理能力**: 保持原有推理能力，SFT 下降 30%

## 超参数调优

### 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `temperature` | 采样温度 | 1.0 ~ 2.0 |
| `demo_length` | 演示长度 | 50 ~ 200 tokens |
| `alpha` | 硬/软混合比例 | 0.5 (软蒸馏为主) |
| `learning_rate` | 学习率 | 2e-5 ~ 5e-5 |

### 演示选择策略

1. **任务示例**: 提供 1-3 个完成示例
2. **质量筛选**: 使用困惑度筛选高质量演示
3. **多样性**: 覆盖不同场景 edge cases

## 与相关方法的对比

| 方法 | 教师来源 | 数据需求 | 遗忘风险 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| SFT | 人工标注 | 高 | 高 | 低 |
| KD | 外部大模型 | 中 | 中 | 中 |
| **SDFT** | **自身** | **低** | **低** | **低** |
| RL | 奖励模型 | 低 | 低 | 高 |

## 应用场景

### 1. 多任务持续学习

在不遗忘旧任务的情况下学习新任务：

```
Task 1: 摘要 → Task 2: 翻译 → Task 3: 问答
SDFT: 每个任务学习后，前两个任务性能几乎不下降
```

### 2. 领域适应

快速适应新领域而不损失通用能力：

```
通用模型 → 医学领域 → 法律领域 → 金融领域
```

### 3. 个性化微调

基于用户历史进行个性化：

```
用户历史交互作为演示 → 生成个性化回复
```

## 扩展变体

### SDFT-C (Contrastive)

引入负样本对比：

$$\mathcal{L} = \log \frac{\exp(s(x, d^+))}{\exp(s(x, d^+)) + \exp(s(x, d^-))}$$

### SDFT-M (Multi-step)

多步自举：

```
Step 1: θ_0 → generate d_1
Step 2: θ_1 trained on d_1 → generate d_2
Step 3: θ_2 trained on d_2
...
```

## 限制与未来方向

### 当前限制

1. 演示质量依赖模型自身能力
2. 对长序列建模仍有挑战
3. 计算成本约为 SFT 的 2 倍

### 研究方向

- 自适应演示长度选择
- 多模态扩展
- 理论分析（遗忘边界）

## 关键论文

**Self-Distillation Enables Continual Learning**
- arXiv: 2601.19897
- 作者: Idan Shenfeld et al. (MIT)
- 发表时间: 2026年1月

## 资源

- **HuggingFace TRL**: `trl.experimental.sdft_trainer`
- **GitHub**: self-distillation.github.io