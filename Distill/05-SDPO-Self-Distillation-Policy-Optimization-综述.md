# SDPO (Self-Distillation Policy Optimization) 综述

## 概述

SDPO (Self-Distillation Policy Optimization) 是一种用于**丰富反馈强化学习**的方法，将tokenized的文本反馈转换为密集的学习信号，无需外部教师模型或显式奖励模型。

## 核心问题

### RLVR 的局限性

传统的可验证奖励强化学习(RLVR)存在严重的**信用分配瓶颈**：

```
传统 RLVR:
- 输入: 问题 x
- 输出: 尝试 y
- 反馈: 标量奖励 r ∈ {0, 1}  # 仅成功/失败
- 问题: 不知道错在哪里，学习信号稀疏
```

### 丰富反馈的浪费

许多可验证环境实际上提供了丰富的文本反馈：
- **运行时错误**: `RuntimeError: division by zero`
- **编译错误**: `SyntaxError: unexpected EOF`
- **评测反馈**: "答案错误: 期望 42，得到 39"
- **评估判断**: "推理不完整，缺少关键步骤"

但这些反馈在传统 RLVR 中被完全忽略。

## SDPO 核心思想

### 关键洞察

**模型有能力通过上下文学习回顾性地识别自己的错误。**

通过将反馈条件化到模型上，可以让模型：
1. 理解为什么失败
2. 预测如何改进
3. 将这些改进蒸馏回策略

### 数学形式化

#### 丰富反馈 RL 框架

在标准 RL 基础上，环境返回 $(r, f)$，其中：
- $r \in \mathbb{R}$: 标量奖励
- $f$: tokenized 文本反馈

#### SDPO 目标

$$\max_{\theta} \mathbb{E}_{(x,r,f) \sim \pi_{\theta}}[r]$$

通过自蒸馏实现，将反馈条件化的预测蒸馏为标准预测：

$$\mathcal{L}_{SDPO} = -\mathbb{E}\left[\log p_{\theta}(y | x) \cdot \underbrace{p_{\theta}(y | x, f)}_{\text{反馈感知的自教师}}\right]$$

## 方法详解

### 三阶段流程

```
Phase 1: Rollout 生成
  输入: 问题 x
  输出: 尝试 y ~ π_θ(· | x)

Phase 2: 反馈获取
  环境执行 y，返回 (r, f)
  f 可以是错误信息、评测结果等

Phase 3: 自蒸馏
  a. 反馈条件模型: π_θ(· | x, f) 预测改进方案
  b. 蒸馏到标准模型: 使 π_θ(· | x) 接近 π_θ(· | x, f)
```

### 训练目标

#### 基础版本

$$\mathcal{L}_{SDPO} = -\sum_{t} \log \pi_{\theta}(y_t | x, y_{<t}) \cdot \pi_{\theta}(y_t | x, f, y_{<t})$$

#### 带优势的版本

$$\mathcal{L}_{SDPO-ADV} = -\sum_{t} A_t \cdot \log \pi_{\theta}(y_t | x, y_{<t})$$

其中优势 $A_t$ 来自反馈条件模型的价值估计。

### 反馈编码策略

#### 策略 1: 前缀编码

```
[Feedback: RuntimeError: division by zero]
[Question: Calculate x/y when x=5, y=0]
[Solution: ...]
```

#### 策略 2: 上下文编码

```
[Question: ...]
[Attempt: ...]
[Feedback: ...]
[Correction: ...]
```

#### 策略 3: 结构化编码

使用特殊 token 标记反馈：

```
<question>...</question>
<attempt>...</attempt>
<feedback type="runtime_error">...</feedback>
<correction>...</correction>
```

## 伪代码实现

```python
class SDPOTrainer:
    def __init__(self, model, env):
        self.model = model
        self.env = env  # 可验证环境

    def collect_rollouts(self, questions):
        """收集带有反馈的 rollout"""
        rollouts = []
        
        for x in questions:
            # 1. 生成尝试
            y = self.model.generate(x, max_length=512)
            
            # 2. 执行并获取反馈
            r, f = self.env.execute(y)
            
            rollouts.append({
                'question': x,
                'attempt': y,
                'reward': r,
                'feedback': f
            })
        
        return rollouts

    def compute_sdpo_loss(self, rollouts):
        """计算 SDPO 损失"""
        total_loss = 0
        
        for rollout in rollouts:
            x = rollout['question']
            y = rollout['attempt']
            f = rollout['feedback']
            
            # 3a. 反馈条件模型 (无梯度)
            with torch.no_grad():
                feedback_input = self.format_feedback(x, y, f)
                conditioned_logits = self.model(feedback_input)
                
                # 提取反馈后的预测分布
                conditioned_probs = F.softmax(
                    conditioned_logits[:, len(feedback_input):, :],
                    dim=-1
                )

            # 3b. 标准模型
            standard_logits = self.model(x)
            log_probs = F.log_softmax(standard_logits, dim=-1)

            # 3c. 蒸馏损失
            loss = -(conditioned_probs * log_probs).sum(dim=-1).mean()
            
            # 可选: 加权奖励
            weight = rollout['reward'] if rollout['reward'] > 0 else 0.1
            total_loss += weight * loss

        return total_loss / len(rollouts)

    def training_step(self, questions):
        # 收集 rollout
        rollouts = self.collect_rollouts(questions)
        
        # 计算损失
        loss = self.compute_sdpo_loss(rollouts)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## 关键优势

### 1. 密集学习信号

相比稀疏的标量奖励，SDPO 提供了token级别的梯度：

| 方法 | 反馈粒度 | 学习信号密度 |
|------|---------|-------------|
| PPO | 序列级别 | 稀疏 |
| DPO | 偏好对级别 | 中等 |
| **SDPO** | **Token级别** | **密集** |

### 2. 无需外部组件

- 不需要奖励模型
- 不需要价值网络
- 不需要参考模型 (可选)

### 3. 样本效率

实验表明 SDPO 在以下基准上样本效率提升：

- **LiveCodeBench v6**: 3x 更少尝试达到同等发现概率
- **科学推理**: 2x 样本效率提升
- **工具使用**: 1.5x 最终准确率提升

## 无丰富反馈的扩展

### 隐式反馈

即使没有显式文本反馈，SDPO 仍可利用：

**成功 rollout 作为失败尝试的隐式反馈**：

```python
if rollout['reward'] == 0:
    # 找到成功的类似尝试作为"应该怎么做"的反馈
    successful = find_similar_successful(rollouts, rollout)
    implicit_feedback = f"正确解法: {successful['attempt']}"
```

### 自我批评

让模型自己生成反馈：

```python
# 先生成尝试
critic_input = f"{x}\n{attempt}\n\n分析这个解法的错误:"
feedback = model.generate(critic_input)
```

## 实验结果

### LiveCodeBench v6

| 方法 | Pass@1 | 平均尝试次数 |
|------|--------|-------------|
| PPO | 23.4% | 50 |
| DPO | 25.1% | N/A |
| RFT | 27.3% | 30 |
| **SDPO** | **31.2%** | **16** |

### 科学推理 (GSM8K)

| 方法 | 准确率 | 训练步数 |
|------|-------|---------|
| SFT | 72.1% | 10K |
| PPO | 76.3% | 20K |
| **SDPO** | **79.8%** | **12K** |

## 超参数调优

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `feedback_weight` | 反馈在损失中的权重 | 0.5 ~ 1.0 |
| `temperature` | 采样温度 | 0.8 ~ 1.2 |
| `max_feedback_length` | 反馈最大长度 | 100 ~ 500 tokens |
| `num_rollouts` | 每问题 rollout 数 | 4 ~ 16 |
| `kl_coef` | KL 约束系数 (可选) | 0.01 ~ 0.1 |

## 应用场景

### 1. 代码生成

利用编译器/运行时的错误信息：

```python
# 反馈示例
"""
File "solution.py", line 5
    if x = 1:
         ^
SyntaxError: invalid syntax
"""
```

### 2. 数学推理

利用逐步验证的反馈：

```python
# 反馈示例
"""
步骤 2 错误: 分配律应用错误
正确: 2(x + 3) = 2x + 6
你的: 2(x + 3) = 2x + 3
"""
```

### 3. 工具使用

利用 API 调用结果：

```python
# 反馈示例
"""
API 调用失败: search(query="...")
错误: 查询参数不能为空字符串
"""
```

## 与相关方法的对比

| 方法 | 反馈类型 | 外部依赖 | 样本效率 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| PPO | 标量 | 高 | 中 | 高 |
| DPO | 偏好 | 中 | 中 | 中 |
| RFT | 结果 | 低 | 中 | 低 |
| **SDPO** | **丰富文本** | **无** | **高** | **中** |

## 局限与未来方向

### 当前局限

1. 反馈质量依赖环境设计
2. 长反馈可能超过上下文限制
3. 某些领域缺乏结构化反馈

### 研究方向

- **自动反馈生成**: 让模型学习生成有用的反馈
- **多轮反馈**: 迭代式改进
- **跨领域迁移**: 将反馈学习迁移到新领域

## 关键论文

**Reinforcement Learning via Self-Distillation**
- arXiv: 2601.20802
- 作者: Jonas Hübotter et al. (ETH Zurich)
- 发表时间: 2026年1月

## 资源

- **项目网站**: self-distillation.github.io
- **HuggingFace TRL**: `trl.experimental.sdpo_trainer`
