# 持续学习框架 (Continual Learning) 综述

## 概述

持续学习(Continual Learning / Lifelong Learning)旨在使模型能够**持续地从新数据中学习**，同时**不遗忘**已学到的知识。这对于大语言模型的实际部署至关重要。

## 核心问题

### 灾难性遗忘 (Catastrophic Forgetting)

```
训练任务A → 准确率 90%
训练任务B → 准确率 85%
但任务A → 准确率下降到 30% !
```

神经网络在学习新任务时倾向于覆盖旧任务的权重，导致旧任务性能急剧下降。

### 持续学习的挑战

1. **稳定性-可塑性困境**:
   - 稳定性: 保持旧知识
   - 可塑性: 学习新知识
   - 两者往往矛盾

2. **数据隐私**:
   - 不能存储所有历史数据
   - 某些数据有使用期限

3. **任务边界模糊**:
   - 现实世界的任务没有清晰边界
   - 任务分布不断变化

## 核心方法分类

### 1. 基于正则化的方法 (Regularization-based)

#### 1.1 EWC (Elastic Weight Consolidation)

**核心思想**: 保护对旧任务重要的参数。

$$\mathcal{L}(\theta) = \mathcal{L}_{new}(\theta) + \lambda \sum_i F_i (\theta_i - \theta_{i,old}^*)^2$$

其中：
- $F_i$: Fisher 信息矩阵对角线，表示参数重要性
- $\lambda$: 正则化强度
- $\theta_{i,old}^*$: 旧任务最优参数

**实现**:

```python
class EWC:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}  # Fisher 信息
        self.optimal_params = {}  # 旧任务参数
    
    def compute_fisher(self, dataloader):
        """计算 Fisher 信息"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) 
                  for n, p in self.model.named_parameters()}
        
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch)
            loss = -F.log_softmax(output, dim=1).mean()
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) / len(dataloader)
        
        return fisher
    
    def penalty(self):
        """计算 EWC 惩罚项"""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * 
                        (p - self.optimal_params[n]).pow(2)).sum()
        return self.lambda_ewc * loss
    
    def update(self, new_dataloader):
        """学习新任务"""
        # 先计算当前任务的 Fisher
        self.fisher_dict = self.compute_fisher(new_dataloader)
        self.optimal_params = {n: p.clone() 
                               for n, p in self.model.named_parameters()}
```

#### 1.2 L2 正则化

简化版本：

$$\mathcal{L}(\theta) = \mathcal{L}_{new}(\theta) + \lambda \|\theta - \theta_{old}\|^2$$

```python
def l2_penalty(model, old_params, lambda_l2=1.0):
    penalty = 0
    for (name, param), (_, old_param) in zip(
        model.named_parameters(), old_params.items()
    ):
        penalty += ((param - old_param) ** 2).sum()
    return lambda_l2 * penalty
```

### 2. 基于记忆的方法 (Replay-based)

#### 2.1 经验回放 (Experience Replay)

存储旧任务的样本，与新数据混合训练：

```python
class ExperienceReplay:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, examples):
        self.buffer.extend(examples)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, 
                           min(batch_size, len(self.buffer)))
    
    def get_mixed_batch(self, new_batch, replay_ratio=0.3):
        replay_size = int(len(new_batch) * replay_ratio)
        replay_batch = self.sample(replay_size)
        return new_batch + replay_batch
```

#### 2.2 生成回放 (Generative Replay)

使用生成模型合成旧任务数据：

```python
class GenerativeReplay:
    def __init__(self, generator_model):
        self.generator = generator_model
        self.task_embeddings = {}
    
    def generate_old_data(self, task_id, num_samples):
        task_emb = self.task_embeddings[task_id]
        return self.generator.sample(
            num_samples, 
            condition=task_emb
        )
```

### 3. 基于架构的方法 (Architecture-based)

#### 3.1 渐进网络 (Progressive Networks)

为新任务添加新列：

```
Task 1: [Input] → [H1_1] → [H2_1] → [Output]
Task 2: [Input] → [H1_2] → [H2_2] → [Output]
                      ↗      ↗
Task 3: [Input] → [H1_3] → [H2_3] → [Output]
```

```python
class ProgressiveLayer(nn.Module):
    def __init__(self, input_size, output_size, num_tasks):
        super().__init__()
        self.columns = nn.ModuleList([
            nn.Linear(input_size, output_size)
            for _ in range(num_tasks)
        ])
        self.adapters = nn.ModuleList([
            nn.Linear(output_size, output_size)
            for _ in range(num_tasks - 1)
        ])
    
    def forward(self, x, task_id):
        h = self.columns[task_id](x)
        # 连接之前任务的特征
        for prev_task in range(task_id):
            prev_h = self.columns[prev_task](x)
            h = h + self.adapters[prev_task](prev_h)
        return F.relu(h)
```

#### 3.2 任务特定参数 (Task-Specific Parameters)

使用适配器(Adapter)或 LoRA：

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.up = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return residual + x

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_tasks):
        super().__init__()
        self.base_model = base_model
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 每个任务一个适配器
        self.task_adapters = nn.ModuleList([
            Adapter(768, 64) for _ in range(num_tasks)
        ])
    
    def forward(self, x, task_id):
        x = self.base_model(x)
        x = self.task_adapters[task_id](x)
        return x
```

### 4. 基于蒸馏的方法 (Distillation-based)

#### 4.1 学习 without Forgetting (LwF)

使用旧模型作为教师：

```python
class LearningWithoutForgetting:
    def __init__(self, model, lambda_distill=1.0):
        self.model = model
        self.old_model = None
        self.lambda_distill = lambda_distill
    
    def before_new_task(self):
        """保存旧模型"""
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
    
    def compute_distill_loss(self, inputs):
        """蒸馏损失"""
        with torch.no_grad():
            old_outputs = self.old_model(inputs)
        
        new_outputs = self.model(inputs)
        
        # KL 散度
        distill_loss = F.kl_div(
            F.log_softmax(new_outputs, dim=1),
            F.softmax(old_outputs, dim=1),
            reduction='batchmean'
        )
        
        return self.lambda_distill * distill_loss
    
    def compute_loss(self, inputs, labels):
        """总损失"""
        # 新任务损失
        new_loss = F.cross_entropy(self.model(inputs), labels)
        
        # 蒸馏损失
        distill_loss = self.compute_distill_loss(inputs)
        
        return new_loss + distill_loss
```

#### 4.2 SDFT (已在单独文档中详述)

### 5. 基于元学习的方法 (Meta-learning)

#### 5.1 OML (Online Meta-Learning)

```python
class OnlineMetaLearning:
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01):
        self.model = model
        self.meta_optimizer = Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
    
    def meta_train_step(self, task_batch):
        """元训练步骤"""
        meta_loss = 0
        
        for support, query in task_batch:
            # 内循环：适应特定任务
            fast_weights = self.inner_loop(support)
            
            # 外循环：在查询集上评估
            query_loss = self.forward_with_weights(query, fast_weights)
            meta_loss += query_loss
        
        # 更新元参数
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def inner_loop(self, support_set):
        """快速适应"""
        fast_weights = list(self.model.parameters())
        
        for _ in range(self.inner_steps):
            loss = self.compute_loss(support_set, fast_weights)
            grads = torch.autograd.grad(loss, fast_weights)
            fast_weights = [w - self.inner_lr * g 
                           for w, g in zip(fast_weights, grads)]
        
        return fast_weights
```

### 6. 最新进展

#### 6.1 TFGN (Task-Free Gating Networks)

无需任务标签的持续学习：

```python
class TFGN(nn.Module):
    """无任务标签的持续学习"""
    def __init__(self, hidden_size, num_experts=8):
        super().__init__()
        self.gating = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 动态选择专家
        gates = F.softmax(self.gating(x.mean(dim=1)), dim=-1)
        
        output = sum(
            gate.unsqueeze(1) * expert(x)
            for gate, expert in zip(gates.unbind(-1), self.experts)
        )
        return output
```

#### 6.2 Fast-Slow Learning

使用上下文作为"快速权重"：

```python
class FastSlowLearning:
    def __init__(self, model):
        self.model = model
        # 慢速权重: 模型参数 (不频繁更新)
        # 快速权重: 上下文/提示 (每次推理更新)
    
    def forward(self, x, context):
        # 快速适应：基于上下文修改表示
        adapted = self.fast_adapt(x, context)
        # 慢速处理：使用固定参数
        output = self.model(adapted)
        return output
    
    def fast_adapt(self, x, context):
        """基于上下文的快速适应"""
        # 使用注意力机制或提示调优
        return x + self.compute_contextual_shift(context)
```

## 方法对比

| 方法 | 计算开销 | 存储开销 | 灵活性 | 效果 | 适用场景 |
|------|---------|---------|-------|------|---------|
| EWC | 低 | 低 (Fisher矩阵) | 中 | 中 | 任务数少 |
| 经验回放 | 低 | 高 (存储数据) | 高 | 好 | 数据可存储 |
| 生成回放 | 中 | 中 (生成器) | 中 | 中 | 隐私敏感 |
| 适配器 | 低 | 低 (小参数) | 高 | 好 | 多任务 |
| LwF | 低 | 低 (旧模型) | 中 | 中 | 模型可复制 |
| SDFT | 低 | 低 | 高 | 好 | 演示学习 |
| TFGN | 中 | 中 | 高 | 好 | 无任务边界 |

## 实践建议

### 1. 方法选择决策树

```
数据可以存储?
├── 是 → 经验回放
└── 否 → 任务标签已知?
    ├── 是 → 适配器/LoRA
    └── 否 → TFGN 或 SDFT
```

### 2. 超参数调优

```python
# EWC
lambda_ewc = 1000  # 任务差异大时增大

# 经验回放
buffer_size = min(1000, 0.1 * total_data)  # 数据量的10%
replay_ratio = 0.3  # 回放比例

# 适配器
adapter_size = 64  # 通常为 hidden_size 的 1/10
```

### 3. 评估指标

```python
def compute_forgetting(accuracies):
    """计算遗忘程度"""
    # accuracies: list of (task_id, acc_after_learning_task_k)
    forgetting = []
    for k in range(len(accuracies)):
        for j in range(k):
            # 学习任务k后，任务j的性能下降
            f = accuracies[j][j] - accuracies[k][j]
            forgetting.append(f)
    return np.mean(forgetting)

def compute_average_accuracy(accuracies):
    """平均准确率"""
    return np.mean([acc[-1] for acc in accuracies])
```

## 关键论文

1. **EWC**: Overcoming catastrophic forgetting in neural networks (2017)
2. **Progressive Networks**: Progressive neural networks (2016)
3. **LwF**: Learning without forgetting (2016)
4. **OML**: Online meta-learning (2019)
5. **TFGN**: Task-Free Continual Learning (2023)
6. **SDFT**: Self-Distillation Enables Continual Learning (2026)
7. **Fast-Slow Learning**: Fast-Slow Learning: Teaching LLMs to Adapt Without Forgetting

## 资源

- **库**: avalanche-lib.org (Avalanche 持续学习库)
- **基准**: CORe50, Split CIFAR, Split MNIST
- **实现**: PyTorch, HuggingFace Transformers

## 总结

持续学习是大模型实际部署的关键能力。从传统的正则化方法到现代的自蒸馏和元学习方法，各种技术各有优劣。在实践中，**适配器(LoRA) + 经验回放** 或 **SDFT** 是目前最实用的组合方案。
