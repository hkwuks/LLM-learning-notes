# ORBIT (On-policy Exploration-Exploitation) 综述

## 概述

ORBIT 是一种用于**可控多预算推理**的在线探索-利用方法，通过智能地分配计算资源来优化推理过程，同时通过在线学习不断改进策略。

## 核心问题

### 推理中的预算约束

大型语言模型在实际部署中面临计算预算限制：

```
场景1: API 调用成本限制
场景2: 实时响应时间要求  
场景3: 边缘设备资源受限
```

**关键问题**: 如何在不同预算约束下做出最优推理决策？

### 传统方法的局限

1. **固定计算**: 无论问题难度都使用相同计算量
2. **静态策略**: 不能根据反馈调整计算分配
3. **贪婪推理**: 缺乏探索，可能错过更好方案

## ORBIT 核心思想

### 关键洞察

**不同问题需要不同的计算量，且最优策略可以通过在线交互学习。**

ORBIT 将推理视为一个序列决策过程：

```
Step 1: 观察问题
Step 2: 决定计算预算
Step 3: 执行推理
Step 4: 获取结果质量反馈
Step 5: 更新策略 (在线学习)
```

### 数学形式化

#### 多预算推理框架

给定问题 $x$ 和可用预算集合 $\mathcal{B} = \{b_1, b_2, ..., b_K\}$，目标是：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, b \sim \pi(\cdot|x)}[Q(x, b)]$$

其中：
- $\pi(b|x)$: 预算选择策略
- $Q(x, b)$: 在预算 $b$ 下解决问题的质量

#### 在线学习更新

使用 Bandit 风格的更新：

$$\pi_{t+1}(b|x) = \pi_t(b|x) + \alpha (Q(x, b) - \bar{Q}) \nabla \log \pi_t(b|x)$$

### 探索-利用权衡

ORBIT 采用 Upper Confidence Bound (UCB) 风格的探索：

$$b^* = \arg\max_b \left[ \hat{Q}(x, b) + c \sqrt{\frac{\log t}{N(x, b)}} \right]$$

其中：
- $\hat{Q}$: 质量估计
- $N(x, b)$: 选择次数 (探索奖励)
- $c$: 探索系数

## 方法详解

### 系统架构

```
┌─────────────────────────────────────────┐
│            ORBIT Framework              │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────┐    ┌─────────────┐        │
│  │  Problem │───▶│ Budget       │        │
│  │   (x)   │    │ Selector     │        │
│  └─────────┘    └──────┬──────┘        │
│                        │               │
│                        ▼               │
│                 ┌─────────────┐        │
│                 │  Reasoning  │        │
│                 │   Engine    │        │
│                 └──────┬──────┘        │
│                        │               │
│                        ▼               │
│  ┌─────────┐    ┌─────────────┐        │
│  │  Online │◀───│   Result    │        │
│  │ Update  │    │  Quality (Q)│        │
│  └─────────┘    └─────────────┘        │
│                                         │
└─────────────────────────────────────────┘
```

### 预算类型

#### 1. Token 预算

```python
budgets = {
    'low': 256,      # 快速响应
    'medium': 512,   # 标准推理
    'high': 1024,    # 复杂问题
    'unlimited': float('inf')  # 关键任务
}
```

#### 2. 推理步数预算

```python
budgets = {
    'greedy': 1,           # 单次推理
    'beam_search': 5,      # Beam search
    'mcts_light': 16,      # 轻量 MCTS
    'mcts_full': 64        # 完整 MCTS
}
```

#### 3. 计算资源预算

```python
budgets = {
    'mobile': {'flops': 1e9, 'memory': '2GB'},
    'desktop': {'flops': 1e11, 'memory': '16GB'},
    'server': {'flops': 1e13, 'memory': '80GB'}
}
```

### 策略网络

#### 输入编码

```python
class BudgetSelector(nn.Module):
    def __init__(self, hidden_dim, num_budgets):
        self.encoder = TransformerEncoder(hidden_dim)
        self.budget_head = nn.Linear(hidden_dim, num_budgets)
    
    def forward(self, problem):
        # 编码问题特征
        h = self.encoder(problem)
        
        # 预测预算分布
        logits = self.budget_head(h)
        probs = F.softmax(logits, dim=-1)
        
        return probs
```

#### 上下文特征

除了问题本身，还考虑：
- **问题长度**: 长问题可能需要更多预算
- **问题类型**: 数学 vs 创意写作
- **历史表现**: 类似问题的最优预算
- **当前负载**: 系统资源状态

## 伪代码实现

```python
class ORBIT:
    def __init__(self, model, budgets, exploration_coef=0.1):
        self.model = model
        self.budgets = budgets
        self.num_budgets = len(budgets)
        self.c = exploration_coef
        
        # 在线统计
        self.N = defaultdict(lambda: np.zeros(self.num_budgets))
        self.Q = defaultdict(lambda: np.zeros(self.num_budgets))
        self.t = 0

    def select_budget(self, problem):
        """基于 UCB 选择预算"""
        problem_id = self.hash_problem(problem)
        
        if problem_id not in self.N:
            # 首次见到，均匀探索
            return np.random.choice(self.num_budgets)
        
        n = self.N[problem_id]
        q = self.Q[problem_id]
        
        # UCB 分数
        ucb_scores = q + self.c * np.sqrt(np.log(self.t + 1) / (n + 1e-6))
        
        # 选择最大值
        budget_idx = np.argmax(ucb_scores)
        
        return budget_idx

    def reason(self, problem, budget_idx):
        """在给定预算下推理"""
        budget = self.budgets[budget_idx]
        
        # 根据预算类型执行不同推理策略
        if isinstance(budget, int):  # Token 预算
            output = self.model.generate(
                problem,
                max_tokens=budget,
                do_sample=True
            )
        elif isinstance(budget, dict):  # 计算资源预算
            output = self.model.generate_with_resource_limit(
                problem,
                **budget
            )
        else:
            output = self.model.generate(problem)
        
        return output

    def get_reward(self, problem, output, ground_truth=None):
        """获取结果质量"""
        if ground_truth is not None:
            # 有监督信号
            return self.compute_accuracy(output, ground_truth)
        else:
            # 无监督信号：使用自一致性或其他启发式
            return self.estimate_quality(output)

    def online_update(self, problem, budget_idx, reward):
        """在线更新策略"""
        problem_id = self.hash_problem(problem)
        
        # 更新计数
        self.N[problem_id][budget_idx] += 1
        
        # 更新 Q 值 (增量平均)
        n = self.N[problem_id][budget_idx]
        old_q = self.Q[problem_id][budget_idx]
        self.Q[problem_id][budget_idx] += (reward - old_q) / n
        
        # 更新策略网络
        if hasattr(self, 'policy_optimizer'):
            probs = self.policy_network(problem)
            log_prob = torch.log(probs[budget_idx])
            loss = -reward * log_prob
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
        
        self.t += 1

    def run_episode(self, problem, ground_truth=None):
        """运行一个完整 episode"""
        # 1. 选择预算
        budget_idx = self.select_budget(problem)
        
        # 2. 执行推理
        output = self.reason(problem, budget_idx)
        
        # 3. 获取奖励
        reward = self.get_reward(problem, output, ground_truth)
        
        # 4. 在线更新
        self.online_update(problem, budget_idx, reward)
        
        return output, reward
```

## 关键技术

### 1. 问题聚类

将相似问题分组，共享统计信息：

```python
class ProblemClusterer:
    def __init__(self, n_clusters=100):
        self.kmeans = KMeans(n_clusters=n_clusters)
    
    def fit(self, problems):
        embeddings = self.embed(problems)
        self.kmeans.fit(embeddings)
    
    def get_cluster(self, problem):
        embedding = self.embed([problem])
        return self.kmeans.predict(embedding)[0]
```

### 2. 迁移学习

新问题的预算选择可以基于相似问题：

```python
def transfer_budget_selection(problem, memory_bank, k=5):
    """从 k 个最近邻迁移预算选择"""
    neighbors = memory_bank.find_nearest(problem, k=k)
    
    # 加权平均邻居的 Q 值
    q_aggregated = np.zeros(num_budgets)
    total_weight = 0
    
    for neighbor, dist in neighbors:
        weight = 1 / (dist + 1e-6)
        q_aggregated += weight * neighbor.q_values
        total_weight += weight
    
    return np.argmax(q_aggregated / total_weight)
```

### 3. 自适应探索

动态调整探索系数：

```python
def adaptive_exploration(t, performance_history):
    """基于性能自适应调整探索"""
    if len(performance_history) < 10:
        return 0.5  # 高探索
    
    recent_perf = np.mean(performance_history[-10:])
    
    if recent_perf > 0.9:  # 性能很好
        return 0.01  # 低探索
    elif recent_perf < 0.5:  # 性能差
        return 1.0   # 高探索
    else:
        return 0.1   # 中等探索
```

## 应用场景

### 1. 动态 API 定价

```python
# 用户选择预算级别
user_budget = 'economy'  # economy, standard, premium

# ORBIT 在该预算内优化
budget_map = {
    'economy': 100,    # tokens
    'standard': 500,
    'premium': 2000
}

output = orbit.reason_with_budget(problem, budget_map[user_budget])
```

### 2. 实时系统

```python
# 根据延迟要求动态选择
latency_sla = 100  # ms

# 估算每预算的延迟
for budget_idx, budget in enumerate(budgets):
    est_latency = estimate_latency(budget)
    if est_latency <= latency_sla:
        # 在满足延迟的选项中选质量最高的
        candidates.append(budget_idx)

best_budget = max(candidates, key=lambda b: estimated_quality[b])
```

### 3. 边缘设备部署

```python
# 根据设备资源动态调整
device_resources = get_device_resources()

# 过滤可行预算
feasible_budgets = [
    b for b in budgets 
    if fits_resource_constraint(b, device_resources)
]

# 在可行选项中选择
budget_idx = orbit.select_from_feasible(problem, feasible_budgets)
```

## 实验结果

### 多预算推理质量

| 方法 | 低预算 | 中预算 | 高预算 | 平均 |
|------|-------|-------|-------|-----|
| 固定策略 | 45% | 72% | 85% | 67% |
| 启发式 | 52% | 75% | 83% | 70% |
| **ORBIT** | **68%** | **82%** | **88%** | **79%** |

### 计算效率

| 方法 | 平均 Token 数 | 准确率 | 效率 (Acc/Tokens) |
|------|-------------|-------|------------------|
| 始终高预算 | 1024 | 85% | 0.083 |
| 始终低预算 | 256 | 45% | 0.176 |
| **ORBIT** | **487** | **82%** | **0.168** |

## 与其他方法的对比

| 方法 | 在线学习 | 预算感知 | 自适应 | 应用场景 |
|------|---------|---------|-------|---------|
| Chain-of-Thought | ❌ | ❌ | ❌ | 固定流程 |
| Self-Consistency | ❌ | ❌ | ❌ | 质量优先 |
| MCTS | ✅ | ⚠️ | ✅ | 搜索空间 |
| **ORBIT** | ✅ | ✅ | ✅ | 预算约束 |

## 扩展变体

### ORBIT-V (Verification)

结合验证器的 ORBIT：

```python
def reason_with_verification(problem, budget):
    # 分配部分预算给验证
    reason_budget = int(budget * 0.7)
    verify_budget = int(budget * 0.3)
    
    # 生成多个候选
    candidates = []
    for _ in range(n_candidates):
        output = model.generate(problem, max_tokens=reason_budget)
        candidates.append(output)
    
    # 用验证预算选择最佳
    best = verifier.select_best(candidates, verify_budget)
    return best
```

### ORBIT-M (Multi-Step)

多步预算分配：

```python
def multi_step_orbit(problem, total_budget):
    remaining = total_budget
    context = problem
    
    for step in range(max_steps):
        # 分配此步预算
        step_budget = select_step_budget(context, remaining)
        
        # 执行推理
        output = model.generate(context, max_tokens=step_budget)
        
        # 检查是否完成
        if is_complete(output):
            break
        
        # 更新上下文和剩余预算
        context = extend_context(context, output)
        remaining -= step_budget
    
    return aggregate_outputs(context)
```

## 关键论文

- ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget Reasoning
- Budget-Aware Neural Machine Translation
- Adaptive Computation Time

## 资源

- **项目链接**: 参见相关论文
- **实现参考**: 可与 MCTS、Beam Search 框架集成

## 总结

ORBIT 代表了推理阶段在线学习的新方向，通过智能的预算分配和持续学习，在保证质量的同时优化计算效率。
