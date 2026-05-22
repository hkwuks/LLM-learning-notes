# CASCADE & Streaming Training 综述

## 概述

CASCADE 是一种**在线经验学习**框架，允许 LLM Agent 在部署任务流中进行持续学习，无需更新基础模型参数，显存占用低于 4GB。它是**流式/在线学习**的代表性方法。

## 核心问题

### 部署后学习的挑战

```
传统流程:
训练 → 部署 → 冻结 → 重新训练 → 重新部署

问题:
1. 重新训练成本高
2. 数据隐私问题
3. 时效性差
4. 计算资源受限
```

### 持续适应的需求

- **新工具出现**: API 更新、新服务
- **用户反馈**: 从实际交互中学习
- **环境变化**: 任务要求、约束条件
- **个性化**: 适应特定用户需求

## CASCADE 核心思想

### 关键洞察

**在推理时进行学习，不修改基础模型权重。**

通过维护一个**经验记忆库**，在推理时检索相关经验来改进决策。

### 系统架构

```
┌────────────────────────────────────────────┐
│              CASCADE Framework             │
├────────────────────────────────────────────┤
│                                            │
│   ┌──────────────┐    ┌──────────────┐    │
│   │   Task Flow  │───▶│   Experience │    │
│   │              │    │    Memory    │    │
│   └──────────────┘    └──────┬───────┘    │
│                              │            │
│                              ▼            │
│   ┌──────────────┐    ┌──────────────┐    │
│   │   LLM Agent  │◀───│   Retriever  │    │
│   │  (Frozen)    │    │              │    │
│   └──────┬───────┘    └──────────────┘    │
│          │                                 │
│          ▼                                 │
│   ┌──────────────┐                        │
│   │    Action    │                        │
│   │   (Update    │───────────────────────▶│
│   │   Memory)    │                        │
│   └──────────────┘                        │
│                                            │
└────────────────────────────────────────────┘
```

### 数学形式化

#### 经验表示

每个经验 $e$ 表示为：

$$e = (s, a, r, s', \text{context})$$

其中：
- $s$: 当前状态/上下文
- $a$: 执行的动作
- $r$: 获得的奖励
- $s'$: 后续状态
- $\text{context}$: 额外上下文信息

#### 检索增强推理

给定当前状态 $s$，检索相关经验：

$$E_{retrieved} = \text{TopK}(\{e \in \text{Memory} : \text{sim}(s, e.s) > \theta\})$$

推理时，将检索到的经验作为上下文：

$$a = \arg\max_a p_{LM}(a | s, E_{retrieved})$$

#### 在线更新

新经验 $e_{new}$ 直接加入记忆：

$$\text{Memory}_{t+1} = \text{Memory}_t \cup \{e_{new}\}$$

可选：遗忘旧经验或合并相似经验。

## 方法详解

### 经验记忆结构

#### 1. 向量存储

```python
class ExperienceMemory:
    def __init__(self, embedding_dim=768):
        self.embeddings = []  # 向量表示
        self.experiences = []  # 原始经验
        self.index = faiss.IndexFlatIP(embedding_dim)  # 快速检索
    
    def add(self, experience):
        # 编码经验
        embedding = self.encode(experience)
        
        # 添加到索引
        self.embeddings.append(embedding)
        self.experiences.append(experience)
        self.index.add(embedding.reshape(1, -1))
    
    def retrieve(self, query_state, k=5):
        # 编码查询
        query_emb = self.encode(query_state)
        
        # 检索最近邻
        distances, indices = self.index.search(
            query_emb.reshape(1, -1), k
        )
        
        return [self.experiences[i] for i in indices[0]]
```

#### 2. 分层组织

```python
class HierarchicalMemory:
    def __init__(self):
        # 短时记忆：最近的经验
        self.short_term = deque(maxlen=100)
        
        # 长时记忆：压缩后的经验
        self.long_term = ExperienceMemory()
        
        # 语义记忆：抽象知识
        self.semantic = KnowledgeGraph()
    
    def consolidate(self):
        """将短时记忆整合到长时记忆"""
        for exp in self.short_term:
            # 检查是否已存在相似经验
            similar = self.long_term.find_similar(exp)
            if similar:
                # 合并/更新
                self.long_term.update(similar, exp)
            else:
                # 添加新经验
                self.long_term.add(exp)
```

### 检索策略

#### 1. 密集检索

使用语义相似度：

```python
def dense_retrieve(query, memory, k=5):
    query_emb = embed(query)
    memory_embs = [embed(e) for e in memory]
    
    # 余弦相似度
    similarities = cosine_similarity(query_emb, memory_embs)
    top_k = np.argsort(similarities)[-k:]
    
    return [memory[i] for i in top_k]
```

#### 2. 稀疏检索

基于关键词匹配：

```python
def sparse_retrieve(query, memory, k=5):
    query_tokens = set(tokenize(query))
    
    scores = []
    for exp in memory:
        exp_tokens = set(tokenize(exp))
        # Jaccard 相似度
        score = len(query_tokens & exp_tokens) / len(query_tokens | exp_tokens)
        scores.append(score)
    
    top_k = np.argsort(scores)[-k:]
    return [memory[i] for i in top_k]
```

#### 3. 混合检索

```python
def hybrid_retrieve(query, memory, k=5, alpha=0.5):
    dense_scores = dense_scores(query, memory)
    sparse_scores = sparse_scores(query, memory)
    
    # 加权组合
    combined = alpha * dense_scores + (1 - alpha) * sparse_scores
    top_k = np.argsort(combined)[-k:]
    
    return [memory[i] for i in top_k]
```

### 推理流程

```python
class CASCADEAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.frozen = True  # 不更新 LLM 参数
    
    def act(self, state, task):
        # 1. 检索相关经验
        relevant_exps = self.memory.retrieve(state, k=5)
        
        # 2. 构建提示
        prompt = self.build_prompt(state, task, relevant_exps)
        
        # 3. LLM 推理
        action = self.llm.generate(prompt)
        
        return action
    
    def learn(self, state, action, reward, next_state):
        # 4. 在线学习：添加新经验
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': time.time()
        }
        
        self.memory.add(experience)
        
        # 可选：触发记忆整合
        if len(self.memory.short_term) >= 100:
            self.memory.consolidate()
```

## 伪代码实现

```python
class CASCADESystem:
    def __init__(self, base_model, memory_size=10000):
        self.base_model = base_model
        self.memory = {
            'experiences': [],
            'embeddings': None,
            'index': faiss.IndexFlatIP(768)
        }
        self.memory_size = memory_size
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def encode(self, text):
        """编码文本为向量"""
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors='pt')
            outputs = self.base_model(**tokens)
            # 使用 [CLS] 或平均池化
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy()
    
    def retrieve_experiences(self, query, k=5):
        """检索相关经验"""
        if len(self.memory['experiences']) == 0:
            return []
        
        query_emb = self.encode(query)
        distances, indices = self.memory['index'].search(query_emb, k)
        
        return [self.memory['experiences'][i] for i in indices[0]]
    
    def add_experience(self, experience):
        """添加新经验"""
        # 编码经验
        exp_text = self.format_experience(experience)
        exp_emb = self.encode(exp_text)
        
        # 添加到记忆
        self.memory['experiences'].append(experience)
        self.memory['index'].add(exp_emb)
        
        # 如果超出容量，移除最旧的经验
        if len(self.memory['experiences']) > self.memory_size:
            self.memory['experiences'].pop(0)
            # 重建索引 (或实现增量删除)
            self.rebuild_index()
    
    def run_task(self, task, max_steps=10):
        """执行任务并学习"""
        state = task.initial_state
        trajectory = []
        
        for step in range(max_steps):
            # 1. 检索相关经验
            relevant = self.retrieve_experiences(state)
            
            # 2. 构建上下文
            context = self.build_context(state, relevant)
            
            # 3. 生成动作
            action = self.generate_action(context)
            
            # 4. 执行动作
            next_state, reward, done = task.execute(action)
            
            # 5. 记录轨迹
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            # 6. 在线学习
            self.add_experience({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'task_type': task.type
            })
            
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def build_context(self, state, experiences):
        """构建包含经验的上下文"""
        prompt_parts = [
            "You are an AI assistant with memory of past experiences.",
            "\nRelevant past experiences:"
        ]
        
        for i, exp in enumerate(experiences, 1):
            prompt_parts.append(f"\n{i}. State: {exp['state']}")
            prompt_parts.append(f"   Action: {exp['action']}")
            prompt_parts.append(f"   Outcome: {exp['reward']}")
        
        prompt_parts.append(f"\nCurrent state: {state}")
        prompt_parts.append("What action should you take?")
        
        return "\n".join(prompt_parts)
```

## Streaming Training 扩展

### 在线 SFT

```python
class StreamingSFT:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer = deque(maxlen=buffer_size)
        self.optimizer = AdamW(model.parameters(), lr=1e-5)
    
    def on_data_arrival(self, example):
        """数据到达时立即处理"""
        self.buffer.append(example)
        
        # 小批量更新
        if len(self.buffer) >= self.batch_size:
            batch = list(self.buffer)[-self.batch_size:]
            self.update(batch)
    
    def update(self, batch):
        """执行一次更新"""
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 持续预训练

```python
class ContinualPretraining:
    def __init__(self, model):
        self.model = model
        self.ewc = EWC(model)  # 可选：使用 EWC 防止遗忘
    
    def train_on_stream(self, data_stream):
        for batch in data_stream:
            loss = self.compute_loss(batch)
            
            # 可选：添加 EWC 正则化
            if self.use_ewc:
                loss += self.ewc.penalty()
            
            loss.backward()
            self.optimizer.step()
```

## 实验结果

### CASCADE 性能

| 方法 | 任务成功率 | 显存占用 | 学习速度 |
|------|-----------|---------|---------|
| 基础模型 | 45% | 4GB | N/A |
| 重新训练 | 78% | 40GB+ | 慢 (小时级) |
| LoRA 微调 | 72% | 8GB | 中 (分钟级) |
| **CASCADE** | **75%** | **<4GB** | **快 (秒级)** |

### 与 RL 基线对比

在 12 个单轮任务上的结果：

| 方法 | 更好结果 | 接近结果 | 较差结果 |
|------|---------|---------|---------|
| REINFORCE+LoRA | 3 | 2 | 7 |
| **CASCADE** | **9** | **2** | **1** |

### 模型规模适应性

| 模型 | CASCADE 成功率 | 基线成功率 |
|------|---------------|-----------|
| Qwen3-4B | 73% | 42% |
| Qwen3-8B | 78% | 48% |
| Qwen3-14B | 82% | 51% |

## 关键技术优势

| 特性 | 传统微调 | CASCADE |
|------|---------|---------|
| 参数更新 | 是 | 否 |
| 显存需求 | 高 | 低 |
| 实时适应 | 否 | 是 |
| 隐私保护 | 需上传数据 | 本地处理 |
| 灾难性遗忘 | 是 | 否 |
| 可解释性 | 低 | 高 |

## 应用场景

### 1. 工具使用学习

```python
# 新 API 发布
def use_new_api(query):
    # CASCADE 从记忆中学习如何调用
    relevant = memory.retrieve(f"API call: {query}")
    
    # 基于历史经验生成调用
    action = llm.generate(prompt_with_experiences)
    
    # 执行并记录结果
    result = execute(action)
    memory.add_experience({'query': query, 'action': action, 'result': result})
```

### 2. 个性化助手

```python
# 适应用户偏好
class PersonalAssistant:
    def __init__(self, user_id):
        self.memory = load_user_memory(user_id)
    
    def respond(self, query):
        # 检索用户相关的历史交互
        relevant = self.memory.retrieve(query, filter_by_user=True)
        
        # 生成个性化回复
        response = generate_with_context(query, relevant)
        
        # 记录交互
        self.memory.add({'query': query, 'response': response})
```

### 3. 边缘设备部署

```python
# 资源受限环境
class EdgeAgent:
    def __init__(self):
        # 使用量化模型
        self.model = load_quantized_model('4bit')
        # 小型本地记忆
        self.memory = ExperienceMemory(max_size=1000)
```

## 局限与改进

### 当前局限

1. **记忆容量有限**: 经验数量受限于存储
2. **检索质量**: 依赖嵌入质量
3. **冷启动**: 初始经验不足时性能差
4. **长程依赖**: 难以关联时间上 distant 的经验

### 改进方向

```python
# 1. 记忆压缩
class CompressedMemory:
    def compress(self, experiences):
        """将多个经验压缩为抽象知识"""
        return self.llm.summarize(experiences)

# 2. 层次化检索
class HierarchicalRetrieval:
    def retrieve(self, query):
        # 先检索高层策略
        strategy = self.strategy_memory.retrieve(query)
        # 再检索具体经验
        examples = self.experience_memory.retrieve(query, strategy)
        return strategy, examples

# 3. 主动学习
class ActiveLearning:
    def should_query_human(self, state):
        """决定是否需要人类反馈"""
        uncertainty = self.compute_uncertainty(state)
        return uncertainty > self.threshold
```

## 相关方法对比

| 方法 | 学习方式 | 参数更新 | 记忆类型 | 适用场景 |
|------|---------|---------|---------|---------|
| SFT | 离线 | 是 | 无 | 固定任务 |
| RL | 在线 | 是 | 无 | 可交互环境 |
| **CASCADE** | 在线 | 否 | 经验 | 部署学习 |
| RAG | 在线 | 否 | 文档 | 知识问答 |
| In-context | 在线 | 否 | 无 | 少样本 |

## 关键论文

- CASCADE: Continual Learning via Experience Replay for LLM Agents
- Streaming Learning for Large Language Models
- Memory-Augmented Neural Networks

## 资源

- **实现参考**: 可与 LangChain、LlamaIndex 集成
- **向量数据库**: FAISS, Chroma, Pinecone
- **相关项目**: MemGPT, Voyager

## 总结

CASCADE 代表了 LLM 部署后学习的新范式，通过经验记忆和检索增强，在保持基础模型冻结的同时实现持续适应，特别适合资源受限和需要实时学习的场景。
